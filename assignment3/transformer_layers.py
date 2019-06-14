import collections
from typing import Optional, Sequence, Any, Union, Callable

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Conv2D, Dropout, Conv2DTranspose, \
    BatchNormalization, Flatten, Activation, Embedding
import tensorflow.keras.backend as K  # pylint: disable=E0611

# https://github.com/keras-team/keras/issues/3878
class LayerNorm(Layer):
    """
    Does layer normalization from https://arxiv.org/abs/1607.06450.
    """

    def __init__(self, axis=-1, eps = 1e-6, **kwargs) -> None:
        if isinstance(axis, collections.Sequence):
            self.axis = axis
        else:
            self.axis = (axis,)
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = [input_shape[axis] for axis in self.axis]

        self.gamma = self.add_variable(name='gamma',
                                       shape=shape,
                                       initializer=tf.keras.initializers.Ones(),
                                       trainable=True)
        self.beta = self.add_variable(name='beta',
                                      shape=shape,
                                      initializer=tf.keras.initializers.Zeros(),
                                      trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        std = K.std(inputs, axis=self.axis, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

# I presume this is just how Sequential is added but at the moment Sequential
# requires input size to be specified at the begining


class Stack(Model):
    """
    A re-implementation of Keras's Sequential layer to work well with tf eager.
    """
    def __init__(self, layers = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self._call = None
        # self._layer_list = tf.contrib.checkpoint.List()
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        # self._layer_list.append(layer)
        self._layers.append(layer)

    def call(self, inputs, **kwargs):
        output = inputs
        for layer in self._layers:
            output = layer(output, **kwargs)
        return output

class DenseStack(Stack):
    """
    A stack of fully connected layers. Can do batch norm and specify an alternate output activation.
    """
    def __init__(self,
                 layers,
                 batch_norm = False,
                 activation = 'relu',
                 output_activation = None,
                 **kwargs) -> None:
        super().__init__()
        if layers is None:
            layers = []
        for _, layer in enumerate(layers[:-1]):
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(Dense(*layer, **kwargs))
            if batch_norm:
                self.add(BatchNormalization())
            self.add(Activation(activation))

        out_layer = layers[-1]
        if not isinstance(out_layer, collections.Iterable):
            out_layer = (out_layer,)
        self.add(Dense(*out_layer, **kwargs))
        if output_activation is not None:
            self.add(Activation(output_activation))

class LayerDropout(Model):
    """
    Optionally drops a full layer. Output is x with probability rate and f(x) with probability (1 - rate).

    Args:
        layer_call (Callable[[], Any]): Function that returns output of layer on inputs
        inputs (Any): What to return if the layer is dropped
        rate (float): Rate at which to drop layers

    Returns:
        Any: Either inputs or output of layer_call function.
    """

    def __init__(self, rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rate = rate

    def call(self, layer, inputs, *args, **kwargs):
        output = K.in_train_phase(
            K.switch(K.random_uniform([]) > self.rate, layer(inputs, *args, **kwargs), inputs),
            layer(inputs, *args, **kwargs))
        return output


class WeightNormDense(Dense):

    def build(self, input_shape):
        super().build(input_shape)
        self.scale = self.add_weight(
            'g',
            [self.units],
            initializer='ones',
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        # rank = common_shapes.rank(inputs)
        # print(rank)
        # if rank > 2:
        # Broadcasting is required for the inputs.
        outputs = standard_ops.tensordot(inputs, self.kernel, [[2], [0]])
        if not context.executing_eagerly():
            shape = inputs.get_shape().as_list()
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)
        # else:
        #     outputs = gen_math_ops.mat_mul(inputs, self.kernel)

        scale = self.scale / (tf.norm(self.kernel, 2, 0) + 1e-8)
        outputs = outputs * scale
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class EmbeddingTranspose(Model):
    """Multiply by the transpose of an embedding layer
    """
    def __init__(self, embedding_layer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = embedding_layer

    def call(self, inputs):
        embed_mat = self.embedding.weights[0]
        return K.dot(inputs, K.stop_gradient(K.transpose(embed_mat)))

class ApplyAttentionMask(Layer):
    """
    Applies a mask to the attention similarities.
    """
    def __init__(self, ):
        super().__init__()

    def call(self, similarity, mask=None):
        """
            Args:
                  similarity: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
                  mask: a Tensor with shape [batch_size, q/k_length, q/k_length]

            Returns:
                masked_similarity: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
        """
        if mask is None:
            return similarity

        similarity_rank_assert = tf.assert_rank_in(similarity, (3, 4))
        mask_rank_assert = tf.assert_rank(mask, 3)

        # There are so many different reasons a mask might be constructed a particular manner.
        # Because of this we don't want to infer a particular construction.
        with tf.control_dependencies([similarity_rank_assert, mask_rank_assert]):
            # If shapes don't match, then similarity has been split for multi-headed attention
            if len(mask.shape) != len(similarity.shape):
                similarity[:, 0].shape.assert_is_compatible_with(mask.shape)
                mask = mask[:, None]
            else:
                similarity.shape.assert_is_compatible_with(mask.shape)

            # We know that we're passing this through a softmax later, thus just add a relatively large negative
            # value to mask the output avoids a hadamard product (though I think that technically it's not
            # any more efficient to do it this way operations wise)
            bias = -1e9 * tf.cast(tf.logical_not(mask), tf.float32)
            masked_similarity = similarity + bias
            return masked_similarity

# Utility padding functions

def convert_padding_mask_to_attention_mask(sequence, padding_mask):
    """Given a padded input tensor of sequences and a boolean mask for each position
    in the sequence, returns a 3D boolean mask for use in attention.

    Args:
        sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length_1, ndim]
        padding_mask (tf.Tensor[bool]): Tensor of shape [batch_size, sequence_length_2]

    Returns:
        tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length_1, sequence_length_2]
    """
    batch_assert = tf.assert_equal(tf.shape(padding_mask)[0], tf.shape(sequence)[0],
                                   message='batch size mismatch between input sequence and  \
                                            padding_mask')
    rank_assert = tf.assert_equal(tf.rank(padding_mask), 2,
                                  message='Can only convert 2D position mask to 3D attention mask')

    with tf.control_dependencies([batch_assert, rank_assert]):
        attention_mask = tf.tile(padding_mask[:, None, :], (1, tf.shape(sequence)[1], 1))
        return attention_mask


def convert_sequence_length_to_sequence_mask(sequence, sequence_lengths):
    """Given a padded input tensor of sequences and a tensor of lengths, returns
    a boolean mask for each position in the sequence indicating whether or not
    that position is padding.

    Args:
        sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
        sequence_lengths (tf.Tensor[int]): Tensor of shape [batch_size]

    Returns:
        tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length]
    """
    batch_assert = tf.assert_equal(tf.shape(sequence_lengths)[0], tf.shape(sequence)[0],
                                   message='batch size mismatch between input sequence and  \
                                            sequence_lengths')
    rank_assert = tf.assert_equal(tf.rank(sequence_lengths), 1,
                                  message='Can only convert 1D sequence_lengths to 2D mask')

    with tf.control_dependencies([batch_assert, rank_assert]):
        indices = tf.tile(tf.range(tf.shape(sequence)[1])[None, :], (tf.shape(sequence_lengths)[0], 1))
        mask = indices < sequence_lengths[:, None]
        return mask


def convert_to_attention_mask(sequence, mask):
    """Automatically convert from None/1D/2D/3D mask to a boolean 3D attention mask.
    Note this does NOT allow for varying the input mask during training. We could replace
    the python if statements with tensorflow conditionals to allow this, but for the
    moment this is really a helper function and assumes that the type of mask
    passed in is fixed.

    Args:
        sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
        mask: Optional[Tensor] of shape [batch_size]
                                     or [batch_size, sequence_length]
                                     or [batch_size, sequence_length, sequence_length]

    Returns:
        Optional[tf.Tensor[bool]]: Tensor of shape [batch_size, sequence_length, sequence_length]
    """
    if mask is None:
        return None
    if len(mask.shape) == 1:
        mask = convert_sequence_length_to_sequence_mask(
            sequence, mask)
    if len(mask.shape) == 2:
        mask = convert_padding_mask_to_attention_mask(
            sequence, mask)
    if mask.dtype != tf.bool:
        mask = tf.cast(mask, tf.bool)
    return mask

__all__ = ['LayerNorm', 'Stack', 'DenseStack', 'PositionEmbedding', 'EmbeddingTranspose']