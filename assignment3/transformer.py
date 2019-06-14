from typing import Optional, List
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dropout, BatchNormalization, Lambda
import tensorflow.keras.backend as K

import transformer_layers
from transformer_layers import Stack, DenseStack, LayerNorm, EmbeddingTranspose, LayerDropout
from transformer_layers import WeightNormDense as Dense
from transformer_attention import MultiHeadAttention

class PositionEmbedding(Model):
    """
    Adds positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self) -> None:
        super(PositionEmbedding, self).__init__()

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        assert hidden_size % 2 == 0, 'Model vector size must be even for sinusoidal encoding'
        power = tf.range(0, hidden_size.value, 2,
                         dtype=tf.float32) / hidden_size.value
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def call(self, inputs, start=1):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        assert inputs.shape[-1] == self.hidden_size, 'Input final dim must match model hidden size'

        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]

        seq_pos = tf.cast(tf.range(start, sequence_length + start)[None, :], tf.float32)  # 1-index positions
        seq_pos_expanded = tf.expand_dims(seq_pos, axis=2)
        index = tf.tile(seq_pos_expanded, [1,1,tf.cast(self.hidden_size.value/2, dtype=tf.int32)]) / self.divisor

        sin_embedding = tf.sin(index)
        cos_embedding = tf.cos(index)

        position_embedding = tf.stack((sin_embedding, cos_embedding), -1)
        position_shape = (1, sequence_length, self.hidden_size)

        position_embedding = tf.reshape(position_embedding, position_shape)

        return inputs + position_embedding

class TransformerFeedForward(Model):
    def __init__(self, filter_size,
                 hidden_size,
                 dropout) -> None:
        super(TransformerFeedForward, self).__init__()
        self.norm = LayerNorm()
        self.feed_forward = DenseStack([filter_size, hidden_size], output_activation=None)
        self.dropout = Dropout(0 if dropout is None else dropout)

    def call(self, inputs):
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        dense_out = self.dropout(dense_out)
        return dense_out + inputs


class TransformerEncoderBlock(Model):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        super().__init__()
        self.norm = LayerNorm()
        self.self_attention = MultiHeadAttention(n_heads)
        self.feed_forward = TransformerFeedForward(filter_size, hidden_size, dropout)

    def call(self, inputs, self_attention_mask=None):

        ####################################  YOUR CODE HERE  ####################################
        # PART 4: Implement the Transformer Encoder & Decoder.

        # Perform a multi-headed self-attention across the inputs.

        # First normalize the input with the LayerNorm initialized in the __init__ function (self.norm)
        # Step 1
        norm_inputs = None

        # Apply the self-attention with the normalized input, use the self_attention mask as the optional mask parameter.
        # Using self.self_attention
        # Step 2
        attn = None

        # Apply the residual connection.
        # res_attn should sum the attention output and the original, non-normalized inputs
        # Step 3
        res_attn = None # Residual connection of the attention block

        # Apply the self.feed_forward to get the output of the EncoderBlock.
        # Step 4
        output = None
        return output


class TransformerDecoderBlock(Model):
    """A decoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: two Tensors encoder_outputs, decoder_inputs
                    encoder_outputs -> a Tensor with shape [batch_size, sequence_length, channels]
                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

    :return: output: Tensor with same shape as decoder_inputs
    """

    def __init__(self,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        super().__init__()
        self.self_norm = LayerNorm()
        self.self_attention = MultiHeadAttention(n_heads)

        self.cross_norm_source = LayerNorm()
        self.cross_norm_target = LayerNorm()
        self.cross_attention = MultiHeadAttention(n_heads)


        self.feed_forward = TransformerFeedForward(filter_size, hidden_size, dropout)

    def call(self, decoder_inputs, encoder_outputs, self_attention_mask=None, cross_attention_mask=None):    
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        ####################################  YOUR CODE HERE  ####################################
        # PART 4: Implement the Transformer Encoder & Decoder.

        # Step 1
        # Normalize the decoder's input using the self_norm
        norm_decoder_inputs = None

        # Step 2
        # Get the target self-attention using the self_attention, then apply the residual layer
        target_selfattn = None
        res_target_self_attn = None


        # Step 3
        # Normalize the output of the self-attention, as well as the encoder_outputs
        # using cross_norm_target and cross_norm_source
        norm_target_selfattn = None
        norm_encoder_outputs = None

        # Step 4
        # Apply the cross attention. Think about what the query and what the memory elements are
        # This is done using self.cross_attention, and use the mask we provide: cross_attention_mask, as the optional mask parameter
        encdec_attention = None

        # Step 5
        # The cross-attention is followed by another residual connection
        res_encdec_attention = None

        # Step 6
        # Apply the feed-foward layer to the output of the residual of the cross_attention
        output = None

        return output

class TransformerEncoder(Model):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self,
                 embedding_layer, n_layers, n_heads, d_model, d_filter, dropout = None) -> None:
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        # The encoding stack is a stack of transformer encoder blocks
        self.encoding_stack = Stack([TransformerEncoderBlock(n_heads, d_filter, d_model, dropout)
                                     for _ in range(n_layers)],
                                    name='encoder_stack')

    def call(self, inputs, encoder_mask=None):
        """
            Args:
                inputs: Either a float32 or in32 Tensor with shape [batch_size, sequence_length, ndim]
                encoder_mask: a boolean Tensor with shape [batch_size, sequence_length, sequence_length]
            Returns:
                output: a Tensor with shape [batch_size, sequence_length, d_model]
        """

        inputs = self.embedding_layer(inputs)
        output = self.encoding_stack(inputs, self_attention_mask=encoder_mask)

        return output


class TransformerDecoder(Model):
    """
        Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    def __init__(self,
                 embedding_layer,
                 output_layer,
                 n_layers,
                 n_heads,
                 d_model,
                 d_filter,
                 dropout = None) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        self.decoding_stack = Stack([TransformerDecoderBlock(n_heads, d_filter, d_model, dropout)
                                     for _ in range(n_layers)],
                                    name='decoder_blocks')
        self.output_layer = output_layer

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def call(self, target_input, encoder_output, encoder_mask=None, decoder_mask=None, mask_future=False,
        shift_target_sequence_right=False):
        """
            Args:
                inputs: a tuple of (encoder_output, target_embedding)
                    encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                    target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
                    cache: Used for fast decoding, a dictionary of tf.TensorArray. None during training.
                mask_future: a boolean for whether to mask future states in target self attention

            Returns:
                a tuple of (encoder_output, output)
                    output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        if shift_target_sequence_right:
            target_input = self.shift_target_sequence_right(target_input)

        target_embedding = self.embedding_layer(target_input)

        # Build the future-mask if necessary. This is an upper-triangular mask
        # which is used to prevent the network from attending to later timesteps
        # in the target embedding
        batch_size = tf.shape(target_embedding)[0]
        sequence_length = tf.shape(target_embedding)[1]
        self_attention_mask = self.get_self_attention_mask(batch_size, sequence_length, decoder_mask, mask_future)
        # Build the cross-attention mask. This is an upper-left block matrix which takes care of the masking
        # of the output shapes
        cross_attention_mask = self.get_cross_attention_mask(
            encoder_output, target_input, encoder_mask, decoder_mask)

        # Now actually do the decoding which should take us to the right dimension
        decoder_output = self.decoding_stack(target_embedding, encoder_outputs=encoder_output, self_attention_mask=self_attention_mask, cross_attention_mask=cross_attention_mask)

        # Use the output layer for the final output. For example, this will map to the vocabulary
        output = self.output_layer(decoder_output)
        return output

    def shift_target_sequence_right(self, target_sequence: tf.Tensor) -> tf.Tensor:
        constant_values = 0 if target_sequence.dtype in [tf.int32, tf.int64] else 1e-10
        pad_array = [[0, 0] for _ in target_sequence.shape]
        pad_array[1][0] = 1
        target_sequence = tf.pad(target_sequence, pad_array, constant_values=constant_values)[:, :-1]
        return target_sequence

    def get_future_mask(self, batch_size, sequence_length):
        """Mask future targets and padding

            :param batch_size: a TF Dimension
            :param sequence_length: a TF Dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask Tensor with shape [batch_size, sequence_length, sequence_length]
        """

        xind = tf.tile(tf.range(sequence_length)[None, :], (sequence_length, 1))
        yind = tf.tile(tf.range(sequence_length)[:, None], (1, sequence_length))
        mask = yind >= xind
        mask = tf.tile(mask[None], (batch_size, 1, 1))

        return mask

    def get_self_attention_mask(self, batch_size, sequence_length, decoder_mask, mask_future):
        if not mask_future:
            return decoder_mask
        elif decoder_mask is None:
            return self.get_future_mask(batch_size, sequence_length)
        else:
            return decoder_mask & self.get_future_mask(batch_size, sequence_length)

    # This is an upper left block matrix which masks the attention for things that don't
    # exist within the internals.
    def get_cross_attention_mask(self, encoder_output, decoder_input, encoder_mask, decoder_mask):
        if encoder_mask is None and decoder_mask is None:
            cross_attention_mask = None
        elif encoder_mask is None:
            # We need to not mask the encoding, but mask the decoding
            # The decoding mask should have shape [batch_size x target_len x target_len]
            # meaning all we have to do is pad the mask out properly
            cross_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                                (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
        elif decoder_mask is None:
            cross_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                                (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
        else:
            dec_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                              (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
            enc_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                              (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
            cross_attention_mask = tf.logical_and(enc_attention_mask, dec_attention_mask)

        return cross_attention_mask

class TransformerInputEmbedding(Model):

    def __init__(self,
                 embed_size,
                 vocab_size = None,
                 dropout = None,
                 batch_norm = False,
                 embedding_initializer=None) -> None:
        super().__init__()
        self.embedding_dense = Lambda(lambda x: x)
        self.using_dense_embedding = False
        self.embedding = Embedding(vocab_size, embed_size) # , weights=[embedding_initializer]

        self.position_encoding = PositionEmbedding()
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.batch_norm = None if batch_norm is False else BatchNormalization()

    def call(self, inputs, start=1):

        # Compute the actual embedding of the inputs by using the embedding layer
        embedding = self.embedding(inputs)
        embedding = self.dropout(embedding)

        if self.batch_norm:
            embedding = self.batch_norm(embedding)

        embedding = self.position_encoding(embedding, start=start)
        return embedding

class Transformer(Model):

    def __init__(self,
                 vocab_size = None,
                 n_layers = 6,
                 n_heads = 8,
                 d_model = 512,
                 d_filter = 2048,
                 dropout = None,
                 embedding_initializer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.dropout_weight = 0 if dropout is None else dropout

        input_embedding = TransformerInputEmbedding(d_model, vocab_size, dropout) # , embedding_initializer=embedding_initializer

        output_layer = EmbeddingTranspose(input_embedding.embedding)

        # Build the encoder stack.
        self.encoder = TransformerEncoder(input_embedding, n_layers, n_heads, d_model, d_filter, dropout)

        # Build the decoder stack.
        self.decoder = TransformerDecoder(input_embedding, output_layer, n_layers, n_heads, d_model, d_filter, dropout)

    def call(self, source_sequence, target_sequence, encoder_mask, decoder_mask, mask_future=True, shift_target_sequence_right=True):

        # Unpack the source and target sequences from the encoder.
        # Source Sequence: [batch_size x source_length]
        # Target Sequence: [batch_size x target_length]
        #
        # Generate the masks for the encoder and decoder. There are a lot of different ways that
        # the attention masks could be passed in, so this method handles a lot of these different
        # mask shapes.
        encoder_mask = transformer_layers.convert_to_attention_mask(source_sequence, encoder_mask)
        decoder_mask = transformer_layers.convert_to_attention_mask(target_sequence, decoder_mask)

        # After the end of the encoder and decoder generation phase, we have
        # Encoder Mask: [batch_size x source_length x source_length]
        # Decoder Mask: [batch_size x target_length x target_length]

        # Next, we perform the encoding of the sentence. This should take
        # as input a tensor of shape [batch_size x source_length x input_feature_shape]
        # and generate a tensor of shape [batch_size x source_length x d_model]

        ####################################  YOUR CODE HERE  ####################################
        # PART 5: Implement the full Transformer block

        # Part 1: Encode
        # Using the self.encoder, encode the source_sequence, and provide the encoder_mask variable as the optional mask.

        encoder_output = None

        # Part 2: Decode
        # Finally, we need to do a decoding this should generate a
        # tensor of shape [batch_size x target_length x d_model]
        # from the encoder output.

        # Using the self.decoder, provide it with the decoder input, and the encoder_output. 

        # As usual, provide it with the encoder and decoder_masks
        # Finally, You should also pass it these two optional arguments:
        # shift_target_sequence_right=shift_target_sequence_right, mask_future=mask_future
        decoder_output = None

        return decoder_output # We return the decoder's output