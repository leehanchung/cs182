from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(prev_h.dot(Wh) + x.dot(Wx) + b)
    cache = (prev_h, x, Wx, Wh, b, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    prev_h, x, Wx, Wh, b, next_h = cache

    dtanh = (1 - next_h**2) * dnext_h
    dx = dtanh.dot(Wx.T)
    dprev_h = dtanh.dot(Wh.T)
    dWx = x.T.dot(dtanh)
    dWh = prev_h.T.dot(dtanh)
    db = np.sum(dtanh, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    _, H = h0.shape
    cache = []
    h = np.zeros([N, T, H])

    # set initial hidden state to h0
    h_state = h0

    for t in range(T):
        # use initial hidden state, x, wx, wh, b to generate new hidden state
        h_state, cache_t = rnn_step_forward(x[:, t, :], h_state, Wx, Wh, b)
        cache.append(cache_t)
        # save new hidden state
        h[:, t, :] = h_state
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    prev_h, x, Wx, Wh, b, next_h = cache[0]
    _, D = x.shape

    dx = np.zeros([N, T, D])
    dh0 = np.zeros([N, H])
    dWx = np.zeros([D, H])
    dWh = np.zeros([H, H])
    db = np.zeros([H,])
    dprev_h_ = np.zeros_like(prev_h)

    for t in reversed(range(T)):
        # upgrade the hidden state gradients first since its the output
        dnext_h = dprev_h_ + dh[:, t, :]
        dx_, dprev_h_, dWx_, dWh_, db_ = rnn_step_backward(dnext_h, cache[t])
        dx[:, t, :] = dx_
        dWx += dWx_
        dWh += dWh_
        db += db_

    # set dh0 as the gradient output of the first unrolled rnn unit
    dh0 = dprev_h_
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # N, T = x.shape
    # V, D = W.shape
    #
    # sometimes looping is quite straightforward
    # out = np.zeros([N, T, D])
    # for n in range(N):
    #     for t in range(T):
    #         out[n, t, :] = W[x[n,t], :]

    # numpy array indexing magic.  use x as index matrix
    # https://docs.scipy.org/doc/numpy/user/basics.indexing.html
    out = W[x]
    cache = (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros_like(W)
    # N, T, D = dout.shape
    # V, _ = W.shape
    #
    # sometimes looping is quite straightforward
    # for n in range(N):
    #     for t in range(T):
    #         dW[x[n, t], :] += dout[n, t, :]

    # numpy magic with np.add.at
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html
    # https://stackoverflow.com/questions/45473896/np-add-at-indexing-with-array
    #
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################

    # # unpack and reshape parameters
    # N, D = x.shape
    # _, H = prev_h.shape
    # _b = b.reshape(4, H)
    # _Wx = Wx.reshape(D, 4, H)
    # _Wh = Wh.reshape(H, 4, H)
    #
    # # according to notebook setup, i, f, o, g gates
    # # i-> input gate, f -> forget gate, o -> output gate, g -> block input
    # i = sigmoid(prev_h.dot(_Wh[:,0,:]) + x.dot(_Wx[:,0,:]) + _b[0,:])
    # f = sigmoid(prev_h.dot(_Wh[:,1,:]) + x.dot(_Wx[:,1,:]) + _b[1,:])
    # o = sigmoid(prev_h.dot(_Wh[:,2,:]) + x.dot(_Wx[:,2,:]) + _b[2,:])
    # g = np.tanh(prev_h.dot(_Wh[:,3,:]) + x.dot(_Wx[:,3,:]) + _b[3,:])

    # while the above code works fine, its not as efficient.
    # using advices from the notebook, compute activation once and then split
    A = np.dot(prev_h, Wh) + np.dot(x, Wx) + b
    ai, af, ao, ag = np.split(A, 4, axis=1)
    i, f, o, g = sigmoid(ai), sigmoid(af), sigmoid(ao), np.tanh(ag)

    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_h, next_c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_h, next_c = cache

    # backprop: next_h = o * np.tanh(next_c)
    do = np.tanh(next_c) * dnext_h
    _dnext_c = o * (1 - np.tanh(next_c)**2) * dnext_h

    # two paths to dnext_c. so sum them together
    dnext_c = dnext_c + _dnext_c

    # backprop: next_c = f * prev_c + i * g
    dg = i * dnext_c
    di = g * dnext_c
    df = prev_c * dnext_c
    dprev_c = f * dnext_c

    # backprop: i, f, o, g = sigmoid(ai), sigmoid(af), sigmoid(ao), np.tanh(ag)
    # since we saved i, f, o, g in cache, use those values instead of calling
    # sigmoid/tanh again.
    dai = i * (1 - i) * di
    daf = f * (1 - f) * df
    dao = o * (1 - o) * do
    dag = (1 - g**2) * dg

    # reverse the split. ai, af, ao, ag = np.split(A, 4, axis=1)
    dA = np.concatenate((dai, daf, dao, dag), axis=1)

    # back prop: A = np.dot(prev_h, Wh) + np.dot(x, Wx) + b
    dx = dA.dot(Wx.T)
    dWx = x.T.dot(dA)
    dprev_h = dA.dot(Wh.T)
    dWh = prev_h.T.dot(dA)
    db = np.sum(dA, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    _, H = h0.shape
    cache = []
    h = np.zeros([N, T, H])
    c = np.zeros([N, T, H])
    # set initial hidden state to h0
    h_state = h0
    c_state = np.zeros_like(h0)

    for t in range(T):
        # use initial hidden state, x, wx, wh, b to generate new hidden state
        h_state, c_state, cache_t = lstm_step_forward(x[:, t, :], h_state, c_state, Wx, Wh, b)
        cache.append(cache_t)
        # save new hidden state
        c[:, t, :] = c_state
        h[:, t, :] = h_state

    cache.append(c)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_h, next_c = cache[0]
    N, T, H = dh.shape
    _, D = x.shape
    # grabbing cell state from cache
    # c = cache[-1]

    dx = np.zeros([N, T, D])
    dWx = np.zeros([D, 4*H])
    dWh = np.zeros([H, 4*H])
    db = np.zeros([4*H,])
    dprev_h_ = np.zeros_like(prev_h)
    # dprev_c_ = np.zeros_like(prev_h)
    # for some reason using c saved in forward pass doesnt give the right answer
    # so just initializing dnext_c equals
    dnext_c = np.zeros_like(prev_h)

    for t in reversed(range(T)):
        # upgrade the hidden state gradients first since its the output
        dnext_h = dprev_h_ + dh[:, t, :]
        # dnext_c = dprev_c_ + c[:, t, :]
        dx[:, t, :], dprev_h_, dnext_c, dWx_, dWh_, db_ = lstm_step_backward(dnext_h, dnext_c, cache[t])
        dWx += dWx_
        dWh += dWh_
        db += db_

    # set dh0 as the gradient output of the first unrolled rnn unit
    dh0 = dprev_h_
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
