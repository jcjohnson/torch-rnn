torch-rnn provides high-peformance, reusable RNN and LSTM modules. These modules have no dependencies other than torch and nn
and each lives in a single file, so they can easily be incorporated into other projects.

# VanillaRNN

```lua
rnn = nn.VanillaRNN(D, H)
```

The VanillaRNN module implements vanilla recurrent neural networks with a hyperbolic tangent nonlinearity.
It transforms a sequence of input vectors of dimension D into a sequence of hidden state vectors of dimension H.
It operates over sequences of length T and minibatches of size N; the sequence length and minibatch size can change on each
forward pass.

The output hidden states are computed using the recurrence relation ```h[t] = tanh(Wh h[t- 1] + Wx x[t] + b)```
where `Wx` is a matrix of input-to-hidden connections, `Wh` is a matrix of hidden-to-hidden connections, and `b` is a bias
term.

You can use a `VanillaRNN` instance in two different ways:

```
h = rnn:forward({h0, x})
{grad_h0, grad_x} = rnn:backward({h0, x}, grad_h)

h = rnn:forward(x)
grad_x = rnn:backward(x, grad_h)
```

`h0` is the initial hidden states, of shape `(N, H)` and `x` is the sequence of input vectors, of shape `(N, T, D)`.
The output `h` is the sequence of hidden states at each timestep, of shape `(N, T, H)`. In some applications, such as
image captioning, it is possible that the initial hidden state will be computed as the output of some other network.

By default, if `h0` is not provided on the forward pass then the initial hidden state will be set to zero. This behavior
might be useful for applications like sentiment analysis, where you want an RNN to process many independent sequences.

If `h0` is not provided and the instance variable `rnn.remember_states` is set to `true`, then the first call to
`rnn:forward` will set the initial hidden state to zero; on subsequent calls to forward, the final hidden state from the 
previous call will be used as the initial hidden state. This behavior is commonly used in language modeling,
where we want to train with very long (potentialy infinite) sequences, and compute gradients using truncated 
back-propagation through time. You cause the model to forget its hidden states by calling `rnn:resetStates()`; then the next call to `rnn:forward` will cause `h0` to be initialized to zeros.

These behaviors are all exercised in the [unit test for VanillaRNN](../test/VanillaRNN_test.lua).

# LSTM

# LanguageModel
