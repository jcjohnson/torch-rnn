# Modules
torch-rnn provides high-peformance, reusable RNN and LSTM modules. These modules have no dependencies other than torch and 
nn and each lives in a single file, so they can easily be incorporated into other projects.

We also provide a LanguageModel module used for character-level language modeling; this is less reusable, but demonstrates 
that LSTM and RNN modules can be mixed with existing torch modules.

## VanillaRNN

```lua
rnn = nn.VanillaRNN(D, H)
```

[VanillaRNN](../VanillaRNN.lua) is a [torch nn.Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module)
subclass implementing a vanilla recurrent neural network with a hyperbolic tangent 
nonlinearity. It transforms a sequence of input vectors of dimension `D` into a sequence of hidden state vectors of 
dimension `H`. It operates over sequences of length `T` and minibatches of size `N`; the sequence length and minibatch size 
can change on  each forward pass.

Ignoring minibatches for the moment, a vanilla RNN computes the next hidden state vector `h[t]` (of shape (`H,)`) from the
previous hidden state `h[t - 1]` and the current input vector `x[t]` (of shape `(D,)`) using the recurrence relation

```
h[t] = tanh(Wh h[t- 1] + Wx x[t] + b)
```

where `Wx` is a matrix of input-to-hidden connections, `Wh` is a matrix of hidden-to-hidden connections, and `b` is a bias
term. The weights `Wx` and `Wh` are stored in a single Tensor `rnn.weight` of shape `(D + H, H)` and the bias `b` is
stored in a Tensor `rnn.bias` of shape `(H,)`.

You can use a `VanillaRNN` instance in two different ways:

```lua
h = rnn:forward({h0, x})
grad_h0, grad_x = unpack(rnn:backward({h0, x}, grad_h))

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

These behaviors are all exercised in the [unit test for VanillaRNN.lua](../test/VanillaRNN_test.lua).

As an implementation note, we implement `:backward` directly to compute both gradients with respect to inputs and 
accumulate gradients with respect to weights since these two operations share a lot of computation. We override 
`:updateGradInput` and `:accGradparameters` to call into `:backward`, so to avoid computing the same thing twice you
should call `:backward` directly rather than calling `:updateGradInput` and then `:accGradParameters`.

The file [VanillaRNN.lua](../VanillaRNN.lua) is standalone, with no dependencies other than torch and nn.

## LSTM
```lua
lstm = nn.LSTM(D, H)
```
An LSTM (short for Long Short-Term Memory) is a fancy type of recurrent neural network that is much more commonly used
than vanilla RNNs. Similar to the `VanillaRNN` above, [LSTM](../LSTM.lua) is a
[torch nn.Module](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module) subclass implementing an LSTM.
It transforms a sequence of input vectors  of dimension `D` into a sequence of hidden state vectors of dimension `H`; it 
operates over sequences of length `T` and minibatches of size `N`, which can be different on each forward pass.

An LSTM differs from a vanilla RNN in that it keeps track of both a *hidden state* and a *cell state* at each timestep.
Ignoring minibatches, the next hidden state vector `h[t]` (of shape `(H,)`) and cell state vector `c[t]` 
(also of shape `(H,)`) are computed from the previous hidden state `h[t - 1]`, previous cell
state `c[t - 1]`, and current input `x[t]` (of shape `(D,)`) using the following recurrence relation:

```
ai[t] = Wxi x[t] + Whi h[t - 1] + bi  # Matrix / vector multiplication
af[t] = Wxf x[t] + Whf h[t - 1] + bf  # Matrix / vector multiplication
ao[t] = Wxo x[t] + Who h[t - 1] + bo  # Matrix / vector multiplication
ag[t] = Wxg x[t] + Whg h[t - 1] + bg  # Matrix / vector multiplication

i[t] = sigmoid(ai[t])  # Input gate
f[t] = sigmoid(af[t])  # Forget gate
o[t] = sigmoid(ao[t])  # Output gate
g[t] = tanh(ag[t])     # Proposed update

c[t] = f[t] * c[t - 1] + i[t] * g[t]  # Elementwise multiplication of vectors
h[t] = o[t] * tanh(c[t])              # Elementwise multiplication of vectors
```

The input-to-hidden matrices `Wxi`, `Wxf`, `Wxo`, and `Wxg` along with the hidden-to-hidden matrices `Whi`, `Whf`, `Who`,
and `Whg` are stored in a single Tensor `lstm.weight` of shape `(D + H, 4 * H)`. The bias vectors `bi`, `bf`, `bo`, and
`bg` are stored in a single tensor `lstm.bias` of shape `(4 * H,)`.

You can use an `LSTM` instance in three different ways:

```lua
h = lstm:forward({c0, h0, x})
grad_c0, grad_h0, grad_x = unpack(lstm:backward({c0, h0, x}, grad_h))

h = lstm:forward({h0, x})
grad_h0, grad_x = unpack(lstm:backward({h0, x}, grad_h))

h = lstm:forward(x)
grad_x = lstm:backward(x, grad_h)
```

In all cases, `c0` is the initial cell state of shape `(N, H)`, `h0` is the initial hidden state of shape `(N, H)`,
`x` is the sequence of input vectors of shape `(N, T, D)`, and `h` is the sequence of output hidden states of shape
`(N, T, H)`.

If the initial cell state or initial hidden state are not provided, then by default they will be set to zero.

If the initial cell state or initial hidden state are not provided and the instance variable `lstm.remember_states`
is set to `true`, then the first call to `lstm:forward` will set the initial hidden and cell states to zero, and
subsequent calls to `lstm:forward` set the initial hidden and cell states equal to the final hidden and cell states
from the previous call, similar to the `VanillaRNN`. You can reset these initial cell and hidden states by calling
`lstm:resetStates()`; then the next call to `lstm:forward` will set the initial hidden and cell states to zero.

These behaviors are exercised in the [unit test for LSTM.lua](../test/LSTM_test.lua).

As an implementation note, we implement `:backward` directly to compute both gradients with respect to inputs and 
accumulate gradients with respect to weights since these two operations share a lot of computation. We override 
`:updateGradInput` and `:accGradparameters` to call into `:backward`, so to avoid computing the same thing twice you
should call `:backward` directly rather than calling `:updateGradInput` and then `:accGradParameters`.

The file [LSTM.lua](../LSTM.lua) is standalone, with no dependencies other than torch and nn.

## LanguageModel
```
model = nn.LanguageModel(kwargs)
```
[LanguageModel](../LanguageModel.lua) uses the above modules to implement a multilayer recurrent neural network language
model with dropout regularization. Since `LSTM` and `VanillaRNN` are `nn.Module` subclasses, we can implement a multilayer
recurrent neural network by simply stacking multiple instance in an `nn.Sequential` container.

`kwargs` is a table with the following keys:
- `idx_to_token`: A table giving the vocabulary for the language model, mapping integer ids to string tokens.
- `model_type`: "lstm" or "rnn"
- `wordvec_size`: Dimension for word vector embeddings
- `rnn_size`: Hidden state size for RNNs
- `num_layers`: Number of RNN layers to use
- `dropout`: Number between 0 and 1 giving dropout strength after each RNN layer
