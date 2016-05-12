require 'torch'
require 'nn'


local layer, parent = torch.class('nn.VanillaRNN', 'nn.Module')

--[[
Vanilla RNN with tanh nonlinearity that operates on entire sequences of data.

The RNN has an input dim of D, a hidden dim of H, operates over sequences of
length T and minibatches of size N.

On the forward pass we accept a table {h0, x} where:
- h0 is initial hidden states, of shape (N, H)
- x is input sequence, of shape (N, T, D)

The forward pass returns the hidden states at each timestep, of shape (N, T, H).

SequenceRNN_TN swaps the order of the time and minibatch dimensions; this is
very slightly faster, but probably not worth it since it is more irritating to
work with.
--]]

function layer:__init(input_dim, hidden_dim)
  parent.__init(self)

  local D, H = input_dim, hidden_dim
  self.input_dim, self.hidden_dim = D, H
  
  self.weight = torch.Tensor(D + H, H)
  self.gradWeight = torch.Tensor(D + H, H)
  self.bias = torch.Tensor(H)
  self.gradBias = torch.Tensor(H)
  self:reset()

  self.h0 = torch.Tensor()
  self.remember_states = false

  self.buffer1 = torch.Tensor()
  self.buffer2 = torch.Tensor()
  self.grad_h0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.gradInput = {self.grad_h0, self.grad_x}
end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  self.bias:zero()
  self.weight:normal(0, std)
  return self
end


function layer:resetStates()
  self.h0 = self.h0.new()
end


function layer:_unpack_input(input)
  local h0, x = nil, nil
  if torch.type(input) == 'table' and #input == 2 then
    h0, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return h0, x
end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end


function layer:_get_sizes(input, gradOutput)
  local h0, x = self:_unpack_input(input)
  local N, T = x:size(1), x:size(2)
  local H, D = self.hidden_dim, self.input_dim
  check_dims(x, {N, T, D})
  if h0 then
    check_dims(h0, {N, H})
  end
  if gradOutput then
    check_dims(gradOutput, {N, T, H})
  end
  return N, T, D, H
end


--[[

Input: Table of
- h0: Initial hidden state of shape (N, H)
- x:  Sequence of inputs, of shape (N, T, D)

Output:
- h: Sequence of hidden states, of shape (N, T, H)
--]]
function layer:updateOutput(input)
  self.recompute_backward = true
  local h0, x = self:_unpack_input(input)
  local N, T, D, H = self:_get_sizes(input)
  self._return_grad_h0 = (h0 ~= nil)
  if not h0 then
    h0 = self.h0
    if h0:nElement() == 0 or not self.remember_states then
      h0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.output:size(1), self.output:size(2)
      assert(prev_N == N, 'batch sizes must be constant to remember states')
      h0:copy(self.output[{{}, prev_T}])
    end
  end

  local bias_expand = self.bias:view(1, H):expand(N, H)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  
  self.output:resize(N, T, H):zero()
  local prev_h = h0
  for t = 1, T do
    local cur_x = x[{{}, t}]
    local next_h = self.output[{{}, t}]
    next_h:addmm(bias_expand, cur_x, Wx)
    next_h:addmm(prev_h, Wh)
    next_h:tanh()
    prev_h = next_h
  end

  return self.output
end


-- Normally we don't implement backward, and instead just implement
-- updateGradInput and accGradParameters. However for an RNN, separating these
-- two operations would result in quite a bit of repeated code and compute;
-- therefore we'll just implement backward and update gradInput and
-- gradients with respect to parameters at the same time.
function layer:backward(input, gradOutput, scale)
  self.recompute_backward = false
  scale = scale or 1.0
  assert(scale == 1.0, 'scale must be 1')
  local N, T, D, H = self:_get_sizes(input, gradOutput)
  local h0, x = self:_unpack_input(input)
  if not h0 then h0 = self.h0 end
  local grad_h = gradOutput

  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

  local grad_h0 = self.grad_h0:resizeAs(h0):zero()
  local grad_x = self.grad_x:resizeAs(x):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  for t = T, 1, -1 do
    local next_h, prev_h = self.output[{{}, t}], nil
    if t == 1 then
      prev_h = h0
    else
      prev_h = self.output[{{}, t - 1}]
    end
    grad_next_h:add(grad_h[{{}, t}])
    local grad_a = grad_h0:resizeAs(h0)
    grad_a:fill(1):addcmul(-1.0, next_h, next_h):cmul(grad_next_h)
    grad_x[{{}, t}]:mm(grad_a, Wx:t())
    grad_Wx:addmm(scale, x[{{}, t}]:t(), grad_a)
    grad_Wh:addmm(scale, prev_h:t(), grad_a)
    grad_next_h:mm(grad_a, Wh:t())
    self.buffer2:resize(1, H):sum(grad_a, 1)
    grad_b:add(scale, self.buffer2)
  end
  grad_h0:copy(grad_next_h)

  if self._return_grad_h0 then
    self.gradInput = {self.grad_h0, self.grad_x}
  else
    self.gradInput = self.grad_x
  end

  return self.gradInput
end


function layer:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end


function layer:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end


function layer:clearState()
  self.buffer1:set()
  self.buffer2:set()
  self.grad_h0:set()
  self.grad_x:set()
  self.output:set()
end


function layer:__tostring__()
  local name = torch.type(self)
  local din, dout = self.input_dim, self.hidden_dim
  return string.format('%s(%d -> %d)', name, din, dout)
end

