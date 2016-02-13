require 'torch'
require 'nn'


local layer, parent = torch.class('nn.SequenceRNN_TN', 'nn.Module')


function layer:__init(input_dim, hidden_dim)
  parent.__init(self)

  self.input_dim = input_dim
  self.hidden_dim = hidden_dim
  
  self.weight = torch.Tensor(input_dim + hidden_dim, hidden_dim)
  self.gradWeight = torch.Tensor(input_dim + hidden_dim, hidden_dim)
  self.bias = torch.Tensor(hidden_dim)
  self.gradBias = torch.Tensor(hidden_dim)
  self:reset()

  self.buffer1 = torch.Tensor()
  self.buffer2 = torch.Tensor()
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  self.bias:zero()
  self.weight:normal(0, std)
  return self
end


function layer:_get_sizes(input, gradOutput)
  local h0, x = input[1], input[2]
  assert(h0:dim() == 2)
  assert(x:dim() == 3)
  local N, H = h0:size(1), h0:size(2)
  local T, D = x:size(1), x:size(3)
  assert(x:size(2) == N)
  assert(H == self.hidden_dim)
  assert(D == self.input_dim)
  if gradOutput then
    assert(gradOutput:dim() == 3)
    assert(gradOutput:size(1) == T)
    assert(gradOutput:size(2) == N)
    assert(gradOutput:size(3) == H)
  end
  return N, T, D, H
end


--[[

Input: Table of
- h0: Initial hidden state of shape (N, H)
- x:  Sequence of inputs, of shape (T, N, D)

Output:
- h: Sequence of hidden states, of shape (T, N, H)
--]]
function layer:updateOutput(input)
  local h0, x = input[1], input[2]

  local N, T, D, H = self:_get_sizes(input)

  local bias_expand = self.bias:view(1, H):expand(N, H)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  
  self.output:resize(T, N, H):zero()
  local prev_h = h0
  for t = 1, T do
    local cur_x = x[t]
    local next_h = self.output[t]
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
  scale = scale or 1.0
  local h0, x = input[1], input[2]
  local grad_h = gradOutput

  local N, T, D, H = self:_get_sizes(input, gradOutput)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

  local grad_h0 = self.gradInput[1]:resizeAs(h0):zero()
  local grad_x = self.gradInput[2]:resizeAs(x):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  for t = T, 1, -1 do
    local next_h, prev_h = self.output[t], nil
    if t == 1 then prev_h = h0 else prev_h = self.output[t - 1] end
    grad_next_h:add(grad_h[t])
    local grad_a = grad_h0:resizeAs(h0)
    grad_a:fill(1):addcmul(-1.0, next_h, next_h):cmul(grad_next_h)
    grad_x[t]:mm(grad_a, Wx:t())
    grad_Wx:addmm(scale, x[t]:t(), grad_a)
    grad_Wh:addmm(scale, prev_h:t(), grad_a)
    grad_next_h:mm(grad_a, Wh:t())
    self.buffer2:resize(H):sum(grad_a, 1)
    grad_b:add(scale, self.buffer2)
  end
  grad_h0:copy(grad_next_h)
  return self.gradInput
end


function layer:updateGradInput(input, gradOutput)
  return self:updateGradInput(input, gradOutput, 0)
end


function layer:accGradParameters(input, gradOutput, scale)
  self:backward(input, gradOutput, scale)
end

