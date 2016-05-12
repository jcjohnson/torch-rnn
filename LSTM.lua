require 'torch'
require 'nn'


local layer, parent = torch.class('nn.LSTM', 'nn.Module')

--[[
If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SequenceLSTM stores this many
scalar values:

NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H

For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]

function layer:__init(input_dim, hidden_dim)
  parent.__init(self)

  local D, H = input_dim, hidden_dim
  self.input_dim, self.hidden_dim = D, H

  self.weight = torch.Tensor(D + H, 4 * H)
  self.gradWeight = torch.Tensor(D + H, 4 * H):zero()
  self.bias = torch.Tensor(4 * H)
  self.gradBias = torch.Tensor(4 * H):zero()
  self:reset()

  self.cell = torch.Tensor()    -- This will be (N, T, H)
  self.gates = torch.Tensor()   -- This will be (N, T, 4H)
  self.buffer1 = torch.Tensor() -- This will be (N, H)
  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (1, 4H)
  self.grad_a_buffer = torch.Tensor() -- This will be (N, 4H)

  self.h0 = torch.Tensor()
  self.c0 = torch.Tensor()
  self.remember_states = false

  self.grad_c0 = torch.Tensor()
  self.grad_h0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  self.bias:zero()
  self.bias[{{self.hidden_dim + 1, 2 * self.hidden_dim}}]:fill(1)
  self.weight:normal(0, std)
  return self
end


function layer:resetStates()
  self.h0 = self.h0.new()
  self.c0 = self.c0.new()
end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end


function layer:_unpack_input(input)
  local c0, h0, x = nil, nil, nil
  if torch.type(input) == 'table' and #input == 3 then
    c0, h0, x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 2 then
    h0, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return c0, h0, x
end


function layer:_get_sizes(input, gradOutput)
  local c0, h0, x = self:_unpack_input(input)
  local N, T = x:size(1), x:size(2)
  local H, D = self.hidden_dim, self.input_dim
  check_dims(x, {N, T, D})
  if h0 then
    check_dims(h0, {N, H})
  end
  if c0 then
    check_dims(c0, {N, H})
  end
  if gradOutput then
    check_dims(gradOutput, {N, T, H})
  end
  return N, T, D, H
end


--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input sequence, (N, T, D)

Output:
- h: Sequence of hidden states, (N, T, H)
--]]


function layer:updateOutput(input)
  self.recompute_backward = true
  local c0, h0, x = self:_unpack_input(input)
  local N, T, D, H = self:_get_sizes(input)

  self._return_grad_c0 = (c0 ~= nil)
  self._return_grad_h0 = (h0 ~= nil)
  if not c0 then
    c0 = self.c0
    if c0:nElement() == 0 or not self.remember_states then
      c0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.cell:size(1), self.cell:size(2)
      assert(prev_N == N, 'batch sizes must be constant to remember states')
      c0:copy(self.cell[{{}, prev_T}])
    end
  end
  if not h0 then
    h0 = self.h0
    if h0:nElement() == 0 or not self.remember_states then
      h0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.output:size(1), self.output:size(2)
      assert(prev_N == N, 'batch sizes must be the same to remember states')
      h0:copy(self.output[{{}, prev_T}])
    end
  end

  local bias_expand = self.bias:view(1, 4 * H):expand(N, 4 * H)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]

  local h, c = self.output, self.cell
  h:resize(N, T, H):zero()
  c:resize(N, T, H):zero()
  local prev_h, prev_c = h0, c0
  self.gates:resize(N, T, 4 * H):zero()
  for t = 1, T do
    local cur_x = x[{{}, t}]
    local next_h = h[{{}, t}]
    local next_c = c[{{}, t}]
    local cur_gates = self.gates[{{}, t}]
    cur_gates:addmm(bias_expand, cur_x, Wx)
    cur_gates:addmm(prev_h, Wh)
    cur_gates[{{}, {1, 3 * H}}]:sigmoid()
    cur_gates[{{}, {3 * H + 1, 4 * H}}]:tanh()
    local i = cur_gates[{{}, {1, H}}]
    local f = cur_gates[{{}, {H + 1, 2 * H}}]
    local o = cur_gates[{{}, {2 * H + 1, 3 * H}}]
    local g = cur_gates[{{}, {3 * H + 1, 4 * H}}]
    next_h:cmul(i, g)
    next_c:cmul(f, prev_c):add(next_h)
    next_h:tanh(next_c):cmul(o)
    prev_h, prev_c = next_h, next_c
  end

  return self.output
end


function layer:backward(input, gradOutput, scale)
  self.recompute_backward = false
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')
  local c0, h0, x = self:_unpack_input(input)
  if not c0 then c0 = self.c0 end
  if not h0 then h0 = self.h0 end

  local grad_c0, grad_h0, grad_x = self.grad_c0, self.grad_h0, self.grad_x
  local h, c = self.output, self.cell
  local grad_h = gradOutput

  local N, T, D, H = self:_get_sizes(input, gradOutput)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

  grad_h0:resizeAs(h0):zero()
  grad_c0:resizeAs(c0):zero()
  grad_x:resizeAs(x):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  local grad_next_c = self.buffer2:resizeAs(c0):zero()
  for t = T, 1, -1 do
    local next_h, next_c = h[{{}, t}], c[{{}, t}]
    local prev_h, prev_c = nil, nil
    if t == 1 then
      prev_h, prev_c = h0, c0
    else
      prev_h, prev_c = h[{{}, t - 1}], c[{{}, t - 1}]
    end
    grad_next_h:add(grad_h[{{}, t}])

    local i = self.gates[{{}, t, {1, H}}]
    local f = self.gates[{{}, t, {H + 1, 2 * H}}]
    local o = self.gates[{{}, t, {2 * H + 1, 3 * H}}]
    local g = self.gates[{{}, t, {3 * H + 1, 4 * H}}]
    
    local grad_a = self.grad_a_buffer:resize(N, 4 * H):zero()
    local grad_ai = grad_a[{{}, {1, H}}]
    local grad_af = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}]
    local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}]
    
    -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
    -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
    -- to compute grad_ao; the other values can be overwritten after we compute
    -- grad_next_c
    local tanh_next_c = grad_ai:tanh(next_c)
    local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
    local my_grad_next_c = grad_ao
    my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_next_h)
    grad_next_c:add(my_grad_next_c)
    
    -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
    -- that we can overwrite it.
    grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_next_h)

    -- Use grad_ai as a temporary buffer for computing grad_ag
    local g2 = grad_ai:cmul(g, g)
    grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

    -- We don't need any temporary storage for these so do them last
    grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
    grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)
    
    grad_x[{{}, t}]:mm(grad_a, Wx:t())
    grad_Wx:addmm(scale, x[{{}, t}]:t(), grad_a)
    grad_Wh:addmm(scale, prev_h:t(), grad_a)
    local grad_a_sum = self.buffer3:resize(1, 4 * H):sum(grad_a, 1)
    grad_b:add(scale, grad_a_sum)

    grad_next_h:mm(grad_a, Wh:t())
    grad_next_c:cmul(f)
  end
  grad_h0:copy(grad_next_h)
  grad_c0:copy(grad_next_c)

  if self._return_grad_c0 and self._return_grad_h0 then
    self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
  elseif self._return_grad_h0 then
    self.gradInput = {self.grad_h0, self.grad_x}
  else
    self.gradInput = self.grad_x
  end

  return self.gradInput
end


function layer:clearState()
  self.cell:set()
  self.gates:set()
  self.buffer1:set()
  self.buffer2:set()
  self.buffer3:set()
  self.grad_a_buffer:set()

  self.grad_c0:set()
  self.grad_h0:set()
  self.grad_x:set()
  self.output:set()
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


function layer:__tostring__()
  local name = torch.type(self)
  local din, dout = self.input_dim, self.hidden_dim
  return string.format('%s(%d -> %d)', name, din, dout)
end

