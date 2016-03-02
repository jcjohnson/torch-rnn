require 'torch'
require 'nn'


local layer, parent = torch.class('nn.GRU', 'nn.Module')

--[[
If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SequenceGRU stores this many
scalar values:

NTD + 4NTH + 5NH + 6H^2 + 6DH + 7H

Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]

function layer:__init(input_dim, hidden_dim)
  parent.__init(self)

  local D, H = input_dim, hidden_dim
  self.input_dim, self.hidden_dim = D, H

  self.weight = torch.Tensor(D + H, 3 * H)
  self.gradWeight = torch.Tensor(D + H, 3 * H):zero()
  self.bias = torch.Tensor(3 * H)
  self.gradBias = torch.Tensor(3 * H):zero()
  self:reset()

  self.cell = torch.Tensor()    -- This will be (N, T, H)
  self.gates = torch.Tensor()   -- This will be (N, T, 3H)
  self.buffer1 = torch.Tensor() -- This will be (N, H)
  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (H,)
  self.grad_a_buffer = torch.Tensor() -- This will be (N, 3H)
  self.h0 = torch.Tensor()
  self.remember_states = false
  self.grad_h0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  --self.bias:zero()
  self.bias:normal(0,std) --self.bias[{{self.hidden_dim + 1, 2 * self.hidden_dim}}]:fill(1)
  self.weight:normal(0, std)
  return self
end


function layer:resetStates()
  self.h0 = self.h0.new()
end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
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
Input:
- h0: Initial hidden state, (N, H)
- x: Input sequence, (N, T, D)

Output:
- h: Sequence of hidden states, (N, T, H)
--]]


function layer:updateOutput(input)
  local h0, x = self:_unpack_input(input)
  local N, T, D, H = self:_get_sizes(input)

  self._return_grad_h0 = (h0 ~= nil)
  
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

  local bias_expand = self.bias:view(1, 3 * H):expand(N, 3 * H)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]

  local h = self.output
  h:resize(N, T, H):zero()
  local prev_h = h0
  self.gates:resize(N, T, 3 * H):zero()
  for t = 1, T do
    local cur_x = x[{{}, t}]
    local next_h = h[{{}, t}]
    local cur_gates = self.gates[{{}, t}]
    cur_gates:addmm(bias_expand, cur_x, Wx)
    cur_gates[{{}, {1, 2 * H}}]:addmm(prev_h, Wh[{{}, {1, 2 * H}}])
    cur_gates[{{}, {1, 2 * H}}]:sigmoid()
    
    local u = cur_gates[{{}, {1, H}}] --update gate : u = sig(Wx * x + Wh * prev_h + b)
    local r = cur_gates[{{}, {H + 1, 2 * H}}] --reset gate : r = sig(Wx * x + Wh * prev_h + b)
    next_h:cmul(r, prev_h) --temporary buffer : r . prev_h
    cur_gates[{{}, {2 * H + 1, 3 * H}}]:addmm(next_h, Wh[{{}, {2 * H + 1, 3 * H}}]) -- hc += Wh * r . prev_h
    local hc = cur_gates[{{}, {2 * H + 1, 3 * H}}]:tanh() --hidden candidate : hc = tanh(Wx * x + Wh * r . prev_h + b)
    next_h:addcmul(prev_h,-1, u, prev_h)
    next_h:addcmul(u,hc)  --next_h = (1-u) . prev_h + u . hc   
    prev_h = next_h
  end

  return self.output
end


function layer:backward(input, gradOutput, scale)
  scale = scale or 1.0
  local h0, x = self:_unpack_input(input)
  
  if not h0 then h0 = self.h0 end

  local grad_h0, grad_x = self.grad_h0, self.grad_x
  local h= self.output
  local grad_h = gradOutput

  local N, T, D, H = self:_get_sizes(input, gradOutput)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias

  grad_h0:resizeAs(h0):zero()
  
  grad_x:resizeAs(x):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  local temp_buffer = self.buffer2:resizeAs(h0):zero()
  for t = T, 1, -1 do
    local next_h= h[{{}, t}]
    local prev_h= nil
    if t == 1 then
      prev_h = h0
    else
      prev_h = h[{{}, t - 1}]
    end
    grad_next_h:add(grad_h[{{}, t}])

    local u = self.gates[{{}, t, {1, H}}]
    local r = self.gates[{{}, t, {H + 1, 2 * H}}]
    local hc = self.gates[{{}, t, {2 * H + 1, 3 * H}}]
    
    
    local grad_a = self.grad_a_buffer:resize(N, 3 * H):zero()
    local grad_au = grad_a[{{}, {1, H}}]
    local grad_ar = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ahc = grad_a[{{}, {2 * H + 1, 3 * H}}]
    
    -- We will use grad_au as temporary buffer
    -- to compute grad_ahc.
    
    local grad_hc = grad_au:fill(0):add(grad_next_h ):cmul(u)  
    grad_ahc:fill(1):addcmul(-1, hc,hc):cmul(grad_hc)
    local grad_r = grad_au:fill(0):addmm(grad_ahc, Wh[{{}, {2 * H + 1, 3 * H}}]:t() ):cmul(prev_h)
    grad_ar:fill(1):add(-1, r):cmul(r):cmul(grad_r)
    
    temp_buffer:fill(0):add(hc):add(-1, prev_h)
    grad_au:fill(1):add(-1, u):cmul(u):cmul(temp_buffer):cmul(grad_next_h)   
    grad_x[{{}, t}]:mm(grad_a, Wx:t())
    grad_Wx:addmm(scale, x[{{}, t}]:t(), grad_a)
    grad_Wh[{{}, {1, 2 * H}}]:addmm(scale, prev_h:t(), grad_a[{{}, {1, 2 * H}}])
    
    local grad_a_sum = self.buffer3:resize(H):sum(grad_a, 1)
    grad_b:add(scale, grad_a_sum)
    temp_buffer:fill(0):add(prev_h):cmul(r)
    grad_Wh[{{}, {2 * H + 1, 3 * H}}]:addmm(scale, temp_buffer:t(), grad_ahc)   
    grad_next_h:addcmul(-1, u, grad_next_h)
    grad_next_h:addmm(grad_a[{{}, {1, 2 * H}}], Wh[{{}, {1, 2 * H}}]:t())
    temp_buffer:fill(0):addmm(grad_a[{{}, {2 * H + 1, 3 * H}}], Wh[{{}, {2 * H + 1, 3 * H}}]:t()):cmul(r)
    grad_next_h:add(temp_buffer)
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
  self:backward(input, gradOutput, 0)
end


function layer:accGradParameters(input, gradOutput, scale)
  self:backward(input, gradOutput, scale)
end
