require 'torch'
require 'nn'

local gradcheck = require 'gradcheck'
require 'SequenceRNN'


local tests = {}
local tester = torch.Tester()


local function forwardTestFactory(N, T, D, H, dtype)
  dtype = dtype or 'torch.DoubleTensor'
  return function()
    local x = torch.randn(N, T, D):type(dtype)
    local h0 = torch.randn(N, H):type(dtype)
    local rnn = nn.SequenceRNN(D, H):type(dtype)

    local Wx = rnn.weight[{{1, D}}]:clone()
    local Wh = rnn.weight[{{D + 1, D + H}}]:clone()
    local b = rnn.bias:view(1, H):expand(N, H)
    local h_naive = torch.zeros(N, T, H):type(dtype)
    local prev_h = h0
    for t = 1, T do
      local a = torch.mm(x[{{}, t}], Wx)
      a = a + torch.mm(prev_h, Wh)
      a = a + b
      local next_h = torch.tanh(a)
      h_naive[{{}, t}] = next_h:clone()
      prev_h = next_h
    end

    local h = rnn:forward{h0, x}
    tester:assertTensorEq(h, h_naive, 1e-7)
  end
end

tests.forwardDoubleTest = forwardTestFactory(3, 4, 5, 6)
tests.forwardSingletonTest = forwardTestFactory(10, 1, 2, 3)
tests.forwardFloatTest = forwardTestFactory(3, 4, 5, 6, 'torch.FloatTensor')


function gradCheckTestFactory(N, T, D, H, dtype)
  dtype = dtype or 'torch.DoubleTensor'
  return function()
    local x = torch.randn(N, T, D)
    local h0 = torch.randn(N, H)

    local rnn = nn.SequenceRNN(D, H)
    local h = rnn:forward{h0, x}

    local dh = torch.randn(#h)

    rnn:zeroGradParameters()
    local dh0, dx = unpack(rnn:backward({h0, x}, dh))
    local dw = rnn.gradWeight:clone()
    local db = rnn.gradBias:clone()

    local function fx(x)   return rnn:forward{h0, x} end
    local function fh0(h0) return rnn:forward{h0, x} end

    local function fw(w)
      local old_w = rnn.weight
      rnn.weight = w
      local out = rnn:forward{h0, x}
      rnn.weight = old_w
      return out
    end

    local function fb(b)
      local old_b = rnn.bias
      rnn.bias = b
      local out = rnn:forward{h0, x}
      rnn.bias = old_b
      return out
    end

    local dx_num = gradcheck.numeric_gradient(fx, x, dh)
    local dh0_num = gradcheck.numeric_gradient(fh0, h0, dh)
    local dw_num = gradcheck.numeric_gradient(fw, rnn.weight, dh)
    local db_num = gradcheck.numeric_gradient(fb, rnn.bias, dh)

    local dx_error = gradcheck.relative_error(dx_num, dx)
    local dh0_error = gradcheck.relative_error(dh0_num, dh0)
    local dw_error = gradcheck.relative_error(dw_num, dw)
    local db_error = gradcheck.relative_error(db_num, db)

    tester:assert(dx_error < 1e-5)
    tester:assert(dh0_error < 1e-5)
    tester:assert(dw_error < 1e-5)
    tester:assert(db_error < 1e-5)
  end
end

tests.gradCheckTest = gradCheckTestFactory(2, 3, 4, 5)


function tests.scaleTest()
  local N, T, D, H = 4, 5, 6, 7
  local rnn = nn.SequenceRNN(D, H)
  rnn:zeroGradParameters()

  local h0 = torch.randn(N, H)
  local x = torch.randn(N, T, D)
  local dout = torch.randn(N, T, H)

  -- Run forward / backward with scale = 0
  rnn:forward{h0, x}
  rnn:backward({h0, x}, dout, 0)
  tester:asserteq(rnn.gradWeight:sum(), 0)
  tester:asserteq(rnn.gradBias:sum(), 0)

  -- Run forward / backward with scale = 2.0 and record gradients
  rnn:forward{h0, x}
  rnn:backward({h0, x}, dout, 2.0)
  local dw2 = rnn.gradWeight:clone()
  local db2 = rnn.gradBias:clone()

  -- Run forward / backward with scale = 4.0 and record gradients
  rnn:zeroGradParameters()
  rnn:forward{h0, x}
  rnn:backward({h0, x}, dout, 4.0)
  local dw4 = rnn.gradWeight:clone()
  local db4 = rnn.gradBias:clone()

  -- Gradients after the 4.0 step should be twice as big
  tester:assertTensorEq(torch.cdiv(dw4, dw2), torch.Tensor(#dw2):fill(2), 1e-6)
  tester:assertTensorEq(torch.cdiv(db4, db2), torch.Tensor(#db2):fill(2), 1e-6)
end


tester:add(tests)
tester:run()

