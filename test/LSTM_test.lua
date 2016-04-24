require 'torch'
require 'nn'

require 'LSTM'
local gradcheck = require 'util.gradcheck'


local tests = torch.TestSuite()
local tester = torch.Tester()


local function check_size(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end


function tests.testForward()
  local N, T, D, H = 3, 4, 5, 6

  local h0 = torch.randn(N, H)
  local c0 = torch.randn(N, H)
  local x  = torch.randn(N, T, D)

  local lstm = nn.LSTM(D, H)
  local h = lstm:forward{c0, h0, x}

  -- Do a naive forward pass
  local naive_h = torch.Tensor(N, T, H)
  local naive_c = torch.Tensor(N, T, H)

  -- Unpack weight, bias for each gate
  local Wxi = lstm.weight[{{1, D}, {1, H}}]
  local Wxf = lstm.weight[{{1, D}, {H + 1, 2 * H}}]
  local Wxo = lstm.weight[{{1, D}, {2 * H + 1, 3 * H}}]
  local Wxg = lstm.weight[{{1, D}, {3 * H + 1, 4 * H}}]
  
  local Whi = lstm.weight[{{D + 1, D + H}, {1, H}}]
  local Whf = lstm.weight[{{D + 1, D + H}, {H + 1, 2 * H}}]
  local Who = lstm.weight[{{D + 1, D + H}, {2 * H + 1, 3 * H}}]
  local Whg = lstm.weight[{{D + 1, D + H}, {3 * H + 1, 4 * H}}]
  
  local bi = lstm.bias[{{1, H}}]:view(1, H):expand(N, H)
  local bf = lstm.bias[{{H + 1, 2 * H}}]:view(1, H):expand(N, H)
  local bo = lstm.bias[{{2 * H + 1, 3 * H}}]:view(1, H):expand(N, H)
  local bg = lstm.bias[{{3 * H + 1, 4 * H}}]:view(1, H):expand(N, H)

  local prev_h, prev_c = h0:clone(), c0:clone()
  for t = 1, T do
    local xt = x[{{}, t}]
    local i = torch.sigmoid(torch.mm(xt, Wxi) + torch.mm(prev_h, Whi) + bi)
    local f = torch.sigmoid(torch.mm(xt, Wxf) + torch.mm(prev_h, Whf) + bf)
    local o = torch.sigmoid(torch.mm(xt, Wxo) + torch.mm(prev_h, Who) + bo)
    local g =    torch.tanh(torch.mm(xt, Wxg) + torch.mm(prev_h, Whg) + bg)
    local next_c = torch.cmul(prev_c, f) + torch.cmul(i, g)
    local next_h = torch.cmul(o, torch.tanh(next_c))
    naive_h[{{}, t}] = next_h
    naive_c[{{}, t}] = next_c
    prev_h, prev_c = next_h, next_c
  end

  tester:assertTensorEq(naive_h, h, 1e-10)
end


function tests.gradcheck()
  local N, T, D, H = 2, 3, 4, 5

  local x = torch.randn(N, T, D)
  local h0 = torch.randn(N, H)
  local c0 = torch.randn(N, H)
  
  local lstm = nn.LSTM(D, H)
  local h = lstm:forward{c0, h0, x}

  local dh = torch.randn(#h)

  lstm:zeroGradParameters()
  local dc0, dh0, dx = unpack(lstm:backward({c0, h0, x}, dh))
  local dw = lstm.gradWeight:clone()
  local db = lstm.gradBias:clone()

  local function fx(x)   return lstm:forward{c0, h0, x} end
  local function fh0(h0) return lstm:forward{c0, h0, x} end
  local function fc0(c0) return lstm:forward{c0, h0, x} end

  local function fw(w)
    local old_w = lstm.weight
    lstm.weight = w
    local out = lstm:forward{c0, h0, x}
    lstm.weight = old_w
    return out
  end

  local function fb(b)
    local old_b = lstm.bias
    lstm.bias = b
    local out = lstm:forward{c0, h0, x}
    lstm.bias = old_b
    return out
  end

  local dx_num = gradcheck.numeric_gradient(fx, x, dh)
  local dh0_num = gradcheck.numeric_gradient(fh0, h0, dh)
  local dc0_num = gradcheck.numeric_gradient(fc0, c0, dh)
  local dw_num = gradcheck.numeric_gradient(fw, lstm.weight, dh)
  local db_num = gradcheck.numeric_gradient(fb, lstm.bias, dh)

  local dx_error = gradcheck.relative_error(dx_num, dx)
  local dh0_error = gradcheck.relative_error(dh0_num, dh0)
  local dc0_error = gradcheck.relative_error(dc0_num, dc0)
  local dw_error = gradcheck.relative_error(dw_num, dw)
  local db_error = gradcheck.relative_error(db_num, db)

  tester:assertle(dh0_error, 1e-4)
  tester:assertle(dc0_error, 1e-5)
  tester:assertle(dx_error, 1e-5)
  tester:assertle(dw_error, 1e-4)
  tester:assertle(db_error, 1e-5)
end


-- Make sure that everything works correctly when we don't pass an initial cell
-- state; in this case we do pass an initial hidden state and an input sequence
function tests.noCellTest()
  local N, T, D, H = 4, 5, 6, 7
  local lstm = nn.LSTM(D, H)

  for t = 1, 3 do
    local x = torch.randn(N, T, D)
    local h0 = torch.randn(N, H)
    local dout = torch.randn(N, T, H)

    local out = lstm:forward{h0, x}
    local din = lstm:backward({h0, x}, dout)

    tester:assert(torch.type(din) == 'table')
    tester:assert(#din == 2)
    check_size(din[1], {N, H})
    check_size(din[2], {N, T, D})

    -- Make sure the initial cell state got reset to zero
    tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
  end
end


-- Make sure that everything works when we don't pass initial hidden or initial
-- cell state; in this case we only pass input sequence of vectors
function tests.noHiddenTest()
  local N, T, D, H = 4, 5, 6, 7
  local lstm = nn.LSTM(D, H)

  for t = 1, 3 do
    local x = torch.randn(N, T, D)
    local dout = torch.randn(N, T, H)

    local out = lstm:forward(x)
    local din = lstm:backward(x, dout)

    tester:assert(torch.isTensor(din))
    check_size(din, {N, T, D})

    -- Make sure the initial cell state and initial hidden state are zero
    tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
    tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
  end
end


function tests.rememberStatesTest()
  local N, T, D, H = 5, 6, 7, 8
  local lstm = nn.LSTM(D, H)
  lstm.remember_states = true

  local final_h, final_c = nil, nil
  for t = 1, 4 do
    local x = torch.randn(N, T, D)
    local dout = torch.randn(N, T, H)
    local out = lstm:forward(x)
    local din = lstm:backward(x, dout)

    if t == 1 then
      tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
      tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
    elseif t > 1 then
      tester:assertTensorEq(lstm.c0, final_c, 0)
      tester:assertTensorEq(lstm.h0, final_h, 0)
    end
    final_c = lstm.cell[{{}, T}]:clone()
    final_h = out[{{}, T}]:clone()
  end

  -- Initial states should reset to zero after we call resetStates
  lstm:resetStates()
  local x = torch.randn(N, T, D)
  local dout = torch.randn(N, T, H)
  lstm:forward(x)
  lstm:backward(x, dout)
  tester:assertTensorEq(lstm.c0, torch.zeros(N, H), 0)
  tester:assertTensorEq(lstm.h0, torch.zeros(N, H), 0)
end


-- If we want to use an LSTM to process a sequence, we have two choices: either
-- we run the whole sequence through at once, or we split it up along the time
-- axis and run the sequences through separately after setting remember_states
-- to true. This test checks that both choices give the same result.
function tests.rememberStatesTestV2()
  local N, T, D, H = 1, 12, 2, 3
  local lstm = nn.LSTM(D, H)

  local x = torch.randn(N, T, D)
  local x1 = x[{{}, {1, T / 3}}]:clone()
  local x2 = x[{{}, {T / 3 + 1, 2 * T / 3}}]:clone()
  local x3 = x[{{}, {2 * T / 3 + 1, T}}]:clone()

  local y = lstm:forward(x):clone()
  lstm.remember_states = true
  lstm:resetStates()
  local y1 = lstm:forward(x1):clone()
  local y2 = lstm:forward(x2):clone()
  local y3 = lstm:forward(x3):clone()

  local yy = torch.cat({y1, y2, y3}, 2)
  tester:assertTensorEq(y, yy, 0)
end


tester:add(tests)
tester:run()

