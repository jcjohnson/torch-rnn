require 'torch'
require 'nn'

require 'GRU'
local gradcheck = require 'util.gradcheck'
local tests = {}
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
  local x  = torch.randn(N, T, D)

  local gru = nn.GRU(D, H)
  local h = gru:forward{h0, x}

  -- Do a naive forward pass
  local naive_h = torch.Tensor(N, T, H)
  

  -- Unpack weight, bias for each gate
  local Wxu = gru.weight[{{1, D}, {1, H}}]
  local Wxr = gru.weight[{{1, D}, {H + 1, 2 * H}}]
  local Wxhc = gru.weight[{{1, D}, {2 * H + 1, 3 * H}}]
  
  
  local Whu = gru.weight[{{D + 1, D + H}, {1, H}}]
  local Whr = gru.weight[{{D + 1, D + H}, {H + 1, 2 * H}}]
  local Whhc = gru.weight[{{D + 1, D + H}, {2 * H + 1, 3 * H}}]
  
  
  local bu = gru.bias[{{1, H}}]:view(1, H):expand(N, H)
  local br = gru.bias[{{H + 1, 2 * H}}]:view(1, H):expand(N, H)
  local bhc = gru.bias[{{2 * H + 1, 3 * H}}]:view(1, H):expand(N, H)
  

  local prev_h = h0:clone()
  for t = 1, T do
    local xt = x[{{}, t}]
    local u = torch.sigmoid(torch.mm(xt, Wxu) + torch.mm(prev_h, Whu) + bu)
    local r = torch.sigmoid(torch.mm(xt, Wxr) + torch.mm(prev_h, Whr) + br)
    local hc = torch.tanh(torch.mm(xt, Wxhc) + torch.mm(torch.cmul(prev_h,r), Whhc) + bhc)
    local next_h = torch.cmul(hc, u) + prev_h - torch.cmul(prev_h, u)
    
    naive_h[{{}, t}] = next_h
    
    prev_h = next_h
  end

  tester:assertTensorEq(naive_h, h, 1e-10)
end


function tests.gradcheck()
  local N, T, D, H = 2, 3, 4, 5

  local x = torch.randn(N, T, D)
  local h0 = torch.randn(N, H)
  
  
  local gru = nn.GRU(D, H)
  local h = gru:forward{h0, x}

  local dh = torch.randn(#h)

  gru:zeroGradParameters()
  local dh0, dx = unpack(gru:backward({h0, x}, dh))
  local dw = gru.gradWeight:clone()
  local db = gru.gradBias:clone()

  local function fx(x)   return gru:forward{h0, x} end
  local function fh0(h0) return gru:forward{h0, x} end

  local function fw(w)
    local old_w = gru.weight
    gru.weight = w
    local out = gru:forward{ h0, x}
    gru.weight = old_w
    return out
  end

  local function fb(b)
    local old_b = gru.bias
    gru.bias = b
    local out = gru:forward{h0, x}
    gru.bias = old_b
    return out
  end

  local dx_num = gradcheck.numeric_gradient(fx, x, dh)
  local dh0_num = gradcheck.numeric_gradient(fh0, h0, dh)
  
  local dw_num = gradcheck.numeric_gradient(fw, gru.weight, dh)
  local db_num = gradcheck.numeric_gradient(fb, gru.bias, dh)

  local dx_error = gradcheck.relative_error(dx_num, dx)
  local dh0_error = gradcheck.relative_error(dh0_num, dh0)

  local dw_error = gradcheck.relative_error(dw_num, dw)
  local db_error = gradcheck.relative_error(db_num, db)

  tester:assertle(dh0_error, 1e-4)
  
  tester:assertle(dx_error, 1e-5)
  tester:assertle(dw_error, 1e-4)
  tester:assertle(db_error, 1e-5)
end


-- Make sure that everything works correctly when we don't pass an initial cell
-- state; in this case we do pass an initial hidden state and an input sequence
function tests.noCellTest()
  local N, T, D, H = 4, 5, 6, 7
  local gru = nn.GRU(D, H)

  for t = 1, 3 do
    local x = torch.randn(N, T, D)
    local h0 = torch.randn(N, H)
    local dout = torch.randn(N, T, H)

    local out = gru:forward{h0, x}
    local din = gru:backward({h0, x}, dout)

    tester:assert(torch.type(din) == 'table')
    tester:assert(#din == 2)
    check_size(din[1], {N, H})
    check_size(din[2], {N, T, D})

    -- Make sure the initial cell state got reset to zero
    --tester:assertTensorEq(gru.c0, torch.zeros(N, H), 0)
  end
end


-- Make sure that everything works when we don't pass initial hidden or initial
-- cell state; in this case we only pass input sequence of vectors
function tests.noHiddenTest()
  local N, T, D, H = 4, 5, 6, 7
  local gru = nn.GRU(D, H)

  for t = 1, 3 do
    local x = torch.randn(N, T, D)
    local dout = torch.randn(N, T, H)

    local out = gru:forward(x)
    local din = gru:backward(x, dout)

    tester:assert(torch.isTensor(din))
    check_size(din, {N, T, D})

    -- Make sure the initial cell state and initial hidden state are zero
    --tester:assertTensorEq(gru.c0, torch.zeros(N, H), 0)
    tester:assertTensorEq(gru.h0, torch.zeros(N, H), 0)
  end
end


function tests.rememberStatesTest()
  local N, T, D, H = 5, 6, 7, 8
  local gru = nn.GRU(D, H)
  gru.remember_states = true

  local final_h = nil
  for t = 1, 4 do
    local x = torch.randn(N, T, D)
    local dout = torch.randn(N, T, H)
    local out = gru:forward(x)
    local din = gru:backward(x, dout)

    if t == 1 then
      tester:assertTensorEq(gru.h0, torch.zeros(N, H), 0)
    elseif t > 1 then
      tester:assertTensorEq(gru.h0, final_h, 0)
    end
    final_h = out[{{}, T}]:clone()
  end

  -- Initial states should reset to zero after we call resetStates
  gru:resetStates()
  local x = torch.randn(N, T, D)
  local dout = torch.randn(N, T, H)
  gru:forward(x)
  gru:backward(x, dout)
  tester:assertTensorEq(gru.h0, torch.zeros(N, H), 0)
end


tester:add(tests)
tester:run()
