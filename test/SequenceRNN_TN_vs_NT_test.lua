require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'

require 'SequenceRNN'
require 'SequenceRNN_TN'


local tests = {}
local tester = torch.Tester()

-- Make sure that both compute the same thing
function tests.equalTest()
  local N, T, D, H = 2, 3, 4, 5

  local x_NT = torch.randn(N, T, D)
  local x_TN = x_NT:permute(2, 1, 3):clone()

  local h0 = torch.randn(N, H)
  local dout_NT = torch.randn(N, T, H)
  local dout_TN = dout_NT:permute(2, 1, 3):clone()

  local rnn_TN = nn.SequenceRNN_TN(D, H)
  local rnn_NT = nn.SequenceRNN(D, H)
  rnn_NT.weight = rnn_TN.weight:clone()
  rnn_NT.bias = rnn_TN.bias:clone()

  local out_TN = rnn_TN:forward{h0, x_TN}
  local out_NT = rnn_NT:forward{h0, x_NT}

  local out_diff = torch.abs(out_TN - out_NT:permute(2, 1, 3)):mean()
  assert(out_diff == 0)

  rnn_TN:zeroGradParameters()
  rnn_NT:zeroGradParameters()
  local dh0_TN, dx_TN = unpack(rnn_TN:backward({h0, x_TN}, dout_TN))
  local dh0_NT, dx_NT = unpack(rnn_NT:backward({h0, x_NT}, dout_NT))

  local dh0_diff = torch.abs(dh0_TN - dh0_NT):mean()
  local dx_diff = torch.abs(dx_TN - dx_NT:permute(2, 1, 3)):mean()
  local dw_diff = torch.abs(rnn_TN.weight - rnn_NT.weight):mean()
  local db_diff = torch.abs(rnn_TN.weight - rnn_NT.weight):mean()

  assert(dh0_diff == 0)
  assert(dx_diff == 0)
  assert(dw_diff == 0)
  assert(db_diff == 0)
end


local function speedTestFactory(N, T, D, H, dtype)
  return function()
    print ''
    local x_NT = torch.randn(N, T, D):type(dtype)
    local x_TN = x_NT:permute(2, 1, 3):clone()

    local h0 = torch.randn(N, H):type(dtype)

    local dout_NT = torch.randn(N, T, H):type(dtype)
    local dout_TN = dout_NT:permute(2, 1, 3):clone()

    local rnn_TN = nn.SequenceRNN_TN(D, H):type(dtype)
    local rnn_NT = nn.SequenceRNN(D, H):type(dtype)
    rnn_NT.weight = rnn_TN.weight:clone()
    rnn_NT.bias = rnn_TN.bias:clone()

    local num_trials = 10
    local timer = torch.Timer()
    local NT_forward_times = torch.Tensor(num_trials)
    local TN_forward_times = torch.Tensor(num_trials)
    local NT_backward_times = torch.Tensor(num_trials)
    local TN_backward_times = torch.Tensor(num_trials)
    for t = 1, num_trials do
      print(string.format('Running trial %d / %d', t, num_trials))
      -- TN forward
      cutorch.synchronize()
      timer:reset()
      rnn_TN:forward{h0, x_TN}
      cutorch.synchronize()
      TN_forward_times[t] = timer:time().real

      -- TN backward
      cutorch.synchronize()
      timer:reset()
      rnn_TN:backward({h0, x_TN}, dout_TN)
      cutorch.synchronize()
      TN_backward_times[t] = timer:time().real

      -- NT forward
      cutorch.synchronize()
      timer:reset()
      rnn_NT:forward{h0, x_NT}
      cutorch.synchronize()
      NT_forward_times[t] = timer:time().real

      -- NT backward
      cutorch.synchronize()
      timer:reset()
      rnn_NT:backward({h0, x_NT}, dout_NT)
      cutorch.synchronize()
      NT_backward_times[t] = timer:time().real
    end

    print('mean NT forward: ', NT_forward_times:mean())
    print('mean TN forward: ', TN_forward_times:mean())
    print('mean NT backward: ', NT_backward_times:mean())
    print('mean TN backward: ', TN_backward_times:mean())
  end
end


tests.cudaSpeedTest = speedTestFactory(100, 200, 1024, 1024, 'torch.CudaTensor')


tester:add(tests)
tester:run()
