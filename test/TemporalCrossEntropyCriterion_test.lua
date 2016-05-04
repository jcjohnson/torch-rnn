require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'

require 'TemporalCrossEntropyCriterion'


local tester = torch.Tester()
local tests = torch.TestSuite()


-- Run a nn.CrossEntropyCriterion explicitly over all minibatch elements
-- and timesteps, and make sure that we get the same results for both
-- loss and gradient.
function tests.naiveTest()
  local N, T, C = 2, 3, 4
  local crit = nn.TemporalCrossEntropyCriterion()
  
  local scores = torch.randn(N, T, C)
  local target = torch.Tensor(N, T):random(C + 1):add(-1):long()
  
  local loss = crit:forward(scores, target)
  local grad_scores = crit:backward(scores, target)
  
  local naive_crit = nn.CrossEntropyCriterion()
  local lsm = nn.LogSoftMax()
  local naive_losses = torch.zeros(N, T)
  local naive_grad = torch.zeros(N, T, C)
  for n = 1, N do
    for t = 1, T do
      if target[{n, t}] ~= 0 then
        local score_slice = scores[{n, t}]:view(1, C)
        local logprobs = lsm:forward(score_slice)
        local target_slice = torch.LongTensor{target[{n, t}]}
        naive_losses[{n, t}] = naive_crit:forward(score_slice, target_slice)
        naive_grad[{n, t}]:copy(naive_crit:backward(score_slice, target_slice))
      end
    end
  end
  
  if crit.batch_average then
    naive_losses:div(N)
    naive_grad:div(N)
  end
  if crit.time_average then
    naive_losses:div(T)
    naive_grad:div(T)
  end
  local naive_loss = naive_losses:sum()
  tester:assertTensorEq(naive_losses, crit.losses, 1e-5)
  tester:assertTensorEq(naive_grad, grad_scores, 1e-5)
  tester:assert(torch.abs(naive_loss - loss) < 1e-5)
end

-- Just make sure it runs, and that the sparsity patten in the
-- loss and gradient are correct.
function simpleTest(dtype)
  return function()
    torch.manualSeed(0)
    local N, T, C = 4, 5, 3
    local crit = nn.TemporalCrossEntropyCriterion():type(dtype)

    local scores = torch.randn(N, T, C):type(dtype)
    local target = torch.Tensor(N, T):random(C + 1):add(-1):type(dtype)

    local loss = crit:forward(scores, target)
    local grad_scores = crit:backward(scores, target)

    -- Make sure that all zeros in target give rise to zeros in the
    -- right place in crit.losses and grad_scores
    for n = 1, N do
      for t = 1, T do
        if target[{n, t}] == 0 then
          tester:assert(crit.losses[{n, t}] == 0)
          tester:assert(torch.all(torch.eq(grad_scores[{n, t}], 0)))
        end
      end
    end
    torch.seed()
  end
end

tests.simpleDoubleTest = simpleTest('torch.DoubleTensor')
tests.simpleFloatTest = simpleTest('torch.FloatTensor')
tests.simpleCudaTest = simpleTest('torch.CudaTensor')


tester:add(tests)
tester:run()
