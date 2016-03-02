require 'torch'
require 'nn'

require 'TemporalAdapter'


local tests = {}
local tester = torch.Tester()


local function check_dims(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end


function tests.simpleTest()
  local D, H = 10, 20
  local N, T = 5, 6
  local mod = nn.TemporalAdapter(nn.Linear(D, H))
  local x = torch.randn(N, T, D)
  local y = mod:forward(x)
  check_dims(y, {N, T, H})
  local dy = torch.randn(#y)
  local dx = mod:backward(x, dy)
  check_dims(dx, {N, T, D})
end


tester:add(tests)
tester:run()

