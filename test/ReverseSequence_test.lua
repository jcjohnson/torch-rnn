require 'torch'
require 'nn'

require 'ReverseSequence'

local tests = torch.TestSuite()
local tester = torch.Tester()

function tests.reverseSequenceTests()
    -- Test every dim 1,2 and 3.
    local input = torch.Tensor({{{1,2,3,4,5}, {6,7,8,9,10}}, {{11,12,13,14,15}, {16,17,18,19,20}}})
    local reverseSequence = nn.ReverseSequence(1)
    local expectedOutput = torch.Tensor({{{11,12,13,14,15}, {16,17,18,19,20}}, {{1,2,3,4,5}, {6,7,8,9,10}}})
    tester:assertTensorEq(reverseSequence:forward(input), expectedOutput, 0)
    local reverseSequence = nn.ReverseSequence(2)
    local expectedOutput = torch.Tensor({{{6,7,8,9,10}, {1,2,3,4,5}},{{16,17,18,19,20}, {11,12,13,14,15}}})
    tester:assertTensorEq(reverseSequence:forward(input), expectedOutput, 0)
    local reverseSequence = nn.ReverseSequence(3)
    local expectedOutput = torch.Tensor({{{5,4,3,2,1}, {10,9,8,7,6}}, {{15,14,13,12,11}, {20,19,18,17,16}}})
    tester:assertTensorEq(reverseSequence:forward(input), expectedOutput, 0)
    local reverseSequence = nn.ReverseSequence(3)
    local expectedOutput = torch.Tensor({{{5,4,3,2,1}, {10,9,8,7,6}}, {{15,14,13,12,11}, {20,19,18,17,16}}})
    -- Backwards should reverse the gradOutput.
    tester:assertTensorEq(reverseSequence:backward(nil, input), expectedOutput, 0)
end

tester:add(tests)
tester:run()