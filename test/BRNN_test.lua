require 'torch'
require 'nn'
require 'LSTM'
require 'ReverseSequence'
require 'BRNN'

torch.manualSeed(123)

local tests = torch.TestSuite()
local tester = torch.Tester()

function tests.BRNNTest()
    local fwd = nn.LSTM(5, 5)
    local bwd = nn.LSTM(5, 5)
    local brnn = nn.BRNN(fwd, bwd)

    local input = torch.rand(1, 5, 5)
    local output = brnn:forward(input)

    fwd:clearState()
    bwd:clearState()

    local fwdOutput = fwd:forward(input)

    local reverseSequence = nn.ReverseSequence(2)

    local reversedInput = reverseSequence:forward(input)
    local bwdOutput = bwd:forward(reversedInput)
    local bwdOutput = reverseSequence:forward(bwdOutput)

    local expectedOutput = torch.add(fwdOutput, bwdOutput)
    tester:assertTensorEq(expectedOutput, output, 0)
end

function tests.BRNNJoinTest()
    local fwd = nn.LSTM(5, 5)
    local bwd = nn.LSTM(5, 5)
    local brnn = nn.BRNN(fwd, bwd, nn.JoinTable(3)) -- Join on hidden dim.

    local input = torch.rand(1, 5, 5)
    local output = brnn:forward(input)

    fwd:clearState()
    bwd:clearState()

    local fwdOutput = fwd:forward(input)

    local reverseSequence = nn.ReverseSequence(2)

    local reversedInput = reverseSequence:forward(input)
    local bwdOutput = bwd:forward(reversedInput)
    local bwdOutput = reverseSequence:forward(bwdOutput)

    local expectedOutput = nn.JoinTable(3):forward({fwdOutput, bwdOutput})
    tester:assertTensorEq(expectedOutput, output, 0)
end

function tests.BRNNChangeDimTest()
    local dim = 1 -- Time is the first dimension (reversal dimension).
    local fwd = nn.LSTM(5, 5)
    local bwd = nn.LSTM(5, 5)
    local brnn = nn.BRNN(fwd, bwd, nn.JoinTable(3), dim)

    local input = torch.rand(5, 1, 5)
    local output = brnn:forward(input)

    fwd:clearState()
    bwd:clearState()

    local fwdOutput = fwd:forward(input)

    local reverseSequence = nn.ReverseSequence(dim)

    local reversedInput = reverseSequence:forward(input)
    local bwdOutput = bwd:forward(reversedInput)
    local bwdOutput = reverseSequence:forward(bwdOutput)

    local expectedOutput = nn.JoinTable(3):forward({fwdOutput, bwdOutput})
    tester:assertTensorEq(expectedOutput, output, 0)
end

tester:add(tests)
tester:run()