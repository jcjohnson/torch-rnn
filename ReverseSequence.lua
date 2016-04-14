------------------------------------------------------------------------
--[[ ReverseSequence ]] --
-- Reverses a sequence on a given dimension.
-- Example: Given a tensor of torch.Tensor({{1,2,3,4,5}, {6,7,8,9,10})
-- nn.ReverseSequence(1):forward(tensor) would give: torch.Tensor({{6,7,8,9,10},{1,2,3,4,5}})
------------------------------------------------------------------------
local ReverseSequence, parent = torch.class("nn.ReverseSequence", "nn.Module")

function ReverseSequence:__init(dim)
    parent.__init(self)
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    assert(dim, "Must specify dimension to reverse sequence over")
    assert(dim <= 3, "Dimension has to be no greater than 3 (Only supports up to a 3D Tensor).")
    self.dim = dim
end

function ReverseSequence:reverseOutput(input)
    self.output:resizeAs(input)
    local indices = torch.LongTensor():resize(input:size())
    local T = input:size(1)
    for x = 1, T do
        indices:narrow(1, x, 1):fill(T - x + 1)
    end
    self.output = input:gather(1, indices)
end

function ReverseSequence:updateOutput(input)
    if (self.dim == 1) then
        self:reverseOutput(input)
    end
    if (self.dim == 2) then
        input = input:transpose(1, 2)
        self:reverseOutput(input)
        self.output = self.output:transpose(1, 2)
    end
    if (self.dim == 3) then
        input = input:transpose(1, 3)
        self:reverseOutput(input)
        self.output = self.output:transpose(1, 3)
    end
    return self.output
end

function ReverseSequence:reverseGradOutput(gradOutput)
    self.gradInput:resizeAs(gradOutput)
    local indices = torch.LongTensor():resize(gradOutput:size())
    local T = gradOutput:size(1)
    for x = 1, T do
        indices:narrow(1, x, 1):fill(T - x + 1)
    end
    self.gradInput = gradOutput:gather(1, indices)
end

function ReverseSequence:updateGradInput(inputTable, gradOutput)
    if (self.dim == 1) then
        self:reverseGradOutput(gradOutput)
    end
    if (self.dim == 2) then
        gradOutput = gradOutput:transpose(1, 2)
        self:reverseGradOutput(gradOutput)
        self.gradInput = self.gradInput:transpose(1, 2)
    end
    if (self.dim == 3) then
        gradOutput = gradOutput:transpose(1, 3)
        self:reverseGradOutput(gradOutput)
        self.gradInput = self.gradInput:transpose(1, 3)
    end
    return self.gradInput
end