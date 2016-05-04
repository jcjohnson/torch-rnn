require 'nn'

local crit, parent = torch.class('nn.TemporalCrossEntropyCriterion', 'nn.Criterion')

--[[
A TemporalCrossEntropyCriterion is used for classification tasks that occur
at every point in time for a timeseries; it works for minibatches and has a
null token that allows for predictions at arbitrary timesteps to be ignored.
This allows it to be used for sequence-to-sequence tasks where each minibatch
element has a different size; just pad the targets of the shorter sequences
with null tokens.

The criterion operates on minibatches of size N, with a sequence length of T,
with C classes over which classification is performed. The sequence length T
and the minibatch size N can be different on every forward pass.

On the forward pass we take the following inputs:
- input: Tensor of shape (N, T, C) giving classification scores for all C
  classes for every timestep of every sequence in the minibatch.
- target: Tensor of shape (N, T) where each element is an integer in the
  range [0, C]. If target[{n, t}] == 0 then the predictions at input[{n, t}]
  are ignored, and result in 0 loss and gradient; otherwise if
  target[{n, t}] = c then we expect that input[{n, t, c}] is the largest
  element of input[{n, t}], and compute loss and gradient in the same way as
  nn.CrossEntropyCriterion.

You can control whether loss is averaged over the minibatch N and sequence
length T by setting the instance variables crit.batch_average (default true)
and crit.time_average (default false).
--]]


function crit:__init()
  parent.__init(self)
  
  -- Set up a little net to compute LogSoftMax
  self.lsm = nn.Sequential()
  self.lsm:add(nn.View(1, 1, -1):setNumInputDims(3))
  self.lsm:add(nn.LogSoftMax())
  self.lsm:add(nn.View(1, -1):setNumInputDims(2))
  -- self.lsm = nn.Identity()
  
  -- Whether to average over space and batch
  self.batch_average = true
  self.time_average = false
  
  -- Intermediates
  self.grad_logprobs = torch.Tensor()
  self.losses = torch.Tensor()
end


function crit:clearState()
  self.lsm:clearState()
  self.grad_logprobs:set()
  self.losses:set()
end


-- Implementation note: We compute both loss and gradient in updateOutput, and
-- just return the gradient from updateGradInput.
function crit:updateOutput(input, target)
  local N, T, C = input:size(1), input:size(2), input:size(3)
  assert(target:dim() == 2 and target:size(1) == N and target:size(2) == T)
  self.lsm:get(1):resetSize(N * T, -1)
  self.lsm:get(3):resetSize(N, T, -1)
  
  -- For CPU tensors, target should be a LongTensor but for GPU tensors
  -- it should be the same type as input ... gross.
  if input:type() == 'torch.FloatTensor' or input:type() == 'torch.DoubleTensor' then
    target = target:long()
  end
  
  -- Figure out which elements are null. We want to use target as an index
  -- tensor for gather and scatter, so temporarily replace 0s with 1s.
  local null_mask = torch.eq(target, 0)
  target[null_mask] = 1
  
  -- Forward pass: compute losses and mask out null tokens
  local logprobs = self.lsm:forward(input)
  self.losses:resize(N, T, 1):gather(logprobs, 3, target:view(N, T, 1)):mul(-1)
  self.losses = self.losses:view(N, T)
  self.losses[null_mask] = 0
  
  -- Backward pass: Compute grad_logprobs
  self.grad_logprobs:resizeAs(logprobs):zero()
  self.grad_logprobs:scatter(3, target:view(N, T, 1), -1)
  self.grad_logprobs[null_mask:view(N, T, 1):expand(N, T, C)] = 0

  if self.batch_average then
    self.losses:div(N)
    self.grad_logprobs:div(N)
  end
  if self.time_average then
    self.losses:div(T)
    self.grad_logprobs:div(T)
  end
  self.output = self.losses:sum()
  self.gradInput = self.lsm:backward(input, self.grad_logprobs)
  
  target[null_mask] = 0
  return self.output
end


function crit:updateGradInput(input, target)
  return self.gradInput
end
