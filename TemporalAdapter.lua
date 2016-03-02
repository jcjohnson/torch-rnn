require 'torch'
require 'nn'

--[[
A TemporalAdapter wraps a module intended to work on a minibatch of inputs
and allows you to use it on a minibatch of sequences of inputs.

The constructor accepts a module; we assume that the module operates
expects to receive a minibatch of inputs of shape (N, A) and produce a
minibatch of outputs of shape (N, B). The resulting TemporalAdapter then
expects inputs of shape (N, T, A) and returns outputs of shape (N, T, B),
applying the wrapped module at all timesteps.

TODO: Extend this to work with modules that want inputs of arbitrary
dimension; right now it can only wrap modules expecting a 2D input.
--]]

local layer, parent = torch.class('nn.TemporalAdapter', 'nn.Module')


function layer:__init(module)
  self.view_in = nn.View(1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)
  self.net = nn.Sequential()
  self.net:add(self.view_in)
  self.net:add(module)
  self.net:add(self.view_out)
end


function layer:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view_in:resetSize(N * T, -1)
  self.view_out:resetSize(N, T, -1)
  self.output = self.net:forward(input)
  return self.output
end


function layer:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

