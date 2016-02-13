local utils = {}


--[[
Utility function to check that a Tensor has a specific shape.

Inputs:
- x: A Tensor object
- dims: A list of integers
--]]
function utils.check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end


return utils
