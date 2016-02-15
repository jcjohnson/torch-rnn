local cjson = require 'cjson'

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
    local msg = 'Expected %d, got %d'
    assert(x:size(i) == d, string.format(msg, d, x:size(i)))
  end
end


function utils.get_kwarg(kwargs, name, default)
  if kwargs == nil then kwargs = {} end
  if kwargs[name] == nil and default == nil then
    assert(false, string.format('"%s" expected and not given', name))
  elseif kwargs[name] == nil then
    return default
  else
    return kwargs[name]
  end
end


function utils.get_size(obj)
  local size = 0
  for k, v in pairs(obj) do size = size + 1 end
  return size
end


function utils.read_json(path)
  local f = io.open(path, 'r')
  local s = f:read('*all')
  f:close()
  return cjson.decode(s)
end


function utils.write_json(path, obj)
  local s = cjson.encode(obj)
  local f = io.open(path, 'w')
  f:write(s)
  f:close()
end



return utils
