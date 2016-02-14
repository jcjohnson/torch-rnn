require 'torch'
require 'hdf5'

local utils = require 'utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(kwargs)
  local h5_file = utils.get_kwarg(kwargs, 'input_h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.timesteps = utils.get_kwarg(kwargs, 'num_timesteps')
  local N, T = self.batch_size, self.timesteps

  -- Just slurp all the data into memory
  local splits = {}
  local f = hdf5.open(h5_file, 'r')
  splits.train = f:read('/train'):all()
  splits.val = f:read('/val'):all()
  splits.test = f:read('/test'):all()

  for k, v in pairs(splits) do
    local num = v:nElement()
    local extra = num % (N * T)

    -- Chop out the extra bits at the end to make it evenly divide
    local vx = v[{{1, num - extra}}]:view(-1, N, T):contiguous()
    local vy = v[{{2, num - extra + 1}}]:view(-1, N, T):contiguous()

    self['x_' .. k] = vx
    self['y_' .. k] = vy
  end
end
