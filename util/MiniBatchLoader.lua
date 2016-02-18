require 'torch'

local MiniBatchLoader = torch.class('MiniBatchLoader')

function MiniBatchLoader:__init(config)
    config = config or {}
    local args
    args, self.train_file, self.valid_file, self.test_file,
          self.batch_size, self.seq_length
        = xlua.unpack(
        {config},
        'MiniBatchLoader',
        'Load data files in torch binary format. Data will be cliped to fit mini batches',
        {arg='train_file', type='string', default='data/train-tiny-shakespeare.t7',
         help='training data in torch binary (see script/preprocess.lua)'},
        {arg='valid_file', type='string', default='data/valid-tiny-shakespeare.t7',
         help='training data in torch binary (see script/preprocess.lua)'},
        {arg='test_file', type='string', default='data/test-tiny-shakespeare.t7',
         help='training data in torch binary (see script/preprocess.lua)'},
        {arg='batch_size', type='number', default=8,
         help='number of sequences to run for each mini batch'},
        {arg='seq_length', type='number', default=6,
         help='number of characters for each sequence'}
    )
    self.x_splits = {}
    self.y_splits = {}
    self.split_sizes = {}
    local b, l = self.batch_size, self.seq_length
    self.x_splits['train'], self.y_splits['train'] = self:loadData(self.train_file, b, l)
    self.x_splits['val'], self.y_splits['val'] = self:loadData(self.valid_file, b, l)
    self.x_splits['test'], self.y_splits['test'] = self:loadData(self.test_file, b, l)
    self.split_sizes['train'] = self.x_splits['train']:size(1)
    self.split_sizes['val'] = self.x_splits['val']:size(1)
    self.split_sizes['test'] = self.x_splits['test']:size(1)
    self.split_idxs = {train=1, val=1, test=1}
    collectgarbage()
end

function MiniBatchLoader:loadData(file_path, b, l)
    local tensor = torch.load(file_path)
    local num = tensor:nElement()
    local extra = num % (b * l)
    -- Chop out the extra bits at the end to make it evenly divide
    -- Each batch will have a continuous stream of data
    local vx = tensor[{{1, num - extra}}]:view(b, -1, l)
    local vy = tensor[{{2, num - extra + 1}}]:view(b, -1, l)
    -- rearrage data so that the last two dimensions are B and L
    -- XXX: This is not very efficient.
    local vxx = torch.IntTensor(vx:size(2), vx:size(1), vx:size(3))
    local vyy = torch.IntTensor(vy:size(2), vy:size(1), vy:size(3))
    for i = 1, vyy:size(1) do
        vyy[i] = vy[{{}, i, {}}]
        vxx[i] = vx[{{}, i, {}}]
    end
    vxx:contiguous()
    vyy:contiguous()
    return vxx, vyy
end

function MiniBatchLoader:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'invalid split ' .. split)
  local x = self.x_splits[split][idx]
  local y = self.y_splits[split][idx]
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
  return x, y
end