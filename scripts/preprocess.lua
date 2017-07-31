require 'pl'
require 'torch'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Preprocess a text file for training a language model.')
cmd:option('--input_text', 'data/tiny-shakespeare.txt', 'Input text file')
-- split output into multiple files
cmd:option('--train_t7', 'data/train-tiny-shakespeare.t7',
           'Output training data file in torch binary file')
cmd:option('--valid_t7', 'data/valid-tiny-shakespeare.t7',
           'Output validating data file in torch binary file')
cmd:option('--test_t7', 'data/test-tiny-shakespeare.t7',
           'Output testing data file in torch binary file')
cmd:option('--output_vocab', 'data/vocab-tiny-shakespeare.t7',
           'Output vocab in torch binary file')
cmd:option('--val_frac', 0.1, 'Validation fraction')
cmd:option('--test_frac', 0.1, 'Testing fraction')
cmd:option('--quiet', false, 'Disable all verbose outputs')
cmd:text()
opt = cmd:parse(arg or {})


-- First pass collect statistics and build vocab
char2index = {}
char_count = 0
vocab_count = 0
f = io.open(opt.input_text)
while true do
    line = f:read()
    if not line then break end
    for c in line:gmatch('.') do
        if not char2index[c] then
            vocab_count = vocab_count + 1
            char2index[c] = vocab_count
        end
        char_count = char_count + 1
    end
    -- new line
    char_count = char_count + 1
end
f:close()
-- XXX: hard code newline string
vocab_count = vocab_count + 1
char2index['\n'] = vocab_count
index2char = {}
-- create index to vocab map
for k, v in pairs(char2index) do index2char[v] = k end

-- compute split size
val_size = math.floor(opt.val_frac * char_count)
test_size = math.floor(opt.test_frac * char_count)
train_size = char_count - val_size - test_size

-- verbose
if not opt.quiet then
    print('Total vocabulary size: ' .. #index2char)
    print('Total tokens in file: ' .. char_count)
    print('  Training size: ' .. train_size)
    print('  Val size: ' .. val_size)
    print('  Test size: ' .. test_size)
end

train = torch.IntTensor(train_size)
valid = torch.IntTensor(val_size)
test = torch.IntTensor(test_size)
dataset = {train, valid, test}

-- second pass reading data to Tensor
split_idx, cur_idx = 1, 1
f = io.open(opt.input_text)
while true do
    line = f:read()
    if not line then break end
    -- XXX: Hard code new line
    line = line .. '\n'
    for c in line:gmatch('.') do
        -- some split has 0 size
        while dataset[split_idx]:size():size() == 0 do
            split_idx = split_idx + 1
        end
        dataset[split_idx][cur_idx] = char2index[c]
        cur_idx = cur_idx + 1
        if cur_idx > dataset[split_idx]:size(1) then
            split_idx = split_idx + 1
            cur_idx = 1
        end
    end
end
f:close()
-- save to file
torch.save(opt.train_t7, train)
if val_size > 0 then torch.save(opt.valid_t7, valid) end
if test_size > 0 then torch.save(opt.test_t7, test) end
torch.save(opt.output_vocab, index2char)