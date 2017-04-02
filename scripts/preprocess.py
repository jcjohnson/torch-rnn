# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import h5py
import codecs


parser = argparse.ArgumentParser()
parser.add_argument('--input_txt', default='data/tiny-shakespeare.txt')
parser.add_argument('--output_h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--output_json', default='data/tiny-shakespeare.json')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--encoding', default='utf-8')
args = parser.parse_args()


if __name__ == '__main__':
  if args.encoding == 'bytes': args.encoding = None

  # First go the file once to see how big it is and to build the vocab
  token_to_idx = {}
  total_size = 0
  with codecs.open(args.input_txt, 'r', args.encoding) as f:
    for line in f:
      total_size += len(line)
      for char in line:
        if char not in token_to_idx:
          token_to_idx[char] = len(token_to_idx) + 1

  # Now we can figure out the split sizes
  val_size = int(args.val_frac * total_size)
  test_size = int(args.test_frac * total_size)
  train_size = total_size - val_size - test_size
 
  if not args.quiet:
    print 'Total vocabulary size: %d' % len(token_to_idx)
    print 'Total tokens in file: %d' % total_size
    print '  Training size: %d' % train_size
    print '  Val size: %d' % val_size
    print '  Test size: %d' % test_size

  # Choose the datatype based on the vocabulary size
  dtype = np.uint8
  if len(token_to_idx) > 255:
    dtype = np.uint32
  if not args.quiet:
    print 'Using dtype ', dtype

  # Create, fill, and store each dataset,
  # one at a time to save memory
  with codecs.open(args.input_txt, 'r', args.encoding) as f:
    with h5py.File(args.output_h5, 'w') as h:
      def fill_and_store(arr_size, set_name):
          """Create a one-dimensional numpy array
          of the given size, fill it,
          and write the result to h under the given name.

          Leaves the source file advanced as far
          as it had to go to fill the array.

          If the remaining part of the file is shorter
          than arr_size, the remainder of the array is
          filled with zeroes.
          """
          arr = np.zeros(arr_size, dtype=dtype)
          for idx in xrange(arr_size):
              char = f.read(1)
              if not char:
                  break
              arr[idx] = token_to_idx[char]

          h.create_dataset(set_name, data=arr)

      fill_and_store(train_size, 'train')
      fill_and_store(val_size, 'val')
      fill_and_store(test_size, 'test')

  # For 'bytes' encoding, replace non-ascii characters so the json dump
  # doesn't crash
  if args.encoding is None:
    new_token_to_idx = {}
    for token, idx in token_to_idx.iteritems():
      if ord(token) > 127:
        new_token_to_idx['[%d]' % ord(token)] = idx
      else:
        new_token_to_idx[token] = idx
    token_to_idx = new_token_to_idx

  # Dump a JSON file for the vocab
  json_data = {
    'token_to_idx': token_to_idx,
    'idx_to_token': {v: k for k, v in token_to_idx.iteritems()},
  }
  with open(args.output_json, 'w') as f:
    json.dump(json_data, f)
