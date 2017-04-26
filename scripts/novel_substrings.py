from __future__ import print_function

import argparse
import six

"""
Check how many substrings in sampled text are novel, not appearing in training
text. For different substring lengths, prints the fraction of sampled substrings
of that lenght that are novel.
"""

parser = argparse.ArgumentParser()
parser.add_argument('sampled_text')
parser.add_argument('training_text')
args = parser.parse_args()


with open(args.sampled_text, 'r') as f:
  s1 = f.read()

with open(args.training_text, 'r') as f:
  s2 = f.read()

for L in six.moves.range(1, 50):
  num_searched = 0
  num_found = 0
  for i in six.moves.range(len(s1) - L + 1):
    num_searched += 1
    sub = s1[i:(i+L)]
    assert len(sub) == L
    if sub in s2:
      num_found += 1
  novel_frac = (num_searched - num_found) / float(num_searched)
  print(L, novel_frac)
