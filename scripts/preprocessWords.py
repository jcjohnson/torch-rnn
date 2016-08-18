# -*- coding: utf-8 -*-

import argparse, json, os, codecs, h5py, re, string
from unidecode import unidecode
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_txt', default='data/tiny-shakespeare.txt')
parser.add_argument('--input_folder', default='')
parser.add_argument('--output_h5', default='data/tiny-shakespeare.h5')
parser.add_argument('--output_json', default='data/tiny-shakespeare.json')
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--case_sensitive', action='store_true')
parser.add_argument('--min_occurrences',type=int,default=20)
parser.add_argument('--min_documents', type=int,default=1)
parser.add_argument('--use_ascii', action='store_true')
parser.add_argument('--encoding', default='utf-8')
args = parser.parse_args()

if __name__ == '__main__':

    if args.encoding == 'bytes': args.encoding = None

    # Build list of files
    infiles = []
    if args.input_folder != '':
        infiles = [os.path.join(args.input_folder,item) for item in os.listdir(args.input_folder) if item[-4:]=='.txt']
    else:
        infiles = [args.input_txt]

    # Sanity check, words can't be in more documents than there are in the corpus
    if args.min_documents > len(infiles):
        args.min_documents = len(infiles)

    # Regex to split on
    regex = '(\W)'

    # List of all words found, form is word:[number of total occurrences, number of files]
    wordlist = {}

    # Build a unified array of tokens
    unified = []

    # Loop through input files
    for inpath in infiles:

        # Open the file and read into string
        infile = codecs.open(inpath, 'r', args.encoding)
        if args.use_ascii:
            datastr = unidecode(infile.read()).encode('ascii', 'ignore')
        else:
            datastr = infile.read()
        infile.close()

        # Split into tokens
        if args.case_sensitive:
            indata = re.split(regex,datastr)
        else:
            indata = re.split(regex,datastr.lower())
            
        # Add to unified token array
        unified += indata
        
        # Now we find word occurrences
        file_words = set()

        # Add words to word lists or increment appropriate counters
        for word in indata:
            if word == '':
                continue
            if word in wordlist:
                wordlist[word][0] += 1
                if word not in file_words:
                    file_words.add(word)
                    wordlist[word][1] += 1
            else:
                file_words.add(word)
                wordlist[word] = [1,1]

    # Build the final dictionary: word to token number
    token_to_idx = {}
    wordid = 1 
    ignore_counts = set(string.punctuation).union(string.whitespace) # Preserve tokens for all encountered punctuation or whitespace

    for item in wordlist:
        if item in ignore_counts or (wordlist[item][0] >= args.min_occurrences and wordlist[item][1] >= args.min_documents):
            token_to_idx[item] = wordid
            wordid += 1

    # Add a wildcard character onto the end of everything...
    token_to_idx['*/WILDCARD/*'] = wordid
    wildcard_id = wordid
    wordid += 1

    maxtoken = wordid

    # Now we create the final token array
    outdata = []

    for word in unified:
        if word == '':
            continue
        if word in wordlist:
            if word in token_to_idx:
                outdata.append(token_to_idx[word])
            else:
                outdata.append(wildcard_id)

    total_size = len(outdata)
    
    # Now we can figure out the split sizes
    val_size = int(args.val_frac * total_size)
    test_size = int(args.test_frac * total_size)
    train_size = total_size - val_size - test_size

    if not args.quiet:
        print 'Total unique words: {0}'.format(len(wordlist))
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

    # Split data up into train,val, and test sets. This avoids zeros popping up (might have been the cause of earlier issues)
    train = np.array(outdata[:train_size], dtype=dtype)
    val = np.array(outdata[train_size:train_size+val_size], dtype=dtype)
    test = np.array(outdata[-test_size:], dtype=dtype)
    splits = [train, val, test]

    # Write data to HDF5 file
    with h5py.File(args.output_h5, 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('val', data=val)
        f.create_dataset('test', data=test)

    # Dump a JSON file for the vocab
    json_data = {
        'token_to_idx': token_to_idx,
        'idx_to_token': {v: k for k, v in token_to_idx.iteritems()},
    }
    with open(args.output_json, 'w') as f:
        json.dump(json_data, f)
