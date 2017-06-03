# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse, json, os, codecs, h5py, re, string, random, six
from unidecode import unidecode

import numpy as np

def load_from_files(file_list,use_ascii,encoding):
    file_contents = []
    for path in file_list:
        with codecs.open(path, 'r', encoding) as infile:
            if use_ascii:
                file_contents.append(unidecode(infile.read()).encode('ascii', 'ignore'))
            else:
                file_contents.append(infile.read())
            infile.close()
    return file_contents
    
def parse_file(file_contents,regex,case_sensitive):
    # Split into tokens
    if not case_sensitive:
        file_contents = file_contents.lower()
    if regex != '':
        return [item for item in re.split(regex,file_contents,flags=re.UNICODE) if item != '']
    else:
        return list(file_contents)
        
def compute_frequency(parsed_files):
    tokenlist = {}
    for item in parsed_files:
        item_tokens = set()
        for token in item:
            if token == '':
                continue
            if token in tokenlist:
                tokenlist[token][0] += 1
                if token not in item_tokens:
                    item_tokens.add(token)
                    tokenlist[token][1] += 1
            else:
                item_tokens.add(token)
                tokenlist[token] = [1,1]
    return tokenlist
    
def tokenize_data(data_per_file,token_to_idx,wildcard_ids):
    unified_idx = []
    wildcard_replace_count = 0
    for item in data_per_file:
        for token in item:
            if token in token_to_idx:
                unified_idx.append(token_to_idx[token])
            else:
                if len(wildcard_ids) != 0:
                    unified_idx.append(random.choice(wildcard_ids))
                wildcard_replace_count += 1
    return unified_idx,wildcard_replace_count
                
def build_tokenset(wordlist,min_documents,min_occurrences,min_wildcards,max_wildcards,wildcard_rate):
    token_to_idx = {}
    wordid = 1 
    ignore_counts = set(string.punctuation).union(string.whitespace) # Preserve tokens for all encountered punctuation or whitespace
    
    total_eliminated = 0

    for item in wordlist:
        if item in ignore_counts or (wordlist[item][0] >= min_occurrences and wordlist[item][1] >= min_documents):
            token_to_idx[item] = wordid
            wordid += 1
        else:
            total_eliminated+=1
            
    wildcard_ids = []
    
    if total_eliminated > 0:
        num_distinct_wild = max(min_wildcards,int(wildcard_rate*total_eliminated))
        if max_wildcards > 0:
            num_distinct_wild = min(max_wildcards,num_distinct_wild)
    
        for wcnum in xrange(num_distinct_wild):
            token_to_idx['*/WILDCARD/*{0}'.format(wcnum)] = wordid
            wildcard_ids.append(wordid)
            wordid += 1

    maxtoken = wordid
    return token_to_idx,wildcard_ids,maxtoken
    
                
def save_to_hdf5(data,filename,train_size,val_size,test_size, dtype):
    # Split data up into train,val, and test sets. This avoids zeros popping up (might have been the cause of earlier issues)
    train = np.array(data[:train_size], dtype=dtype)
    val = np.array(data[train_size:train_size+val_size], dtype=dtype)
    test = np.array(data[-test_size:], dtype=dtype)
    splits = [train, val, test]

    # Write data to HDF5 file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('val', data=val)
        f.create_dataset('test', data=test)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', default='data/tiny-shakespeare.txt')
    parser.add_argument('--input_folder', default='')
    parser.add_argument('--output_h5', default='data/tiny-shakespeare.h5')
    parser.add_argument('--output_json', default='data/tiny-shakespeare.json')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--use_ascii', action='store_true')
    parser.add_argument('--encoding', default='utf-8')

    parser.add_argument('--use_words',action='store_true')
    parser.add_argument('--case_sensitive', action='store_true')
    parser.add_argument('--min_occurrences',type=int,default=20)
    parser.add_argument('--min_documents', type=int,default=1)
    parser.add_argument('--wildcard_rate',type=float,default=0.01)
    parser.add_argument('--wildcard_max',type=int, default=-1)
    parser.add_argument('--wildcard_min',type=int,default=10)
    args = parser.parse_args()

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
    regex = '(\W)' if args.use_words else ''
    if not args.use_words:
        args.case_sensitive = True
        args.min_occurrences = 0
        args.min_documents = 0
    
    files_parsed = [parse_file(f,regex,args.case_sensitive) for f in load_from_files(infiles,args.use_ascii,args.encoding)]
    
    wordlist = compute_frequency(files_parsed)
    
    # Build the final dictionary: word to token number
    token_to_idx,wildcard_ids,maxtoken = build_tokenset(wordlist,args.min_documents,args.min_occurrences,args.wildcard_min,args.wildcard_max,args.wildcard_rate)

    # Now we create the final token array
    outdata,wildcard_replace_count = tokenize_data(files_parsed,token_to_idx,wildcard_ids)

    total_size = len(outdata)
    
    # Now we can figure out the split sizes
    val_size = int(args.val_frac * total_size)
    test_size = int(args.test_frac * total_size)
    train_size = total_size - val_size - test_size

    if not args.quiet:
        if len(wildcard_ids) > 0:
            wildcard_spec = ' ({0} wildcards)'.format(len(wildcard_ids))
            print('Total unique tokens: {0}'.format(len(wordlist)))
        else:
             wildcard_spec = ''
        print('Total vocabulary size: {0}{1}'.format(len(token_to_idx), wildcard_spec))
        print('Total tokens in file: {0}'.format(total_size))
        if len(wildcard_ids) > 0:
            print('Total wildcards in file: {0} ({1}%)'.format(wildcard_replace_count,100.0*wildcard_replace_count/total_size))
        print('  Training size: {0}'.format(train_size))
        print('  Val size: {0}'.format(val_size))
        print('  Test size: {0}'.format(test_size))

    # Choose the datatype based on the vocabulary size
    dtype = np.uint8
    if len(token_to_idx) > 255:
        dtype = np.uint32
    if not args.quiet:
        print('Using dtype {0}'.format(dtype))

    save_to_hdf5(outdata,args.output_h5,train_size,val_size,test_size,dtype)

    # Dump a JSON file for the vocab
    json_data = {
        'token_to_idx': token_to_idx,
        'idx_to_token': {v: k for k, v in token_to_idx.iteritems()},
        'wildcards':wildcard_ids,
        'tokenize_regex':regex,
        'case_sensitive':args.case_sensitive,
        'use_ascii':args.use_ascii
    }
    with open(args.output_json, 'w') as f:
        json.dump(json_data, f)
