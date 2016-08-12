# -*- coding: utf-8 -*-

import argparse, json, os, codecs, h5py
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

    infiles = []
    if args.input_folder != '':
        infiles = [os.path.join(args.input_folder,item) for item in os.listdir(args.input_folder) if item[-4:]=='.txt']
    else:
        infiles = [args.input_txt]

    if args.min_documents > len(infiles):
        args.min_documents = len(infiles)

    punctuations = '.,?!&():;"\'\\/ \t'
    punctuation = {i for i in punctuations}

    wordlist = {i: [0,args.min_documents] for i in punctuations}
    wordlist['\n'] = [0,args.min_documents]

    unified = []

    for inpath in infiles:

        infile = codecs.open(inpath, 'r', args.encoding)
        if args.use_ascii:
            datastr = unidecode(infile.read()).encode('ascii', 'ignore')
            datastr = datastr.replace('//------------------------------//', '')
            datastr = datastr.replace('//', '')
        else:
            datastr = infile.read()
        infile.close()

        if args.case_sensitive:
            indata = datastr.split('/n')
        else:
            indata = datastr.lower().split('/n')

        unified += indata
        file_words = {i for i in punctuations}
        file_words.add('\n')

        for line in indata:
            startchar = 0
            for cnt, cha in enumerate(line):
                if cha in punctuation:
                    if cnt > startchar:
                        word = line[startchar:cnt]
                        if word in wordlist:
                            wordlist[word][0] += 1
                            if word not in file_words:
                                file_words.add(word)
                                wordlist[word][1] += 1
                        else:
                            wordlist[word] = [1,1]
                    wordlist[cha][0] += 1
                    startchar = cnt + 1

            if cnt > startchar:
                word = line[startchar:cnt]
                if word in wordlist:
                    wordlist[word][0] += 1
                    if word not in file_words:
                        file_words.add(word)
                        wordlist[word][1] += 1
                else:
                    wordlist[word] = [1, 1]
            wordlist['\n'][0] += 1

    finaldict = {i: c+1 for c, i in enumerate(punctuation)}
    finaldict['\n'] = len(punctuation)
    wordid = len(punctuation) + 2

    for item in wordlist:
        if item in punctuation or item == '\n':
            continue
        if wordlist[item][0] >= args.min_occurrences and wordlist[item][1] >= args.min_documents:
            finaldict[item] = wordid
            wordid += 1

    finaldict['*/WILDCARD/*'] = wordid
    wordid += 1

    maxtoken = wordid

    outdata = []

    c = 0

    for line in unified:
        startchar = 0
        for cnt, cha in enumerate(line):
            if cha in punctuation:
                if cnt > startchar:
                    word = line[startchar:cnt]
                    if word in finaldict:
                        outdata.append(finaldict[word])
                    else:
                        outdata.append(maxtoken-1)
                outdata.append(finaldict[cha])
                startchar = cnt + 1

        if cnt > startchar:
            word = line[startchar:cnt]
            if word in finaldict:
                outdata.append(finaldict[word])
            else:
                outdata.append(maxtoken - 1)
        outdata.append(finaldict['\n'])

    # First go the file once to see how big it is and to build the vocab
    total_size = len(outdata)
    token_to_idx = finaldict
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

    # Just load data into memory ... we'll have to do something more clever
    # for huge datasets but this should be fine for now
    train = np.zeros(train_size, dtype=dtype)
    val = np.zeros(val_size, dtype=dtype)
    test = np.zeros(test_size, dtype=dtype)
    splits = [train, val, test]

    # Go through the file again and write data to numpy arrays
    split_idx, cur_idx = 0, 0
    for token in outdata:
        splits[split_idx][cur_idx] = token
        cur_idx += 1
        if cur_idx == splits[split_idx].size:
            split_idx += 1
            cur_idx = 0

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
