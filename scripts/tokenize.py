import argparse, json, re, random, h5py
import numpy as np
from preprocess import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_str',type=str, default='')
    parser.add_argument('--input_txt',type=str,default='')
    parser.add_argument('--input_folder', default='')
    parser.add_argument('--input_json',type=str, default='data/tiny-shakespeare.json')

    parser.add_argument('--output_json',type=str, default='')
    parser.add_argument('--output_h5', default='')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    token_to_idx = []
    wildcard_set = []
    regex = ''
    case_sensitive = False
    use_ascii = False
    unified = []

    if args.output_json == '' and args.output_h5 == '':
        print 'No output file specified'
    else:
        with open(args.input_json) as jsonfile:
            json_data = json.load(jsonfile)
            token_to_idx = json_data['token_to_idx']
            wildcard_set = json_data['wildcards']
            case_sensitive = json_data['case_sensitive']
            regex = json_data['tokenize_regex']
            use_ascii = json_data['use_ascii']
        
    # Build list of files
    infiles = []
    file_contents = []
    if args.input_folder != '':
        infiles = [os.path.join(args.input_folder,item) for item in os.listdir(args.input_folder) if item[-4:]=='.txt']
    elif args.input_txt != '':
        infiles = [args.input_txt]
        
    if len(infiles) != 0:
        file_contents = load_from_files(infiles,use_ascii,args.encoding)
        
    if args.input_str != '':
        file_contents.append(args.input_str) 
        
    files_parsed = [parse_file(f,regex,case_sensitive) for f in file_contents]
    
    outdata,wildcard_replace_count = tokenize_data(files_parsed,token_to_idx,wildcard_set)
    
    if args.output_h5 != '':
        total_size = len(outdata)
        
        # Choose the datatype based on the vocabulary size
        dtype = np.uint8
        if len(token_to_idx) > 255:
            dtype = np.uint32
        if not args.quiet:
            print 'Using dtype ', dtype
        val_size = int(args.val_frac * total_size)
        test_size = int(args.test_frac * total_size)
        train_size = total_size - val_size - test_size
        save_to_hdf5(outdata,args.output_h5,train_size,val_size,test_size,dtype)
        
        if not args.quiet:
            if len(wildcard_set) > 0:
                wildcard_spec = ' ({0} wildcards)'.format(len(wildcard_set))
            else:
                 wildcard_spec = ''
            print 'Total vocabulary size: {0}{1}'.format(len(token_to_idx), wildcard_spec)
            print 'Total tokens in file: {0}'.format(total_size)
            if len(wildcard_set) > 0:
                print 'Total wildcards in file: {0} ({1}%)'.format(wildcard_replace_count,100.0*wildcard_replace_count/total_size)
            else:
                print 'Total Ignored: {0}'.format(wildcard_replace_count)
            print '  Training size: {0}'.format(train_size)
            print '  Val size: {0}'.format(val_size)
            print '  Test size: {0}'.format(test_size)
            
    if args.output_json != '':
        json_data = {'tokens':outdata}
        with open(args.output_json,'w') as jsonfile:
            json.dump(json_data,jsonfile)
        
        