import argparse, json, re

parser = argparse.ArgumentParser()

parser.add_argument('--input',type=str, default='the quick brown fox jumped over the lazy dogs')
parser.add_argument('--outfile',type=str, default='encoded_input.json')
parser.add_argument('--dictionary',type=str, default='data/tiny-shakespeare.json')
parser.add_argument('--use_ascii',action='store_true')
parser.add_argument('--case_sensitive',action='store_true')
args = parser.parse_args()

datastr = ''
token_to_idx = []
regex = '(\W)'

with open(args.dictionary) as jsonfile:
    token_to_idx = json.load(jsonfile)['token_to_idx']

if args.use_ascii:
    datastr = unidecode(args.input).encode('ascii', 'ignore')
else:
    datastr = args.input
    
if args.case_sensitive:
    indata = re.split(regex,datastr,flags=re.UNICODE)
else:
    indata = re.split(regex,datastr.lower(),flags=re.UNICODE)
    
tokens = []
print indata

for item in indata:
    if item == '':
        continue
    if item not in token_to_idx:
        item = '*/WILDCARD/*0'
    tokens.append(token_to_idx[item])
    
json_data = {'tokens':tokens}

with open(args.outfile,'w') as jsonfile:
    json.dump(json_data,jsonfile)