#!/usr/bin/env python

import os
import sys
import argparse

import re

def clean_token(word):
    #word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", word)
    word = re.sub(r":", ": ", word)
    word = re.sub(r"\.", ". ", word)
    return word

def preprocess_data(args):
    with open(args.infile) as fin, \
            open(args.outfile, 'w') as fout:
        new_line = []
        for line in fin:
            if line.strip() == '':
                if args.empty_line:
                    fout.write('{}\n'.format(\
                            ' <sep> '.join(new_line)))
                    new_line = []
                continue
            words = line.strip().split()
            for word in words:
                if args.lowercase:
                    word = word.lower()
                if args.tokenizer:
                    word = clean_token(word)
                new_line.append(' '.join(word))
            if not args.empty_line:
                fout.write('{}\n'.format(\
                        ' <sep> '.join(new_line)))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(
            description='Preprocess data for char RNN LM')
    # Required arguments
    pa.add_argument('--infile', \
            help='input filename')
    pa.add_argument('--outfile', \
            help='output filename')
    # Optional arguments
    pa.add_argument('--lowercase', action='store_true',\
            dest='lowercase', \
            help='lowercase all characteres')
    pa.set_defaults(lowercase=False)
    pa.add_argument('--clean-token', action='store_true', \
            dest='tokenizer', \
            help='clean tokens')
    pa.set_defaults(tokenizer=False)
    pa.add_argument('--empty-line', action='store_true', \
            dest='empty_line', \
            help='sentence is splited by empty line')
    pa.set_defaults(empty_line=False)
    args = pa.parse_args()

    preprocess_data(args)


