#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np

def generate_sorting_number(args):
    with open(args.outfn, 'w') as fout:
        seq = np.arange(args.low, args.high)

        for i in range(args.size):
            np.random.shuffle(seq)
            pl = seq[:args.seq_len]
            seq_str = ' '.join(str(num) for num in sorted(pl))
            pl_str = ' '.join(str(num) for num in pl)
            fout.write(\
                '{} <sort> {}\n'.format(pl_str, seq_str))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(
            description='Generate random sorting number data')
    pa.add_argument('--low', type=int, default=0,\
            help='lower bound of number sequence')
    pa.add_argument('--high', type=int, default=5,\
            help='upper bound of number sequence')
    pa.add_argument('--seq-len', type=int, default=3,\
            help='subsequence length')
    pa.add_argument('--size', type=int, default=100,\
            help='number of number sequences')
    pa.add_argument('--outfn', \
            help='output filename')
    args = pa.parse_args()
    if args.seq_len > (args.high - args.low):
        sys.stderr.write(\
            'Error: seq-len can not be bigger than the range high-low')
        sys.exit(1)
    generate_sorting_number(args)
    
