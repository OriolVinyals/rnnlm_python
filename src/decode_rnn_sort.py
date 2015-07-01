#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import time
import neuralnet.rnn as rnn

import random

def build_idx_word(fn):
    word4idx = {}
    idx4word ={}
    word_idx = 0
    with open(fn) as fin:
        for line in fin:
            word = line.strip()
            idx4word[word] = word_idx
            word4idx[word_idx] = word
            word_idx += 1
    if '</s>' not in idx4word:
        sys.stderr.write('Error: Vocab does not contain </s>.\n')
    if '<unk>' not in idx4word:
        sys.stderr.write('Error: Vocab does not contain <unk>.\n')
    return idx4word, word4idx

def decode_sent(word, model, word4idx, idx4word):
    eos_idx = idx4word['</s>']
    sent = []
    #target_idx not used, set it to a constant
    target_idx = [([0], [0.0])]
    model.ResetStates()
    nums = word.split(' ')
    nums.insert(0, '</s>')
    nums.insert(len(nums), '<sort>')
    for n in nums:
        idx = idx4word[n]
        input_idx = [[idx]]
        _, probs = model.ForwardPropagate(input_idx, target_idx)

    nums = []
    max_len = 20
    while True:
        idx = np.argmax(probs[0])
        if idx == eos_idx:
            nums.append(word4idx[idx])
            break
        else:
            nums.append(word4idx[idx])
        if len(nums) >= max_len:
            break

        input_idx = [[idx]]
        _, probs = model.ForwardPropagate(input_idx, target_idx)

    return ' '.join(nums)


def decode_rnn_sort(args):
    idx4word, word4idx = build_idx_word(args.vocabfile)

    rnn_model = rnn.RNN()
    rnn_model.ReadModel(args.inmodel)

    print 'Randomly sample sentence from RNN LM'
    while True:
        word = raw_input('Enter the (partial) word (lowercased) to start with: ')
        if word == '0':
            break
        sent = decode_sent(word, rnn_model, word4idx, idx4word)
        print 'The decoded sorted sentence starting with {} is: \n{}'.format(\
                word, sent)

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Train a RNN language model')
    ## Require arguments 
    pa.add_argument('--vocabfile', \
            help='vocabulary filename (REQUIRED)')
    pa.add_argument('--inmodel', \
            help='inmodel name (REQUIRED)')
    args = pa.parse_args()

    if args.vocabfile == None or \
        args.inmodel == None:
        sys.stderr.write('Error: Invalid input arguments!\n')
        pa.print_help()
        sys.exit(1)

    decode_rnn_sort(args)

