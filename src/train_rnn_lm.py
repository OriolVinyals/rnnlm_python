#!/usr/bin/env python

import os
import sys
import argparse

import numpy as np
import time
import neuralnet.rnn as rnn

def load_vocab(fn):
    vocab = {}
    word_idx = 0
    with open(fn) as fin:
        for line in fin:
            word = line.strip()
            vocab[word] = word_idx
            word_idx += 1
    if '</s>' not in vocab:
        sys.stderr.write('Error: Vocab does not contain </s>.\n')
    if '<unk>' not in vocab:
        sys.stderr.write('Error: Vocab does not contain <unk>.\n')
    return vocab

def load_txt(fn):
    txt = []
    with open(fn) as fin:
        for line in fin:
            txt.append(line.strip())
    return txt

def eval_lm(model, batched_data, vocab):
    total_logp = 0.0
    sents_processed = 0
    ivcount = 0
    eos_idx = vocab['</s>']
    for batch_idx in range(len(batched_data['input_idxs'])):
        input_idxs = batched_data['input_idxs'][batch_idx]
        target_idxs = batched_data['target_idxs'][batch_idx]

        inds_reset = []
        for i, first_ind in enumerate(input_idxs[0]):
            if first_ind == eos_idx:
                inds_reset.append(i)
        model.ResetStates(inds_reset)
        curr_logp = 0.0

        loss, probs = model.ForwardPropagate(input_idxs, target_idxs)
        curr_logp += loss
        ivcount += sum([sum(target_idxs[x][1]) for x in range(len(target_idxs))])

        sents_processed += model.batch_size
        total_logp += curr_logp
        if (sents_processed % 200) == 0:
            print '.',
            sys.stdout.flush()
    
    print ' '
    print 'IV words: {}'.format(ivcount)
    print 'model perplexity: {}'.format(np.exp(-total_logp / ivcount))
    
    return total_logp

def batch_data(txt, vocab, batch_size=1, bptt=1, separator=""):
    print '************************Data Preparation****************************'
    print 'bptt : {}'.format(bptt)
    print 'batch_size: {}'.format(batch_size)
    eos_idx = vocab['</s>']
    unk_idx = vocab['<unk>']
    append_idx = 0
    sent_in_idx = []
    for sent in txt:
        idx_words = []
        if separator:
            sep_seen = 0.0
        else:
            sep_seen = 1.0
        idx_words.append((eos_idx,sep_seen))
        for word in sent.split():
            if word in vocab:
                idx_words.append((vocab[word],sep_seen))
                if word == separator:
                    sep_seen = 1.0
            else:
                assert False
                idx_words.append((unk_idx,sep_seen))
        idx_words.append((eos_idx,sep_seen))
        sent_in_idx.append(idx_words)

    # Shuffle examples
    np.random.shuffle(sent_in_idx)
    batch_data = {}
    batch_data['input_idxs'] = []
    batch_data['target_idxs'] = []
    in_flight_sents = [[]] * batch_size
    done = False
    while not done:
        cur_inputs = []
        cur_targets = []
        done = True
        for b in range(batch_size):
            if len(in_flight_sents[b]) <= 1:
                if len(sent_in_idx) > 0:
                    in_flight_sents[b] = sent_in_idx[0]
                    del sent_in_idx[0]
                    done = False
            else:
                done = False
        if done:
            break
            
        for t in range(bptt):
            cur_input = [0] * batch_size
            cur_target = [0] * batch_size
            cur_weight = [0.0] * batch_size
            for b in range(batch_size):
                sent = in_flight_sents[b]
                if (len(sent) >= 2):
                    cur_input[b] = sent[0][0]
                    cur_target[b] = sent[1][0]
                    cur_weight[b] = sent[1][1]
                    in_flight_sents[b] = sent[1:]
                else:
                    #Done with this
                    in_flight_sents[b] = []
            cur_inputs.append(cur_input)
            cur_targets.append((cur_target, cur_weight))

        batch_data['input_idxs'].append(cur_inputs)
        batch_data['target_idxs'].append(cur_targets)
    print 'Data Preparation Done!'
    return batch_data

def batch_sgd_train(rnn_model, init_learning_rate, batch_size, train_txt, \
        valid_txt, outmodel, vocab, bptt, tol, separator):    
    #train_txt = train_txt[:100]
    #valid_txt = valid_txt[:100]
    batched_data = batch_data(train_txt, vocab, batch_size, bptt, separator)
    eos_idx = vocab['</s>']
    if valid_txt == []:
        batched_valid = None
    else:
        batched_valid = batch_data(valid_txt, vocab, batch_size, bptt, separator)

    last_logp = -np.finfo(float).max
    curr_logp = last_logp

    sents_processed = 0
    iters = 0

    learning_rate = init_learning_rate
    batch_indices = np.arange(len(batched_data['input_idxs']))
    halve_learning_rate = False

    start_time = time.time()
    end_time = start_time
    last_end_time = end_time
    while True:
        iters += 1
        print '******************************* Iteration {} ***********************************'.format(iters)

        rnn_model.set_learning_rate(learning_rate)
        print 'learning_rate = {}\n'.format(learning_rate)

        logp = 0.0
        ivcount = 0;

        for batch_idx in batch_indices:
            input_idxs = batched_data['input_idxs'][batch_idx]
            target_idxs = batched_data['target_idxs'][batch_idx]
            
            #We reset the state of those whose first input is eos
            inds_reset = []
            for i, first_ind in enumerate(input_idxs[0]):
                if first_ind == eos_idx:
                    inds_reset.append(i)
            rnn_model.ResetStates(inds_reset)

            loss, probs = rnn_model.ForwardPropagate(input_idxs, target_idxs)
            dWhh, dWoh, dWhx, dbo = rnn_model.BackPropagate(input_idxs, target_idxs)
            rnn_model.UpdateWeight(dWhh, dWoh, dWhx, dbo)
            logp += loss
            ivcount += sum([sum(target_idxs[x][1]) for x in range(len(target_idxs))])

            sents_processed += batch_size

            if (sents_processed % 500) == 0:
                print '.',
                sys.stdout.flush()

        print ''
        print 'iv words in training: {}'.format(ivcount)
        print 'perlexity on training:{}'.format(np.exp(-logp / ivcount))
        print 'log-likelihood on training:{}'.format(logp)

        print 'epoch done!'

        #curr_logp = logp

        if batched_valid != None:
            print '-------------Validation--------------'
            curr_logp = eval_lm(rnn_model, batched_valid, vocab)
            print 'log-likelihood on validation: {}'.format(curr_logp)

        last_end_time = end_time
        end_time = time.time()
        print 'time elasped {} secs for this iteration out of {} secs in total.'.format(\
                end_time - last_end_time, end_time - start_time)

        obj_diff = curr_logp - last_logp
        if obj_diff < 0:
            print 'validation log-likelihood decrease; restore parameters'
            #halve_learning_rate = True
            rnn_model.RestoreModel()
        else:
            rnn_model.CacheModel()

        if obj_diff <= tol:
            if not halve_learning_rate:
                halve_learning_rate = True
            else:
                if outmodel != '':
                    rnn_model.WriteModel(outmodel)
                break

        if halve_learning_rate:
            learning_rate *= 0.5
            halve_learning_rate = False
            if outmodel != '':
                rnn_model.WriteModel(outmodel)
        last_logp = curr_logp

def train_rnn_lm(args):
    vocab = load_vocab(args.vocabfile)
    train_txt = load_txt(args.trainfile)
    valid_txt = []
    if args.valid:
        valid_txt = load_txt(args.validfile)
        
    vocab_size = len(vocab)

    rnn_model = rnn.RNN()
    rnn_model.set_init_range(args.init_range)
    rnn_model.set_learning_rate(args.init_alpha)
    rnn_model.set_input_size(vocab_size)
    rnn_model.set_hidden_size(args.nhidden)
    rnn_model.set_output_size(vocab_size)
    rnn_model.set_bptt_unfold_level(args.bptt)
    rnn_model.set_batch_size(args.batchsize)

    rnn_model.AllocateModel()
    if args.inmodel == None:
        rnn_model.InitializeParemters()
    else:
        rnn_model.ReadModel(args.inmodel)
        
    print args
    batch_sgd_train(rnn_model, args.init_alpha, args.batchsize, train_txt, \
        valid_txt, args.outmodel, vocab, args.bptt, args.tol, args.separator)


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Train a RNN language model')
    ## Require arguments 
    pa.add_argument('--trainfile', \
            help='train filename (REQUIRED)')
    pa.add_argument('--vocabfile', \
            help='vocabulary filename (REQUIRED)')
    pa.add_argument('--outmodel', \
            help='output model name (REQUIRED)')
    ## Optional arguments
    pa.add_argument('--validfile', \
            help='validation filename')
    pa.add_argument('--validate', action='store_true', \
            dest='valid', \
            help='validation during training')
    pa.set_defaults(valid=False)
    pa.add_argument('--inmodel', \
            help='input model name')
    pa.add_argument('--separator', \
            help='separator to use')
    pa.add_argument('--nhidden', type=int, default=10, \
            help='hidden layer size, integer > 0')
    pa.add_argument('--bptt', type=int, default=1, \
            help='backpropagate through time level, integer >= 1')
    pa.add_argument('--init-alpha', type=float, default=0.1, \
            help='initial learning rate, scalar > 0')
    pa.add_argument('--init-range', type=float, default=0.1, \
            help='random initial range, scalar > 0')
    pa.add_argument('--batchsize', type=int, default=1, \
            help='training batch size, integer >= 1')
    pa.add_argument('--tol', type=float, default=1e-3, \
            help='minimum improvement for log-likelihood, scalar > 0')
    args = pa.parse_args()

    if args.trainfile == None or \
        args.vocabfile == None or \
        args.outmodel == None or \
        args.nhidden < 1 or \
        args.init_alpha <= 0 or \
        args.init_range <= 0 or \
        args.bptt < 1 or \
        args.batchsize < 1 or \
        args.tol <= 0.0:
        sys.stderr.write('Error: Invalid input arguments!\n')
        pa.print_help()
        sys.exit(1)
    if args.valid and args.validfile == None:
        sys.stderr.write('Error: If valid, the validfile can not be None!')
        sys.exit(1)

    #np.__config__.show()
    train_rnn_lm(args)
