#!/usr/bin/env python

import numpy as np
DTYPE = np.double

def sigmoid(x):
    return (np.tanh(0.5 * x) + 1) * 0.5

class RNNUnit():
    """This class stores the activations of the RNN Unit"""
    def __init__(self, hidden_size, batch_size, dtype):
        self.h = np.zeros([hidden_size, batch_size], dtype=dtype)

    """Returns h = sigmoid(Whx*x + Whh*h)"""
    def forward_function(self, x, hprev, Whx, Whh):
        self.h = np.dot(Whx, x) + np.dot(Whh, hprev)
        self.h = sigmoid(self.h)
        return self.h

    """Returns dEdhprev, dEdWhx, and dEdWhh"""
    def backward_function(self, x, hprev, dEdh, Whx, Whh):
        dEdh = dEdh * (1 - self.h) * self.h
        dEdhprev = np.dot(dEdh.T, Whh)
        dWhh = np.dot(dEdh, hprev.T)
        dWhx = np.dot(dEdh, x.T)
        return dEdhprev.T, dWhx, dWhh


#Tests for the gradient computation of the single RNNUnit
if __name__ == '__main__':
    hidden_size = 10
    input_size = 5
    batch_size = 2
    Whh = np.zeros([hidden_size, hidden_size], dtype=DTYPE)
    Whh += np.random.uniform(-0.1, 0.1, [hidden_size, hidden_size])
    Whx = np.zeros([hidden_size, input_size], dtype=DTYPE)
    Whx += np.random.uniform(-0.1, 0.1, [hidden_size, input_size])
    x = np.zeros([input_size, batch_size], dtype=DTYPE)
    x += np.random.uniform(-0.1, 0.1, [input_size, batch_size])
    hprev = np.zeros([hidden_size, batch_size], dtype=DTYPE)
    hprev += np.random.uniform(-0.1, 0.1, [hidden_size, batch_size])
    
    
    rnn_unit = RNNUnit(hidden_size, batch_size, DTYPE)
    
    # Exact gradient computation
    h = rnn_unit.forward_function(x, hprev, Whx, Whh)
    E = np.sum(h)
    dEdh = np.ones([hidden_size, batch_size], dtype=DTYPE)
    dEdhprev, dWhx, dWhh = rnn_unit.backward_function(x, hprev, dEdh, Whx, Whh)
    
    # Numerical gradient computation
    epsilon = 1e-7
    numdWhh = np.zeros([hidden_size,hidden_size],dtype=DTYPE)
    for i in range(hidden_size):
        for j in range(hidden_size):
            newWhh = np.copy(Whh)
            newWhh[i,j] += epsilon
            
            h = rnn_unit.forward_function(x, hprev, Whx, newWhh)
            newE = np.sum(h)
            numdWhh[i,j] = (newE - E) / epsilon
    
    diff = np.sum(numdWhh - dWhh)
    assert diff < 1e-3
    print 'Check Passed! Diff is', diff
    
    # Numerical gradient computation
    epsilon = 1e-7
    numdWhx = np.zeros([hidden_size,input_size],dtype=DTYPE)
    for i in range(hidden_size):
        for j in range(input_size):
            newWhx = np.copy(Whx)
            newWhx[i,j] += epsilon
            
            h = rnn_unit.forward_function(x, hprev, newWhx, Whh)
            newE = np.sum(h)
            numdWhx[i,j] = (newE - E) / epsilon
    
    diff = np.sum(numdWhx - dWhx)
    assert diff < 1e-3
    print 'Check Passed! Diff is', diff
    exit(0)
