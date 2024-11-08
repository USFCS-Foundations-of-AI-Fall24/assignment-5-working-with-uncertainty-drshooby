

import random
import argparse
import codecs
import os
from collections import defaultdict

import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""


        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        with open(basename + ".trans") as transitionfile:
            lines = [line.strip().split() for line in transitionfile]
            transition_tmp = defaultdict(dict)
            for line in lines:
                transition_tmp[line[0]][line[1]] = line[2]
            self.transitions = dict(transition_tmp)

        with open(basename + ".emit") as emissionfile:
            lines = [line.strip().split() for line in emissionfile]
            emission_tmp = defaultdict(dict)
            for line in lines:
                emission_tmp[line[0]][line[1]] = line[2]
            self.emissions = dict(emission_tmp)



def main():
    hmm = HMM()
    hmm.load('cat')

if __name__ == '__main__':
    main()


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        pass

    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.






