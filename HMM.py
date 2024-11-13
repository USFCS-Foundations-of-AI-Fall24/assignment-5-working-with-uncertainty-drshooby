import random
import argparse
import codecs
import os
from collections import defaultdict
from os.path import basename


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

   ## you do this.
    def generate(self, n, basename):
        """return an n-length Sequence by randomly sampling from this HMM."""
        sequence = Sequence([], [])

        init_probabilities = self.transitions["#"]
        init_states = list(init_probabilities.keys())
        init_weights = [float(init_probabilities[state]) for state in init_states]

        curr_state = random.choices(init_states, weights=init_weights, k=1)[0]

        for _ in range(n):
            e_probabilities = self.emissions[curr_state]
            actions = list(e_probabilities.keys())
            action_weights = [float(e_probabilities[action]) for action in actions]
            action = random.choices(actions, weights=action_weights, k=1)[0]

            sequence.stateseq.append(curr_state)
            sequence.outputseq.append(action)

            t_probabilities = self.transitions[curr_state]
            next_states = list(t_probabilities.keys())
            next_weights = [float(t_probabilities[state]) for state in next_states]
            curr_state = random.choices(next_states, weights=next_weights, k=1)[0]

        with open(basename + "_sequence.tagged.obs", "w") as obsfile_tagged:
            for state, out in zip(sequence.stateseq, sequence.outputseq):
                obsfile_tagged.write(state + "\n")
                obsfile_tagged.write(out + "\n")

        with open(basename + "_sequence.obs", "w") as obsfile:
            for out in sequence.outputseq:
                obsfile.write("\n")
                obsfile.write(out + "\n")

    def get_transitions_mars(self, n=5, fname="LANDER_TEST.trans"):

        probs = defaultdict(dict)
        probs['#'][(1, 1)] = 1.0

        def in_bounds(i, j):
            return 0 < i < n + 1 and 0 < j < n + 1

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                transitions = []
                if in_bounds(i, j + 1):
                    transitions.append(((i, j + 1), 0.15))
                if in_bounds(i + 1, j):
                    transitions.append(((i + 1, j), 0.15))
                if in_bounds(i + 1, j + 1):
                    transitions.append(((i + 1, j + 1), 0.70))
                # sanity check: should be 3 max for this, since you can only go down, diagonal, or right
                total_probabilities = 0
                for state, prob in transitions:
                    probs[(i, j)][state] = prob
                    total_probabilities += prob
                if total_probabilities != 1.0:
                    probs[(i, j)][(i, j)] = 1.0 - total_probabilities

        with open(fname, "w") as lander_file:
            for k, v in probs.items():
                # outer dict
                start = f"{k[0]},{k[1]}" if k != "#" else k
                # inner dict
                for state, prob in v.items():
                    lander_file.write(f'{start} {state[0]},{state[1]} {prob}\n')

    def get_emissions_mars(self, n=5, fname="LANDER_TEST.emit"):
        probs = defaultdict(dict)
        # correct p=0.6, other directions 0.1
        # up down right left
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        def in_bounds(i, j):
            return 0 < i < n + 1 and 0 < j < n + 1

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                transitions = []
                missed = 0
                for dx, dy in dirs:
                    if in_bounds(i + dx, j + dy):
                        transitions.append(((i + dx, j + dy), 0.1))
                    else:
                        missed += 0.1

                # update the probability of the current position (from Piazza)
                transitions.insert(0, ((i, j), 0.6 + missed))

                for state, prob in transitions:
                    probs[(i, j)][state] = prob

        with open(fname, "w") as lander_file:
            for k, v in probs.items():
                # outer dict
                start = f"{k[0]},{k[1]}"
                # inner dict
                for state, prob in v.items():
                    lander_file.write(f'{start} {state[0]},{state[1]} {prob}\n')

    def forward(self, sequence):

        M = defaultdict(dict)
        M[0]["#"] = 1.0
        for s in self.transitions["#"]:
            if s == "#":
                continue
            t = float(self.transitions["#"][s])
            e = float(self.emissions[s][sequence.outputseq[0]])
            M[0][s] = t * e

        for i in range(1, len(sequence.outputseq)):
            obs = sequence.outputseq[i]
            for curr in self.transitions:
                if curr == "#":
                    continue
                sum_ = 0
                for prev in self.transitions:
                    if prev == "#":
                        continue
                    transition_prob = self.transitions[prev][curr]
                    emission_prob = self.emissions[curr][obs]
                    dp = M[i - 1][prev]
                    sum_ += dp * float(transition_prob) * float(emission_prob)
                M[i][curr] = sum_

        return max(M[len(sequence.outputseq) - 1])


    # def viterbi(self, sequence):
    #
    #     M = defaultdict(dict)
    #     T = self.transitions
    #     E = self.emissions
    #     O = sequence.outputseq
    #     state_values = sequence.stateseq
    #     states = [state for state in self.transitions if state != "#"]
    #
    #     backpointers = defaultdict(dict)
    #
    #     for s in state_values:
    #         transition = T["#"][s]
    #         emission = E[s][O[0]]
    #         M[1][s] = float(transition) * float(emission)
    #
    #     for o in sequence.outputseq:
    #         for s in state_values:
    #             val =

def generate_sequence_from_obs(obsfile):
    seq = Sequence([], [])
    with open(obsfile) as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
        for line in lines:
            seq.outputseq.append(line)
    return seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="HMM basename (e.g., cat, partsofspeech, etc.")
    parser.add_argument("--generate", type=int, help="Generate a random sequence of length n")
    parser.add_argument("--forward", type=str, help="Run the forward algorithm on a .obs file")
    args = parser.parse_args()
    hmm = HMM()
    hmm.load(args.basename)
    if args.generate:
        hmm.generate(args.generate, args.basename)
    if args.forward:
        obsfile = args.forward
        seq = generate_sequence_from_obs(obsfile)
        r = hmm.forward(seq)
        print(r)


if __name__ == '__main__':
    main()





