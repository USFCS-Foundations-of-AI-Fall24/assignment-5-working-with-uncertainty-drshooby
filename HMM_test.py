import os
import unittest

from HMM import HMM


class HMMTest(unittest.TestCase):

    def test_load(self):
        want_transition = {
            '#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'},
            'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'},
            'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'},
            'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}
        }

        want_emission = {
            'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
            'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
            'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}
        }

        hmm = HMM()
        hmm.load('cat')
        self.assertEqual(hmm.transitions, want_transition)
        self.assertEqual(hmm.emissions, want_emission)

    def test_lander_transition_gen(self):

        # cells where 3 probabilities are available = (4 * 3) * 4
        # cells where 2 probabilities are available = (4 * 2) + (4 * 2)
        # cells where 1 probability is available = 1
        # file should have 65 lines + initial probability = 66

        want = 66
        hmm = HMM()
        hmm.get_transitions_mars()

        with open("LANDER_TEST.trans") as test_file:
            line_count = len(test_file.readlines())

        self.assertEqual(line_count, want)
        os.remove("LANDER_TEST.trans")
