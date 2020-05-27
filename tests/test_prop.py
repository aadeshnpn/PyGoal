from unittest import TestCase

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


"""Testing the expected value computation for PPO"""

#### Trace Related Functions

def create_trace_flloat(traceset, i, keys):
    setslist = [create_sets(traceset[k][:i]) for k in keys]
    dictlist = [FiniteTrace.fromStringSets(s) for s in setslist]
    keydictlist = dict()
    j = 0
    for k in keys:
        keydictlist[k] = dictlist[j]
        j += 1
    t = FiniteTraceDict.fromDictSets(keydictlist)
    return t


def create_sets(trace):
    return [set([l]) for l in trace]


def create_trace_dict(trace, i, keys):
    tracedict = dict()
    for k in keys:
        tracedict[k] = trace[k][:i+1]
    return tracedict


def create_trace_skeleton(state, keys):
    # Create a skeleton for trace
    trace = dict(zip(keys, [list() for i in range(len(keys))]))
    j = 0
    for k in keys:
        trace[k].append(state[j])
        j += 1
    return trace


def trace_accumulator(trace, state, keys):
    for j in range(len(keys)):
        # Add the state variables to the trace
        # temp = trace[keys[j]][-1].copy()
        # temp.append(state[j])
        trace[keys[j]].append(state[j])
    return trace


def recognition(trace, keys, goalspec):
    # goalspec = 'F P_[C][True,none,==]'
    # parse the formula
    parser = LTLfGParser()

    # Define goal formula/specification
    parsed_formula = parser(goalspec)

    # Change list of trace to set
    traceset = trace.copy()
    akey = list(traceset.keys())[0]
    # print('recognizer', traceset)
    # Loop through the trace to find the shortest best trace
    for i in range(0, len(traceset[akey])+1):
        t = create_trace_flloat(traceset, i, keys)
        result = parsed_formula.truth(t)
        if result is True:
            # self.set_state(self.env, trace, i)
            return True, i

    return result, i


class TestTraceKeyDoor(TestCase):

    def setUp(self):
        pass

    def test_finally_true(self):
        traceset = dict()
        traceset['C'] = [False] * 9 + [True]

        goalspecs = 'F P_[C][True,none,==]'
        result, i = recognition(traceset, ['C'], goalspecs)
        self.assertTrue(result)

    def test_globally_true(self):
        traceset = dict()
        traceset['C'] = [True] * 10

        goalspecs = 'G P_[C][True,none,==]'
        result, i = recognition(traceset, ['C'], goalspecs)
        self.assertTrue(result)

    def test_until_true(self):
        traceset = dict()
        traceset['C'] = [True] * 10

        goalspecs = 'G(P_[C][True,none,==]) U P_[D][True,none,==]'
