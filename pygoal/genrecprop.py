"""Genrator, Recognizer, and Propapagator class."""

import numpy as np
from abc import abstractmethod

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


# GenRecProp algorithm
class GenRecProp:
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], bt_flag=False):
        self.env = env
        self.keys = keys
        self.goalspec = goalspec
        self.bt_flag = bt_flag
        self.gtable = gtable
        self.max_trace_len = max_trace
        self.actionsidx = actions
        self.nprandom = np.random.RandomState()        # pylint: disable=E1101

    @abstractmethod
    def gtable_key(self, state):
        raise NotImplementedError

    @abstractmethod
    def get_curr_state(self, env):
        raise NotImplementedError

    @abstractmethod
    def create_trace_skeleton(self, state):
        raise NotImplementedError

    @abstractmethod
    def train(self, epoch=150, verbose=False):
        raise NotImplementedError

    @abstractmethod
    def test(self, render=False, verbose=False):
        raise NotImplementedError

    @abstractmethod
    def get_action_policy(self, policy, state):
        raise NotImplementedError

    def get_action(self, state):
        return self.nprandom.choice(
            self.actionsidx,
            p=list(self.gtable[state].values())
            )

    def create_gtable_indv(self, state):
        p = np.ones(len(self.actionsidx), dtype=np.float64)
        p = p / (len(self.actionsidx) * 1.0)

        self.gtable[state] = dict(
                        zip(self.actionsidx, p))

    def trace_accumulator(self, trace, state):
        for j in range(len(self.keys)):
            # Add the state variables to the trace
            temp = trace[self.keys[j]][-1].copy()
            temp.append(state[j])
            trace[self.keys[j]].append(temp)
        return trace

    def evaluate_trace(self, goalspec, trace):
        # Test the prop algorithm
        parser = LTLfGParser()
        parsed_formula = parser(goalspec)

        # Create a trace compatiable with Flloat library
        t = self.create_trace_flloat(self.list_to_trace(trace), 0)
        result = parsed_formula.truth(t)

        return result

    def list_to_trace(self, trace):
        for j in range(len(self.keys)):
            temp = trace[self.keys[j]]
            trace[self.keys[j]] = [temp]

        return trace

    def create_trace_flloat(self, traceset, i):
        setslist = [self.create_sets(traceset[k][i]) for k in self.keys]
        a = self.create_sets(traceset['A'][i])
        setslist.append(a)
        dictlist = [FiniteTrace.fromStringSets(s) for s in setslist]
        keydictlist = dict()
        keydictlist['A'] = dictlist[-1]
        j = 0
        for k in self.keys:
            keydictlist[k] = dictlist[j]
            j += 1
        t = FiniteTraceDict.fromDictSets(keydictlist)
        return t

    def create_sets(self, trace):
        if len(trace) == 1:
            return [set(trace)]
        else:
            return [set([l]) for l in trace]

    def create_trace_dict(self, trace, i):
        tracedict = dict()
        for k in self.keys:
            tracedict[k] = trace[k][i]
        tracedict['A'] = trace['A'][i]
        return tracedict

    def run_policy(self, policy, max_trace_len=20, verbose=False):
        state = self.get_curr_state(self.env)
        try:
            action = self.get_action_policy(policy, state)
        except KeyError:
            return False

        trace = dict(zip(self.keys, [list() for k in range(len(self.keys))]))
        trace['A'] = []

        def updateTrace(trace, state):
            j = 0
            for k in self.keys:
                trace[k].append(state[j])
                j += 1
            return trace

        trace = updateTrace(trace, state)
        j = 0
        while True:
            next_state, reward, done, info = self.env.step(
                self.env.env_action_dict[action])
            next_state = self.get_curr_state(self.env)
            trace = updateTrace(trace, next_state)
            state = next_state
            action = self.get_action_policy(policy, state)
            trace['A'].append(str(action))

            # Run the policy as long as the goal is not achieved or less than j
            traceset = trace.copy()
            if self.evaluate_trace(self.goalspec, traceset):
                print('j', traceset)
                return True
            if j > max_trace_len:
                return False
            j += 1
        return False

    def generator(self):
        self.env.restart()
        state = self.get_curr_state(self.env)
        trace = self.create_trace_skeleton(state)

        done = False
        # Trace generator and accumulator
        j = 0

        while not done:
            # Explore action or exploit
            try:
                action = self.get_action(self.gtable_key(state))
            except KeyError:
                self.create_gtable_indv(self.gtable_key(state))
                action = self.get_action(self.gtable_key(state))

            # Addd action to the trace
            try:
                temp = trace['A'][-1].copy()
                temp.append(action)
                trace['A'].append(temp)
            except IndexError:
                trace['A'].append(action)
            # Map the action to env_action
            next_state, reward, done, info = self.env.step(
                self.env.env_action_dict[action])

            nstate = self.get_curr_state(self.env)
            trace = self.trace_accumulator(trace, nstate)
            state = nstate
            if j >= self.max_trace_len:
                break
            j += 1

        return trace

    def recognizer(self, trace):
        """Recognizer.

        Which will reconize traces from the generator system."""
        # parse the formula
        parser = LTLfGParser()

        # Define goal formula/specification
        parsed_formula = parser(self.goalspec)

        # Change list of trace to set
        traceset = trace.copy()
        akey = list(traceset.keys())[0]
        # Loop through the trace to find the shortest best trace
        for i in range(0, len(traceset[akey])):
            t = self.create_trace_flloat(traceset, i)
            result = parsed_formula.truth(t)

            if result is True:
                return True, self.create_trace_dict(trace, i)

        return result, self.create_trace_dict(trace, i)

    def propagate(self, result, trace):
        """Propagate the error to shape the probability."""

        traces = [trace[k][::-1] for k in self.keys]
        tracea = trace['A'][::-1]
        psi = 0.9
        j = 1
        for i in range(0, len(traces[0])-1, 1):
            a = tracea[i]
            tempvals = [t[i+1] for t in traces]
            ss = self.gtable_key(tempvals)
            try:
                prob = self.gtable[ss][a]
            except KeyError:
                self.create_gtable_indv(self.gtable_key(ss))
                prob = self.gtable[ss][a]

            Psi = pow(psi, j)
            j += 1
            if result is False:
                new_prob = prob - (Psi * prob)
            else:
                new_prob = prob + (Psi * prob)

            self.gtable[ss][a] = new_prob
            probs = np.array(list(self.gtable[ss].values()))
            probs = probs / probs.sum()

            self.gtable[ss] = dict(zip(self.gtable[ss].keys(), probs))

    def gen_rec_prop(self, epoch=50):
        # Run the generator, recognizer loop for some epocs
        for _ in range(epoch):
            # Generator
            trace = self.generator()

            # Recognizer
            result, trace = self.recognizer(trace)

            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
