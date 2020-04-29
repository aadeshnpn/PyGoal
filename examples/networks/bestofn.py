"""Experiment for best-of-N problem with Graph.

Expreriments and results for best-of-N
problem with Graph
"""

import copy
import numpy as np
import pickle
# from joblib import Parallel, delayed

from graphenv import GraphBestofNEnvironment

from pygoal.lib.genrecprop import GenRecProp
from pygoal.utils.bt import goalspec2BT, display_bt
from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict


def env_setup(
        num_agent=10, num_site=2,
        attach_mode='linear',
        detach_mode='linear', seed=1234):
    # num_agent = 10
    # num_site = 2
    # attach_mode = 'importance 2 linear' # choices for attach interface are' always', 'linear', 'exponential', 'importance linear', 'importance 2 linear', 'importance exponential' or \'importance 2 exponential\'
    # detach_mode = 'power law' # choices are 'uniform', 'linear', 'exponential', 'power law', or 'perfect'
    # seed = 1234
    env = GraphBestofNEnvironment(
        num_agent, num_site, attach_mode, detach_mode, seed=seed)
    env.reset()
    return env


# GenRecProp algorithm
class GenRecPropGraph:
    def __init__(
        self, env, keys, goalspec, gtable, max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        self.env = env
        self.keys = keys
        self.goalspec = goalspec
        self.epoch = epoch
        self.gtable = gtable
        self.max_trace_len = max_trace
        self.actionsidx = actions
        self.tcount = 0
        if seed is None:
            self.nprandom = np.random.RandomState()   # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(    # pylint: disable=E1101
                seed)

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
        # Quick fix. create_trace_float requires action to be list of list
        temp = trace['A']
        trace['A'] = [temp]
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

    def parse_goalspec(self):
        """This class will parse goal specification
        and develope a BT structure. """
        pass

    def run_policy(self, max_trace_len=20, verbose=False):
        state = self.get_curr_state(self.env)
        # action = self.get_action_policy(self.gtable, state)
        action = self.get_action(state[0])

        trace = dict(zip(self.keys, [list() for k in range(len(self.keys))]))
        trace['A'] = [action]

        def updateTrace(trace, state):
            j = 0
            for k in self.keys:
                trace[k].append(state[j])
                j += 1
            return trace

        trace = updateTrace(trace, state)
        j = 0
        while True:
            self.env.render()
            next_state, reward, done, info = self.env.step(
                action)

            next_state = self.get_curr_state(self.env)
            trace = updateTrace(trace, next_state)
            state = next_state
            try:
                action = self.get_action(state)
                # action = self.get_action_policy(self.gtable, state)
                trace['A'].append(action)
            # Handeling terminal state
            except KeyError:
                trace['A'].append(9)
            # Run the policy as long as the goal is not achieved or less than j
            traceset = trace.copy()
            if self.evaluate_trace(self.goalspec, traceset):
                print(traceset['G'])
                return True
            if j > max_trace_len:
                return False
            j += 1
        return False

    def generator(self, env_reset=False):
        self.env.reset()
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

            # action = self.get_action(state)
            # Addd action to the trace
            try:
                temp = trace['A'][-1].copy()
                temp.append(action)
                trace['A'].append(temp)
            except IndexError:
                trace['A'].append(action)
            # Map the action to env_action
            # next_state, reward, done, info = self.env.step(
            #     self.env.env_action_dict[action])
            # print(action)
            next_state, reward, _, info = self.env.step(
                action)
            # self.env.showGraph()
            # self.env.render()
            # print(action)
            nstate = self.get_curr_state(self.env)
            # print(j, nstate)
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
        # print('recognizer',traceset)
        akey = list(traceset.keys())[0]
        # Loop through the trace to find the shortest best trace
        for i in range(0, len(traceset[akey])):
            t = self.create_trace_flloat(traceset, i)
            # print(i, t['C'], parsed_formula)
            result = parsed_formula.truth(t)
            # print(i, t['C'], parsed_formula, result)
            if result is True:
                self.set_state(self.env, trace, i)
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

    def env_action_dict(self, action):
        return action

    def get_curr_state(self, env):
        # Things that I need to make the trace
        state = env.state
        site_degree = np.zeros(env.getNumSites()+1)
        unique, count = np.unique(state, return_counts=True)
        for j in range(len(unique)):
            if unique[j] == 0:
                pass
            else:
                site_degree[j] = count[j]
        # print(unique, count, site_degree)
        # print(state, np.max(site_degree))
        return ''.join(map(str, state)), int(np.max(site_degree))

    # Need to work on this
    def set_state(self, env, trace, i):
        pass

    def create_trace_skeleton(self, state):
        # Create a skeleton for trace
        trace = dict(zip(self.keys, [[list()] for i in range(len(self.keys))]))
        j = 0
        for k in self.keys:
            trace[k][0].append(state[j])
            j += 1
        trace['A'] = [list()]
        # Hack to fix the bug of terminal state. Adding big action number
        # This makes the trace length same accross the keys
        trace['A'][0].append(9)
        return trace

    def get_action_policy(self, policy, state):
        # action = policy[tuple(state)]
        action = self.gtable.action(state[0])
        return np.argmax(action.detach().cpu().numpy())

    def gtable_key(self, state):
        # ss = state
        return state[0]

    def get_policy(self):
        policy = dict()
        for s, v in self.gtable.items():
            elem = sorted(v.items(),  key=lambda x: x[1], reverse=True)
            try:
                policy[s] = elem[0][0]
                pass
            except IndexError:
                pass

        return policy

    def train(self, epoch, verbose=False):
        # Run the generator, recognizer loop for some epocs
        # for _ in range(epoch):
        while self.tcount <= epoch:
            # Generator
            trace = self.generator()
            # print(self.tcount, trace['G'][-1])
            # Recognizer
            result, trace = self.recognizer(trace)
            # print(result, trace['G'])
            # No need to propagate results after exciding the train epoch
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1
            print(self.tcount, result)
            # print(trace['A'])
            # print(trace['I'][1]
        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        return self.run_policy(policy, self.max_trace_len)



def main():
    env = env_setup(10, 2)
    keys = ['S', 'C']
    actions = list(range(0, 2+1))
    gtable = dict()
    goalspec = 'F P_[C][5,none,<=]'
    genrecprop = GenRecPropGraph(env, keys, goalspec, gtable, actions=actions, max_trace=40)
    genrecprop.get_curr_state(env)
    # for epoch in range(100):
    #     env.step()
    #     genrecprop.get_curr_state(env)
    #     env.showGraph()
    trace = genrecprop.generator()
    print(gtable)
    result, trace = genrecprop.recognizer(trace)
    # print('recognizer',result, trace)
    genrecprop.propagate(result, trace)
    print(gtable)
    # print(genrecprop.gtable)
    # print(trace['S'][-1], trace['C'][-1])


if __name__ == "__main__":
    main()
