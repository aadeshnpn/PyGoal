"""Genrator, Recognizer, and Propapagator class."""

import numpy as np
from abc import abstractmethod
import copy

from flloat.parser.ltlfg import LTLfGParser
from flloat.semantics.ltlfg import FiniteTrace, FiniteTraceDict
import time


# GenRecProp algorithm
class GenRecProp:
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        self.env = env
        self.keys = keys
        self.goalspec = goalspec
        self.epoch = epoch
        self.gtable = gtable
        self.max_trace_len = max_trace
        self.actionsidx = actions
        if seed is None:
            self.nprandom = np.random.RandomState()   # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(    # pylint: disable=E1101
                seed)

    @abstractmethod
    def gtable_key(self, state):
        raise NotImplementedError

    @abstractmethod
    def get_curr_state(self, env):
        raise NotImplementedError

    @abstractmethod
    def set_state(self, env, trace, i):
        raise NotImplementedError

    @abstractmethod
    def create_trace_skeleton(self, state):
        raise NotImplementedError

    @abstractmethod
    def train(self, epoch=150, verbose=False):
        raise NotImplementedError

    @abstractmethod
    def inference(self, render=False, verbose=False):
        raise NotImplementedError

    @abstractmethod
    def get_action_policy(self, policy, state):
        raise NotImplementedError

    @abstractmethod
    def get_policy(self):
        raise NotImplementedError

    @abstractmethod
    def env_action_dict(self):
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

    def run_policy(self, policy, max_trace_len=20, verbose=False):
        state = self.get_curr_state(self.env)
        try:
            action = self.get_action_policy(policy, state)
        except KeyError:
            if verbose:
                print('State does not exist in the policy', state)
            action = self.nprandom.choice(self.actionsidx)
            # return False
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
        result = False
        while True:
            # next_state, reward, done, info = self.env.step(
            #    self.env.env_action_dict[action])
            if verbose:
                self.env.render()
                time.sleep(1)
            next_state, reward, done, info = self.env.step(
                self.env_action_dict(action))

            next_state = self.get_curr_state(self.env)
            trace = updateTrace(trace, next_state)
            state = next_state
            try:
                action = self.get_action_policy(policy, state)
                trace['A'].append(action)
            # Handeling terminal state
            except KeyError:
                trace['A'].append(9)
            # Run the policy as long as the goal is not achieved or less than j
            # print(j, trace)
            traceset = trace.copy()
            result = self.evaluate_trace(self.goalspec, traceset)
            if self.goalspec[0] == 'G':
                if not result:
                    return result, trace
            else:
                if result:
                    return True, trace
            if j > max_trace_len:
                return result, trace
            j += 1
        return result, trace

    def generator(self, env_reset=False):
        if env_reset:
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
            # next_state, reward, done, info = self.env.step(
            #     self.env.env_action_dict[action])
            next_state, reward, done, info = self.env.step(
                self.env_action_dict(action))

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
            if self.goalspec[0] == 'G':
                if not result:
                    return result, self.create_trace_dict(trace, i)
            else:
                if result:
                    return result, self.create_trace_dict(trace, i)

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


class GenRecPropMDP(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0
        self.trace_len_data = np.ones(epoch) * max_trace

    def env_action_dict(self, action):
        action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        return action_dict[action]

    def set_state(self, env, trace, i):
        state = []
        for k in self.keys:
            temp = trace[k][i][-1]
            state.append(temp)
        env.curr_loc = env.state_dict[state[0]]

    def get_curr_state(self, env):
        # env.format_state(env.curr_loc)
        curr_loc = env.curr_loc
        is_cheese = curr_loc == env.cheese
        # is_trap = curr_loc == env.trap
        # reward = env.curr_reward        # F841
        # return (env.format_state(curr_loc), is_cheese, is_trap, reward)
        # return (env.format_state(curr_loc), is_cheese, is_trap)
        return (env.format_state(curr_loc), is_cheese)

    def create_trace_skeleton(self, state):
        # Create a skeleton for trace
        trace = dict(zip(self.keys, [[list()] for i in range(len(self.keys))]))
        j = 0
        for k in self.keys:
            trace[k][0].append(state[j])
            j += 1
        trace['A'] = [list()]
        return trace

    def get_action_policy(self, policy, state):
        # action = policy[state[0]]
        action = policy[tuple(state)]
        return action

    def gtable_key(self, state):
        # ss = state[0]
        ss = state
        return tuple(ss)

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

        # Generator
        trace = self.generator()

        # Recognizer
        result, trace = self.recognizer(trace)
        # print(result, len(trace), trace)
        self.trace_len_data[self.tcount] = len(trace['IC'])
        # No need to propagate results after exciding the train epoch
        if self.tcount <= epoch:
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1

        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        result, trace = self.run_policy(policy, self.max_trace_len)
        return result


class GenRecPropMDPNear(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0

    def get_curr_state(self, env):
        # env.format_state(env.curr_loc)
        curr_loc = env.curr_loc
        # is_cheese = curr_loc == env.cheese
        # is_trap = curr_loc == env.trap
        near_cheese = env.check_near_object(curr_loc, 'cheese')
        # near_trap = env.check_near_object(curr_loc, 'trap')
        # return (
        #    env.format_state(curr_loc), is_cheese, is_trap,
        #    near_cheese, near_trap)
        return (env.format_state(curr_loc), near_cheese)

    def env_action_dict(self, action):
        action_dict = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1)
        }
        return action_dict[action]

    def set_state(self, env, trace, i):
        state = []
        for k in self.keys:
            temp = trace[k][i][-1]
            state.append(temp)
        env.curr_loc = env.state_dict[state[0]]

    def create_trace_skeleton(self, state):
        # Create a skeleton for trace
        trace = dict(zip(self.keys, [[list()] for i in range(len(self.keys))]))
        j = 0
        for k in self.keys:
            trace[k][0].append(state[j])
            j += 1
        trace['A'] = [list()]
        return trace

    def get_action_policy(self, policy, state):
        # action = policy[state[0]]
        action = policy[tuple(state)]
        return action

    def gtable_key(self, state):
        # ss = state[0]
        ss = state
        return tuple(ss)

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

        # Generator
        trace = self.generator()

        # Recognizer
        result, trace = self.recognizer(trace)
        # No need to propagate results after exciding the train epoch
        if self.tcount <= epoch:
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1

        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        result, trace = self.run_policy(policy, self.max_trace_len)
        return result


class GenRecPropTaxi(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.tcount = 0

    def env_action_dict(self, action):
        return action

    def get_curr_state(self, env):
        temp = list(env.decode(env.s))
        return (str(temp[0])+str(temp[1]), temp[2], temp[3])

    def set_state(self, env, trace, i):
        state = []
        for k in self.keys:
            if k == 'L':
                temp = trace[k][i][-1]
                state.append(int(temp[0]))
                state.append(int(temp[1]))
            else:
                temp = trace[k][i][-1]
                state.append(int(temp))
        state = env.encode(*tuple(state))
        env.env.s = state

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
        action = policy[tuple(state)]
        return action

    def gtable_key(self, state):
        ss = state
        return tuple(ss)

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

        # Generator
        trace = self.generator()
        # Recognizer
        result, trace = self.recognizer(trace)
        # No need to propagate results after exciding the train epoch
        if self.tcount <= epoch:
            # Progagrate the error generate from recognizer
            self.propagate(result, trace)
            # Increment the count
            self.tcount += 1

        return result

    def inference(self, render=False, verbose=False):
        # Run the policy trained so far
        policy = self.get_policy()
        result, trace = self.run_policy(policy, self.max_trace_len)
        return result


class GenRecPropUpdated(GenRecProp):
    def __init__(
        self, env, keys, goalspec, gtable=dict(), max_trace=40,
            actions=[0, 1, 2, 3], epoch=10, seed=None):
        super().__init__(
            env, keys, goalspec, gtable, max_trace, actions, epoch, seed)
        self.trace = dict()
        self.itrace = dict()
        self.env_done = False
        self.parallel_node = False
        self.list_goalspec = []

    # Override generator method
    def generator(self, env_reset=False):
        state = self.get_curr_state(self.env)
        # trace = self.create_trace_skeleton(state)
        try:
            if len(self.trace['A']) >= self.max_trace_len:
                return self.trace
        except KeyError:
            self.trace = self.create_trace_skeleton(state)
        # Explore action or exploit
        try:
            action = self.get_action(self.gtable_key(state))
        except KeyError:
            self.create_gtable_indv(self.gtable_key(state))
            action = self.get_action(self.gtable_key(state))

        # Add action to the trace
        try:
            temp = self.trace['A'][-1].copy()
            temp.append(action)
            self.trace['A'].append(temp)
        except IndexError:
            self.trace['A'].append(action)
        # Map the action to env_action
        # next_state, reward, done, info = self.env.step(
        #     self.env.env_action_dict[action])
        next_state, reward, done, info = self.env.step(
            self.env_action_dict(action))
        # Collect the env done variable
        self.env_done = done
        nstate = self.get_curr_state(self.env)
        self.trace = self.trace_accumulator(self.trace, nstate)
        # state = nstate

        return self.trace

    def recognizer(self, trace):
        """Recognizer.

        Which will reconize traces from the generator system."""
        results = dict()

        def minirecognizer(goalspec):
            # Change list of trace to set
            traceset = trace.copy()
            akey = list(traceset.keys())[0]
            # Loop through the trace to find the shortest best trace
            trace_len = len(traceset[akey])-1
            t = self.create_trace_flloat(traceset, trace_len)

            # parse the formula
            parser = LTLfGParser()
            # Define goal formula/specification
            parsed_formula = parser(goalspec)
            # Evaluate the trace
            result = parsed_formula.truth(t)
            # print(result, goalspec, trace['KE'][-1], trace['LV'][-1])
            if goalspec[0] == 'G':
                if not result:
                    self.env_done = True
                    return result, self.create_trace_dict(
                        trace, trace_len)
            else:
                if result:
                    return result, self.create_trace_dict(
                        trace, trace_len)
            return result, self.create_trace_dict(
                trace, trace_len)
        # print(self.list_goalspec, self.parallel_node)
        if self.parallel_node:
            results = {goal: minirecognizer(
                goal) for goal in self.list_goalspec}
        else:
            results = {goal: minirecognizer(
                goal) for goal in [self.goalspec]}
        # print([r[0] for r in list(results.values())])
        # result = np.all([r[0] for r in list(results.values())])
        # result = [r[0] for r in list(results.values())]
        # return result, results[self.goalspec][1]
        return results
        # result = parsed_formula.truth(t)
        # if self.goalspec[0] == 'G':
        #     if not result:
        #         return result, self.create_trace_dict(
        #             trace, trace_len)
        # else:
        #     if result:
        #         return result, self.create_trace_dict(
        #             trace, trace_len)

    def propagate(self, results, trace):
        """Propagate the error to shape the probability."""
        traces = [trace[k][::-1] for k in self.keys]
        print(traces[0])
        tracea = trace['A'][::-1]
        action_struc = {'F': {True: 2, False: -1}, 'G': {True: 1, False: -2}}
        goals = [r[0] for r in list(results.keys())]
        vals = [r[0] for r in list(results.values())]
        result = bool(np.all([r[0] for r in list(results.values())]))
        gamma = np.sum(
            [action_struc[goals[i]][vals[i]] for i in range(
                len(goals))])
        # psi = 0.9
        psi = 1.0 / (1+np.exp(-1 * np.abs(gamma)))

        def func1(p, j):
            return pow(p, j)

        def func2(p, j):
            return pow(p, pow(j, j))

        print([r[0] for r in results.values()], trace['KE'], trace['LV'], psi)
        func_struc = {'F': func1, 'G': func2}

        j = 1
        for i in range(0, len(traces[0])-1, 1):
            a = tracea[i]
            tempvals = [t[i] for t in traces]
            ss = self.gtable_key(tempvals)
            try:
                prob = self.gtable[ss][a]
            except KeyError:
                self.create_gtable_indv(self.gtable_key(ss))
                prob = self.gtable[ss][a]

            # Psi = pow(psi, j)
            Psi = np.average(
                    [func_struc[goals[i]](psi, j) for i in range(
                        len(goals))])
            j += 1
            # print(i, Psi, end=" ")
            # if j == 2:
            if result is False:
                new_prob = prob - (Psi * prob)
            else:
                new_prob = prob + (Psi * prob)
            print(j, type(result), self.gtable[ss], ss, a, Psi, prob, new_prob, end="\n")
            self.gtable[ss][a] = new_prob
            probs = np.array(list(self.gtable[ss].values()))
            probs = probs / probs.sum()

            self.gtable[ss] = dict(zip(self.gtable[ss].keys(), probs))
            # if j == 2:
            # print(j, prob, new_prob, probs, end='\n')

    def run_policy(self, policy, max_trace_len=20, verbose=False):
        state = self.get_curr_state(self.env)

        try:
            action = self.get_action_policy(policy, state)
        except KeyError:
            if verbose:
                print('State does not exist in the policy', state)
            action = self.nprandom.choice(self.actionsidx)

        def updateTrace(trace, state):
            j = 0
            for k in self.keys:
                trace[k].append(state[j])
                j += 1
            return trace

        try:
            if len(self.itrace['A']) >= self.max_trace_len:
                return self.itrace
        except KeyError:
            self.itrace = dict(
                zip(self.keys, [list() for k in range(len(self.keys))]))
            self.itrace['A'] = [action]
            self.itrace = updateTrace(self.itrace, state)
        # print(self.itrace)

        # j = 0
        result = False
        if verbose:
            self.env.render()
            time.sleep(1)
        next_state, reward, done, info = self.env.step(
            self.env_action_dict(action))

        next_state = self.get_curr_state(self.env)
        self.itrace = updateTrace(self.itrace, next_state)
        state = next_state
        try:
            action = self.get_action_policy(policy, state)
            self.itrace['A'].append(action)
        # Handeling terminal state
        except KeyError:
            self.itrace['A'].append(9)
        # Run the policy as long as the goal is not achieved or less than j
        # print(j, trace)
        traceset = self.itrace.copy()
        result = self.evaluate_trace(self.goalspec, traceset)
        if self.goalspec[0] == 'G':
            if not result:
                return result, self.itrace
        else:
            if result:
                return True, self.itrace
        if len(self.itrace['A']) > max_trace_len:
            return result, self.itrace

        return result, self.itrace
