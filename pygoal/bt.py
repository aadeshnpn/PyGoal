"""Behavior Tree nodes definition."""

from flloat.parser.ltlfg import LTLfGParser
from py_trees import Behaviour, Blackboard, Status


class Policy(Behaviour):
    """Policy exection for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior implements the exection of policy for the agents. This allows
    the agents to execute the policy.
    """

    def __init__(self, name):
        """Init method for the policy behavior."""
        super(Policy, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.frames = []

    def setup(self, timeout, env, policy, goalspec, render=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        policy. Env is the current environment,
        policy is the item the agent need to execute in the envrionment.
        """
        self.env = env
        self.policy = policy
        self.goalspec = goalspec
        self.render = render

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def run_policy(self):
        state = get_curr_state(self.env)
        if self.render:
            self.blackboard.frames.append(self.env.render(mode='ansi'))
        # print(state, self.goalspec)
        try:
            action = self.policy[state]
        except KeyError:
            return False
        trace = {
            'T':list(), 'PI':list(), 'DI':list(),
            'A': list()
            }

        def updateTrace(trace, state):
            trace['T'].append(state[0])
            trace['PI'].append(state[1])
            trace['DI'].append(state[2])
            return trace

        trace = updateTrace(trace, state)
        j = 0
        while True:
            next_state, reward, done, info = self.env.step(action)
            if self.render:
                self.blackboard.frames.append(self.env.render(mode='ansi'))
            next_state = get_curr_state(self.env)
            trace = updateTrace(trace, next_state)
            state = next_state
            action = self.policy[state]
            # Run the policy as long as the goal is not achieved or less than j
            traceset = trace.copy()
            if evaluate_trace(self.goalspec, traceset):
                return True
            if j>30:
                return False
            j += 1
        return False

    def update(self):
        """
        Execute the Policy.

        This method executes the policy until the goal is fulfilled.
        """
        if self.run_policy():
            # print('BT update', True, self.goalspec)
            return Status.SUCCESS
        else:
            # print('BT update', False)
            return Status.FAILURE


class PolicyNode(Behaviour):
    """Policy learning for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior implements training of policy for the agents. This allows
    the agents to execute the policy.
    """

    def __init__(self, name):
        """Init method for the policy behavior."""
        super(PolicyNode, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, goalspec, gtable, train_epoch=150, verbose=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        policy. Env is the current environment,
        policy is the item the agent need to execute in the envrionment.
        """
        self.goalspec = goalspec
        self.gtable = gtable
        self.n = 0
        self.train_epoch = train_epoch
        self.verbose = verbose

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def run_genrecprop(self):
        """Run the algorithm.

        Run the generator, recognizer, propagate  algorithm.
        """
        self.trace, self.gtable = generator(self.env, self.gtable)
        result, self.trace = recognizer(self.trace, self.goalspec)
        # self.gtable = propagate(result, self.trace, self.gtable, self.env)
        # Check if the policy is alread capable of finding the goal
        # If that is true no need to update the policy
        self.evaluate_trace()
        if not self.result:
            self.gtable = propagate(result, self.trace, self.gtable, self.env)
            self.n += 1
        elif self.result and nprandom.rand() > 0.5:
            self.gtable = propagate(result, self.trace, self.gtable, self.env)
            self.n += 1
        else:
            pass


    def evaluate_trace(self):
        """Evaluate the trace.

        Evaluate the trace obtained by running
        GenRecProp algorithm.
        """
        # Test the prop algorithm
        parser = LTLfGParser()
        parsed_formula = parser(self.goalspec)

        # Create a trace compatiable with Flloat library
        t = create_trace_flloat(list_to_trace(self.trace.copy()), 0)

        self.result = parsed_formula.truth(t)

    def update(self):
        """
        Execute the GenRecProc algorithm.

        This method executes a step of GenRecProp algorithm.
        """
        # Since the environment needs to be shared around BT nodes
        self.env = self.blackboard.shared_content['env']
        self.run_genrecprop()
        self.evaluate_trace()
        # print(self.name, self.result, end=' ')

        if self.result:
            # If the goal is achieved, store the policy
            policy = get_policy(self.gtable)
            self.blackboard.policies[self.name] = policy
            if self.verbose:
                print(self.n, self.name, Status.SUCCESS)
            return Status.SUCCESS
        else:
            if self.verbose:
                print(self.n, self.name, Status.FAILURE)
            return Status.FAILURE


class DummyNode(Behaviour):
    """Policy exection for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior implements the exection of policy for the agents. This allows
    the agents to execute the policy.
    """

    def __init__(self, name):
        """Init method for the policy behavior."""
        super(DummyNode, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.frames = []

    def setup(self, timeout):
        """Have defined the setup method.

        This method defines the other objects required for the
        policy. Env is the current environment,
        policy is the item the agent need to execute in the envrionment.
        """
        pass

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Execute the Policy.

        This method executes the policy until the goal is fulfilled.
        """
        pass
