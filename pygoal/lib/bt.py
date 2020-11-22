"""Behavior Tree nodes definition."""

from py_trees import Behaviour, Blackboard, Status
from pygoal.lib.planner import Planner
from py_trees.composites import Parallel


class GoalNode(Behaviour):
    """Policy learning for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior implements training of policy for the agents. This allows
    the agents to execute the policy.
    """

    def __init__(self, name, planner, train=True):
        """Init method for the policy behavior."""
        super(GoalNode, self).__init__(name)
        self.blackboard = Blackboard()
        self.planner = planner
        self.train = train
        self.blackboard.shared_content = dict()

    def setup(
            self, timeout, planner=Planner.DEFAULT,
            train=True, epoch=20, verbose=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        policy. Env is the current environment,
        policy is the item the agent need to execute in the envrionment.
        """
        self.planner = planner
        self.goalspec = self.name
        self.planner.goalspec = self.goalspec
        self.n = 0
        self.train = train
        self.planner.epoch = epoch
        # # This is the first node to get the environment
        # if (
        #     'env' not in self.blackboard.shared_content.keys()
        #         ):
        #     self.blackboard.shared_content['env'] = self.planner.env
        #     # self.blackboard.policies = dict()
        # # This is second node and sequence
        # elif (
        #     'env' in self.blackboard.shared_content.keys()
        #         and type(self.parent) != Sequence):
        #     self.blackboard.shared_content['env'] = self.planner.env

        # self.planner.env = self.blackboard.shared_content['env']

        self.verbose = verbose

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def run_planner(self):
        """Run the algorithm.

        Run the planner.
        """
        if self.train:
            return self.planner.train(self.planner.epoch)
        else:
            # print('Inference')
            return self.planner.inference()

    def update(self):
        """
        Execute the GenRecProc algorithm.

        This method executes a step of GenRecProp algorithm.
        """
        # Since the environment needs to be shared around BT nodes
        # elf.env = self.blackboard.shared_content['env']
        result = self.run_planner()
        if result:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class CompetentNode(Behaviour):
    """Policy learning for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior implements training of policy for the agents. This allows
    the agents to execute the policy.
    """

    def __init__(self, name, planner, train=True):
        """Init method for the policy behavior."""
        super(CompetentNode, self).__init__(name)
        self.blackboard = Blackboard()
        self.planner = planner
        self.train = train
        try:
            self.blackboard.shared_content['curve'][name] = [0, 0, 0]
        except AttributeError:
            self.blackboard.shared_content = dict()
            self.blackboard.shared_content['curve'] = dict()
        try:
            self.blackboard.shared_content[
                'ctdata'][name] = list()
            self.blackboard.shared_content[
                'cidata'][name] = list()
        except KeyError:
            self.blackboard.shared_content['ctdata'] = dict()
            self.blackboard.shared_content['cidata'] = dict()
            self.blackboard.shared_content[
                'ctdata'][name] = list()
            self.blackboard.shared_content[
                'cidata'][name] = list()

    def setup(
            self, timeout, planner=Planner.DEFAULT,
            train=True, epoch=20, verbose=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        policy. Env is the current environment,
        policy is the item the agent need to execute in the envrionment.
        """
        self.planner = planner
        self.goalspec = planner.goalspec
        # self.planner.goalspec = self.goalspec
        self.n = 0
        self.train = train
        self.planner.epoch = epoch

        # # This is the first node to get the environment
        # if (
        #     'env' not in self.blackboard.shared_content.keys()
        #         ):
        #     self.blackboard.shared_content['env'] = self.planner.env
        #     # self.blackboard.policies = dict()
        # # This is second node and sequence
        # elif (
        #     'env' in self.blackboard.shared_content.keys()
        #         and type(self.parent) != Sequence):
        #     self.blackboard.shared_content['env'] = self.planner.env

        # self.planner.env = self.blackboard.shared_content['env']

        self.verbose = verbose

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def run_planner(self):
        """Run the algorithm.

        Run the planner.
        """
        # Trick
        if isinstance(self.parent, Parallel):
            self.planner.gtable = self.parent.children[0].planner.gtable

        if self.train:
            return self.planner.train(self.planner.epoch)
        else:
            # print('Inference')
            return self.planner.inference(self.planner.epoch)

    def update(self):
        """
        Execute the GenRecProc algorithm.

        This method executes a step of GenRecProp algorithm.
        """
        # Since the environment needs to be shared around BT nodes
        # elf.env = self.blackboard.shared_content['env']
        result = self.run_planner()
        if result:
            return Status.SUCCESS
        else:
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
