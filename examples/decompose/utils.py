# import os
# from pathlib import Path
# import numpy as np
# from py_trees.trees import BehaviourTree
# from py_trees.common import Status

# from joblib import Parallel, delayed

# import matplotlib

# # If there is $DISPLAY, display the plot
# if os.name == 'posix' and "DISPLAY" not in os.environ:
#     matplotlib.use('Agg')

# import matplotlib.pyplot as plt     # noqa: E402
# # matplotlib.rc('text', usetex=True)


from flloat.parser.ltlf import LTLfGParser
from flloat.ltlf import (
    LTLfAtomic, LTLfEventually, LTLfAlways)
import py_trees

from py_trees.composites import Sequence, Selector, Parallel


from py_trees import Behaviour, Blackboard


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


# Recursive script to build a BT from LTL specs
def rparser(formula, root, planner):
    if len(formula) > 2:
        for i in range(len(formula)):
            root.add_children([DummyNode(str(formula[i]))])
    elif len(formula) == 2:
        if type(formula[0]) not in [
                LTLfEventually, LTLfAlways, LTLfAtomic]:
            op = find_control_node(formula[0].operator_symbol)
            root.add_children([rparser(formula[0].formulas, op)])
        else:
            # Creat BT execution node
            root.add_children([DummyNode(str(formula[0]))])
        if type(formula[1]) not in [
                LTLfEventually, LTLfAlways, LTLfAtomic]:
            op = find_control_node(formula[1].operator_symbol)
            root.add_children([rparser(formula[0].formulas, op)])
        else:
            root.add_children([DummyNode(str(formula[1]))])

    elif len(formula) == 1:
        root.add_children([DummyNode(str(formula))])
    return root


def goalspec2BT(goalspec):
    parser = LTLfGParser()
    ltlformula = parser(goalspec)
    # If the specification is already atomic no need to call his
    if type(ltlformula) in [LTLfAtomic, LTLfEventually, LTLfAlways]:
        # root = DummyNode(str(ltlformula), planner)
        root = DummyNode(str(ltlformula))
    else:
        rootnode = find_control_node(ltlformula.operator_symbol)
        root = rparser(ltlformula.formulas, rootnode)

    return root


def find_control_node(operator):
    # print(operator, type(operator))
    if operator in ['U']:
        # sequence
        control_node = Sequence(operator)
    elif operator == '&':
        # parallel
        control_node = Parallel(operator)
    elif operator == '|':
        # Selector
        control_node = Selector(operator)
    else:
        # decorator
        control_node = Selector(operator)
    return control_node


def display_bt(behaviour_tree, save=False):
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(behaviour_tree.root)
    print(output)
    if save:
        py_trees.display.render_dot_tree(
            behaviour_tree.root,
            py_trees.common.VisibilityLevel.DETAIL,
            name='/tmp/'+behaviour_tree.root.name)

