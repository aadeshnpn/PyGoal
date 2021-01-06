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

from py_trees import Behaviour, Blackboard, Status, BehaviourTree

from flloat.parser.ltlfg import LTLfGParser
from flloat.syntax.ltlfg import (
    LTLfgAtomic, LTLfEventually, LTLfAlways)
import py_trees
from py_trees.composites import (
    Sequence, Selector, Parallel)

from pygoal.lib.bt import GoalNode
from pygoal.lib.planner import Planner


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


class UntilNode(Behaviour):
    """Until Sub-tree.
    Until sub-tree is eqivalent to the Until LTL operator.
    """

    def __init__(self, name):
        """Init method for the Until sub-tree.
        Until sub-tree has following structure
        Sequence
         - Selector
            -p_2
            -\\phi_1
         - Sequence
            -p_1
            -\\phi_2
        """
        super(UntilNode, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        # Define a sequence to combine the primitive behavior
        root = Sequence('U')
        selec = Selector('Se')
        p2 = DummyNode('p2')
        p1 = DummyNode('p1')
        goal1 = DummyNode('g1')
        goal2 = DummyNode('g2')
        selec.add_children([p2, goal1])
        seq = Sequence('S')
        seq.add_children([p1, goal2])
        root.add_children([selec, seq])
        self.bt = BehaviourTree(root)

    def setup(self, timeout):
        """Have defined the setup method.
        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        pass

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.
        This will execute the primitive behaviors defined in the sequence
        """
        self.bt.tick()
        return self.bt.root.status


def get_name(formula):
    # return [''.join(a) for a in list(formula.find_labels())][0]
    return str(formula)


def create_until_node(a, b):
    root = Sequence('Seq')
    selec = Selector('Se')
    p2 = CondNode('C' + b.name)
    p1 = CondNode('C' + a.name)
    goal1 = a
    goal2 = b
    selec.add_children([p2, goal1])
    seq = Sequence('S')
    seq.add_children([p1, goal2])
    root.add_children([selec, seq])
    return root


def recursive_until(node):
    if node.children:
        if (isinstance(node, Sequence) and node.name == 'U'):
            fchild = node.children[0]
            for i in range(len(node.children)-1):
                schild = node.children[i+1]
                subtree = create_until_node(fchild, schild)
                node.remove_child(fchild)
                node.remove_child(schild)
                node.add_children([subtree])
                fchild = subtree
                if (isinstance(node.children[i], Sequence) and node.children[i].name == 'U'):
                    # Control nodes
                    # fn_c(c)
                    print(node.children[i].name)
                    recursive_until(node.children[i])
            # recursive_setup(c, fn_e, fn_c)


def recursive_fix_until(root):
    pass


def until_subtree_fix2(child, formula, planner, node, id):
    if child.name == 'p2':
        par = child.parent
        par.replace_child(child, CondNode('C' + get_name(formula[1])))

    elif child.name == 'g1':
        par = child.parent
        par.replace_child(child, node(get_name(formula[0]), planner, id=id))

    elif child.name == 'p1':
        par = child.parent
        par.replace_child(child, CondNode('C' + get_name(formula[0])))

    elif child.name == 'g2':
        par = child.parent
        par.replace_child(child, node(get_name(formula[1]), planner, id=id))


# Recursive script to build a BT from LTL specs
def rparser(formula, root, planner, node, nid):
    if len(formula) > 2:
        for i in range(len(formula)):
            root.add_children([node(get_name(formula[i]), planner, id=nid)])
            nid += 1
    elif len(formula) == 2:
        if type(formula[0]) not in [
                LTLfEventually, LTLfAlways, LTLfgAtomic]:
            op = find_control_node(formula[0].operator_symbol)
            root.add_children(
                [rparser(formula[0].formulas, op, planner, node, nid)])
        else:
            # Creat BT execution node
            root.add_children([node(get_name(formula[0]), planner, id=nid)])
            nid += 1
        if type(formula[1]) not in [
                LTLfEventually, LTLfAlways, LTLfgAtomic]:
            op = find_control_node(formula[1].operator_symbol)
            root.add_children(
                [rparser(formula[0].formulas, op, planner, node, nid)])
        else:
            root.add_children([node(get_name(formula[1]), planner, id=nid)])
            nid += 1
    elif len(formula) == 1:
        root.add_children([node(get_name(formula), planner, id=nid)])
        nid += 1
    return root


def goalspec2BT(goalspec, planner=Planner.DEFAULT, node=GoalNode):
    parser = LTLfGParser()
    ltlformula = parser(goalspec)
    # nodeid for unique naming to the nodes
    nid = 0
    # If the specification is already atomic no need to call his
    if type(ltlformula) in [LTLfgAtomic, LTLfEventually, LTLfAlways]:
        # root = DummyNode(str(ltlformula), planner)
        root = node(get_name(ltlformula), planner, id=nid)
        nid += 1
    else:
        rootnode = find_control_node(ltlformula.operator_symbol)
        print(ltlformula.formulas)
        root = rparser(ltlformula.formulas, rootnode, planner, node, nid)

    return root


def find_control_node(operator):
    # print(operator, type(operator))
    if operator in ['U']:
        # sequence
        # control_node = UntilNode(operator)
        control_node = Sequence(operator)
    elif operator == '&':
        # parallel
        control_node = Sequence(operator)
        # control_node = Deco
    # elif operator == '|':
    #     # Selector
    #     control_node = Selector(operator)
    # else:
    #     # decorator
    #     control_node = Selector(operator)
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


def reset_env(env):
    env.restart()


class CondNode(Behaviour):
    """Condition node for the proving decomposition.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the Until LTL.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(CondNode, self).__init__(name, id=0)
        self.blackboard = Blackboard()
        try:
            self.blackboard.nodes[name] = self
        except AttributeError:
            self.blackboard.nodes = dict()
            self.blackboard.nodes[name] = self
        self.value = True
        self.id = id

    def setup(self, timeout, value):
        """Have defined the setup method.

        This method defines the other objects required for the
        condition node. value is the only property.
        """
        self.value = value

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Return the value.
        """
        if self.value:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class LTLNode(Behaviour):
    """LTL node for the proving decomposition.

    Inherits the Behaviors class from py_trees. This
    behavior implements the LTL node for the Until LTL.
    """

    def __init__(self, name):
        """Init method for the LTL node."""
        super(LTLNode, self).__init__(name)
        self.blackboard = Blackboard()
        try:
            self.blackboard.nodes[name] = self
        except KeyError:
            self.blackboard.nodes = dict()
            self.blackboard.nodes[name] = self
        self.goalspec = None

    def setup(self, timeout, goalspec, value=False):
        """Have defined the setup method.

        This method defines the other objects required for the
        LTL node. LTL specfication is the only property.
        """
        self.goalspec = goalspec
        self.value = value

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Return the value.
        """
        # parser = LTLfParser()
        # ltlformula = parser(self.goalspec)
        if self.value:
            return Status.SUCCESS
        else:
            return Status.FAILURE
