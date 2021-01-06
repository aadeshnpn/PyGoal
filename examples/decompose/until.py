from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.trees import BehaviourTree
from py_trees import Blackboard, Status, Behaviour
import py_trees


class CondNode(Behaviour):
    """Condition node for the proving decomposition.

    Inherits the Behaviors class from py_trees. This
    behavior implements the condition node for the Until LTL.
    """

    def __init__(self, name):
        """Init method for the condition node."""
        super(CondNode, self).__init__(name)
        self.blackboard = Blackboard()
        try:
            self.blackboard.nodes[name] = self
        except AttributeError:
            self.blackboard.nodes = dict()
            self.blackboard.nodes[name] = self
        self.value = True

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


def skeleton():
    main = Sequence('1')
    selec = Selector('2')
    p2 = CondNode('p2')
    p1 = CondNode('p1')
    goal1 = LTLNode('g1')
    goal2 = LTLNode('g2')
    selec.add_children([p2, goal1])
    seq = Sequence('3')
    seq.add_children([p1, goal2])
    main.add_children([selec, seq])
    root = BehaviourTree(main)
    return [root, p1, p2, goal1, goal2]


def setup_nodes(val1, val2, goal1, goal2):
    goal1.value = val1
    goal2.value = val2


def post_handler(p1, p2, goal1, goal2):
    blackboard = Blackboard()
    # print(blackboard.nodes)
    blackboard.nodes[p1.name].value = blackboard.nodes[goal1.name].value
    blackboard.nodes[p2.name].value = blackboard.nodes[goal2.name].value


def testuntil(t1, name):
    root, p1, p2, goal1, goal2 = skeleton()
    parser = LTLfParser()
    goalspec = 'a U b'
    ltlformula = parser(goalspec)
    # t1 = [{
    #     'a': False, 'b': False
    # }]
    print(name)
    print('-' * 50)
    print(ltlformula.truth(t1), t1)
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    # output = py_trees.display.ascii_tree(root.root)
    # print(output)
    # setup_nodes(t1[0]['a'], t1[0]['b'], goal1, goal2)
    for k in range(len(t1)):
        setup_nodes(t1[k]['a'], t1[k]['b'], goal1, goal2)
        root.tick()
        post_handler(p1, p2, goal1, goal2)
    if root.root.status == Status.FAILURE:
        print(False, Status.FAILURE)
    else:
        print(True, Status.SUCCESS)
    print('-' * 50)


def main():
    t1 = [{
        'a': False, 'b': False
    }]
    t2 = [{
        'a': False, 'b': True
    }]
    t3 = [{
        'a': True, 'b': False
    }]
    t4 = [{
        'a': True, 'b': True
    }]
    t5 = [
        {'a': False, 'b': False},
        {'a': True, 'b': True}
        ]
    t6 = [
        {'a': True, 'b': False},
        {'a': True, 'b': True}
        ]
    t = [t1, t2, t3, t4, t5, t6]
    name = ['one', 'two', 'three', ' four', 'five', 'six']
    for i in range(len(t)):
        testuntil(t[i], name[i])


if __name__ == "__main__":
    main()
