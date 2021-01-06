from flloat.parser.ltlf import LTLfParser

from py_trees.composites import Sequence, Selector
from py_trees.trees import BehaviourTree
from py_trees import Blackboard, Status, Behaviour


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
    blackboard.nodes[p1.name].value = blackboard.nodes[goal2.name].value
    blackboard.nodes[p2.name].value = blackboard.nodes[goal1.name].value


def testone():
    root, p1, p2, goal1, goal2 = skeleton()
    parser = LTLfParser()
    goalspec = 'a U b'
    ltlformula = parser(goalspec)
    t1 = [{
        'a': False, 'b': False
    }]
    print(ltlformula.truth(t1), t1)

    setup_nodes(t1[0]['a'], t1[0]['b'], goal1, goal2)
    root.tick(
       post_tick_handler=post_handler(p1, p2, goal1, goal2)
    )
    print(root.root.status)


def main():
    testone()


if __name__ == "__main__":
    main()
