"""Verifying the decomposition of LTL into BT."""

from utils import goalspec2BT, recursive_until, LTLNode, post_tick_until
# from pygoal.utils.bt import goalspec2BT, recursive_until
from py_trees.trees import BehaviourTree
import py_trees


def decompose():
    # goalspec = 'P_[KE][1,none,==] U P_[KA][1,none,==]'
    # goalspec = 'P_[KE][1,none,==] U P_[KA][1,none,==] U P_[KB][1,none,==]'
    # goalspec = 'P_[KA][1,none,==] U P_[KB][1,none,==] U P_[KC][1,none,==] U P_[KD][1,none,==], U P_[KE][1,none,==]'
    goalspec = '(F(P_[KE][1,none,==]) U G(P_[KA][1,none,==]))'
    # goalspec = '(P_[KA][1,none,==] & P_[KB][1,none,==]) U (P_[KC][1,none,==] & P_[KD][1,none,==])'
    # goalspec = '((P_[KA][1,none,==] U P_[KB][1,none,==]) & (P_[KC][1,none,==] U P_[KD][1,none,==])) & (P_[KE][1,none,==] & P_[KF][1,none,==])'
    root = goalspec2BT(goalspec, planner=None)
    # for i in root.iterate():
    #     print(i.id, i.name)
    behaviour_tree = BehaviourTree(root)
    # bt = UntilNode('U')
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(behaviour_tree.root)
    print(output)
    recursive_until(root)
    output = py_trees.display.ascii_tree(behaviour_tree.root)
    print(output)


def test_decompose_tick():
    goalspec = 'P_[KE][1,none,==] U P_[KA][1,none,==]'
    root = goalspec2BT(goalspec, planner=None, node=LTLNode)
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    behaviour_tree = BehaviourTree(root)
    recursive_until(root)
    # output = py_trees.display.ascii_tree(behaviour_tree.root)
    # print(output)
    return behaviour_tree


def test_multiple_ticks():
    ticks = [
        [False, True],
        [False, False],
        [True, False],
        [True, True]
        ]
    for tick in ticks:
        behavior_tree = test_decompose_tick()
        ltlnode = [
            node for node in behavior_tree.root.iterate() if isinstance(
                node, LTLNode)]
        # Now we have a behavior tree. Lets define the tick
        # This order is important for testing only. In real
        # setting the status of the node is used
        ltlnode[0].value, ltlnode[1].value = tick[1], tick[0]
        behavior_tree.tick()
        print(behavior_tree.root.status, tick)

    ticks1 = [
        [True, False],
        [True, True]
        ]
    ticks2 = [
        [True, False],
        [False, True]
        ]
    ticks3 = [
        [True, False],
        [False, False]
        ]
    ticks4 = [
        [True, False],
        [True, False]
        ]
    ticks5 = [
        [True, False],
        [False, False],
        [True, False],
        [True, True],
        ]

    ticks = [ticks1, ticks2, ticks3, ticks4, ticks5]
    for tick in ticks:
        behavior_tree = test_decompose_tick()
        ltlnode = [
            node for node in behavior_tree.root.iterate() if isinstance(
                node, LTLNode)]
        # Now we have a behavior tree. Lets define the tick
        for i in range(len(tick)):
            ltlnode[0].value, ltlnode[1].value = tick[i][1], tick[i][0]
            # print(ltlnode[0].value, ltlnode[1].value)
            behavior_tree.tick()
            # print(i, behavior_tree.root.status, tick[i])
            post_tick_until(behavior_tree.root)
        print(behavior_tree.root.status, tick)


def main():
    # decompose()
    # test_decompose_tick()
    test_multiple_ticks()


def older():
    # from examples.decompose.utils import DummyNode

    from flloat.parser.ltlf import LTLfParser

    # parse the formula
    parser = LTLfParser()
    # formula = "r U (t U (b U p))"  # F(b) U F(p)"
    # formula = "p U (b U (t U r))"
    formula = "((F r U t) U F b) U F p"
    parsed_formula = parser(formula)
    print(parsed_formula)
    # evaluate over finite traces
    t1 = [
        {"r": True, "t": False, "b": False, "p": False},
        {"r": True, "t": False, "b": False, "p": False},
        {"r": False, "t": True, "b": False, "p": False},
        {"r": False, "t": False, "b": True, "p": False},
        {"r": False, "t": False, "b": False, "p": True},
    ]
    # t1 = [
    #     {"r": False, "t": True, "b": False, "p": False},
    #     {"r": True, "t": False, "b": False, "p": False},
    #     {"r": False, "t": False, "b": False, "p": False},
    #     {"r": False, "t": False, "b": True, "p": False},
    #     {"r": False, "t": False, "b": False, "p": True}
    #     ]

    print('t1', parsed_formula.truth(t1, 0))

    # t2 = [
    #     {"a": False, "b": False},
    #     {"a": True, "b": True},
    #     {"a": False, "b": True},
    # ]
    # assert not parsed_formula.truth(t2, 0)

    # # from LTLf formula to DFA
    dfa = parsed_formula.to_automaton()
    # print(dir(dfa))
    print(dfa.get_transitions(), dfa.size)
    # assert dfa.accepts(t1)
    # assert not dfa.accepts(t2)

    # # print the automaton
    graph = dfa.to_graphviz()
    graph.render("./1")  # requires Graphviz installed on your system.


if __name__ == "__main__":
    main()