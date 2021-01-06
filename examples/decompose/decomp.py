"""Verifying the decomposition of LTL into BT."""

from utils import goalspec2BT, UntilNode
from py_trees.trees import BehaviourTree
import py_trees


def decompose():
    # goalspec = 'P_[KE][1,none,==] U P_[KA][1,none,==] U P_[KB][1,none,==]'
    # goalspec = 'P_[KE][1,none,==] U P_[KA][1,none,==]'
    goalspec = '(P_[KA][1,none,==] & P_[KB][1,none,==]) U (P_[KC][1,none,==] & P_[KD][1,none,==])'
    root = goalspec2BT(goalspec, planner=None)
    for i in root.iterate():
        print(i.id, i.name)
    behaviour_tree = BehaviourTree(root)
    # bt = UntilNode('U')
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    output = py_trees.display.ascii_tree(behaviour_tree.root)
    print(output)


def main():
    decompose()


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