"""Utility functions."""

from flloat.parser.ltlfg import LTLfGParser
from flloat.syntax.ltlfg import (
    LTLfgAtomic, LTLfEventually, LTLfAlways
)
from py_trees.composites import Sequence, Selector, Parallel

from pygoal.lib.bt import DummyNode


# Recursive script to build a BT from LTL specs
def rparser(formula, root):
    if len(formula) > 2:
        for i in range(len(formula)):
            root.add_children([DummyNode(str(formula[i]))])
    elif len(formula) == 2:
        if type(formula[0]) not in [
                LTLfEventually, LTLfAlways, LTLfgAtomic]:
            op = find_control_node(formula[0].operator_symbol)
            root.add_children([rparser(formula[0].formulas, op)])
        else:
            # Creat BT execution node
            root.add_children([DummyNode(str(formula[0]))])
        if type(formula[1]) not in [
                LTLfEventually, LTLfAlways, LTLfgAtomic]:
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
    if type(ltlformula) in [LTLfgAtomic, LTLfEventually, LTLfAlways]:
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
