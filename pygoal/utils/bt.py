"""Utility function."""

from flloat.parser.ltlfg import LTLfGParser
from flloat.syntax.ltlfg import (
    LTLfgAtomic, LTLfEventually, LTLfAlways)
import py_trees
from py_trees.composites import Sequence, Selector, Parallel

from pygoal.lib.bt import GoalNode
from pygoal.lib.planner import Planner


def get_name(formula):
    # return [''.join(a) for a in list(formula.find_labels())][0]
    return str(formula)


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
        root = rparser(ltlformula.formulas, rootnode, planner, node, nid)

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


def reset_env(env):
    env.restart()
