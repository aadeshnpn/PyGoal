"""Utility function."""

from flloat.parser.ltlfg import LTLfGParser
from flloat.syntax.ltlfg import (
    LTLfgAtomic, LTLfEventually, LTLfAlways)
import py_trees
from py_trees.composites import Sequence, Selector, Parallel

from pygoal.lib.bt import GoalNode
from pygoal.lib.planner import Planner


# Recursive script to build a BT from LTL specs
def rparser(formula, root, planner):
    if len(formula) > 2:
        for i in range(len(formula)):
            root.add_children([GoalNode(str(formula[i]), planner)])
    elif len(formula) == 2:
        if type(formula[0]) not in [
                LTLfEventually, LTLfAlways, LTLfgAtomic]:
            op = find_control_node(formula[0].operator_symbol)
            root.add_children([rparser(formula[0].formulas, op, planner)])
        else:
            # Creat BT execution node
            root.add_children([GoalNode(str(formula[0]), planner)])
        if type(formula[1]) not in [
                LTLfEventually, LTLfAlways, LTLfgAtomic]:
            op = find_control_node(formula[1].operator_symbol)
            root.add_children([rparser(formula[0].formulas, op, planner)])
        else:
            root.add_children([GoalNode(str(formula[1]), planner)])

    elif len(formula) == 1:
        root.add_children([GoalNode(str(formula), planner)])
    return root


def goalspec2BT(goalspec, planner=Planner.DEFAULT):
    parser = LTLfGParser()
    ltlformula = parser(goalspec)
    # If the specification is already atomic no need to call his
    if type(ltlformula) in [LTLfgAtomic, LTLfEventually, LTLfAlways]:
        # root = DummyNode(str(ltlformula), planner)
        root = GoalNode(str(ltlformula), planner)
    else:
        rootnode = find_control_node(ltlformula.operator_symbol)
        root = rparser(ltlformula.formulas, rootnode, planner)

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
