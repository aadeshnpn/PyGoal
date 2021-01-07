"""Utility function."""

import copy
from flloat.parser.ltlfg import LTLfGParser
from flloat.syntax.ltlfg import (
    LTLfgAtomic, LTLfEventually, LTLfAlways)
import py_trees
from py_trees.composites import (
    Sequence, Selector)

from pygoal.lib.bt import GoalNode, ConditionNode
from pygoal.lib.planner import Planner


def get_name(formula):
    # return [''.join(a) for a in list(formula.find_labels())][0]
    return str(formula)


# Recursive script to build a BT from LTL specs
def rparser(formula, root, planner, node, nid):
    if len(formula) >= 2:
        for i in range(len(formula)):
            if type(formula[i]) not in [
                    LTLfEventually, LTLfAlways, LTLfgAtomic]:
                op = find_control_node(formula[i].operator_symbol)
                root.add_children(
                    [rparser(formula[i].formulas, op, planner, node, nid)])
            else:
                # Creat BT execution node
                root.add_children(
                    [node(get_name(formula[i]), planner, id=nid)])
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
        root = node(get_name(ltlformula), planner, id=nid)
        nid += 1
    else:
        rootnode = find_control_node(ltlformula.operator_symbol)
        root = rparser(ltlformula.formulas, rootnode, planner, node, nid)

    return root


def create_until_node(a, b):
    root = Sequence('Seq')
    selec = Selector('Se')
    p2 = ConditionNode(str(b.id), b)
    p1 = ConditionNode(str(a.id), a)
    goal1 = a
    goal2 = b
    selec.add_children([p2, goal1])
    seq = Sequence('S')
    seq.add_children([p1, goal2])
    root.add_children([seq, selec])
    return root


# Recursively modify the parsed BT to add the Untiil sub-tree
def recursive_until(node):
    if node.children:
        # If the root node already has until uperator
        if (isinstance(node, Sequence) and node.name == 'U'):
            # Only two node is required at a time to create until sub-tree
            fchild = node.children[0]
            schild = node.children[1]
            clist_remove = []
            # If theer are more than two nodes
            for i in range(len(node.children)-1):
                subtree = create_until_node(
                    copy.copy(fchild), copy.copy(schild))
                clist_remove += [fchild, schild]
                node.add_children([subtree])
                fchild = subtree
                schild = node.children[i+2]
                if (
                    isinstance(
                        node.children[i], Sequence) and node.children[
                            i].name == 'U'):
                    recursive_until(node.children[i])
            # Since the BT nodes are already in the Until sub-tree, they
            # are deleted from the main tree
            [node.remove_child(child) for child in clist_remove]
        # If the until operator is embedded into the children nodes
        else:
            for node in node.children:
                recursive_until(node)


def post_tick_until(root):
    condition_nodes = [
        node for node in root.iterate() if isinstance(node, ConditionNode)]
    for node in condition_nodes:
        node.value = node.obj.value


def find_control_node(operator):
    # print(operator, type(operator))
    if operator in ['U']:
        # sequence
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
