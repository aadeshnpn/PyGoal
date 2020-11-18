"""Compute Competency in a GridWorld"""
# import time
import numpy as np
import gym
import gym_minigrid     # noqa: F401
from gym_minigrid.minigrid import (     # noqa: F401
    Grid, OBJECT_TO_IDX, Key, Door, Goal, Ball, Box, Lava,
    COLOR_TO_IDX)

# import py_trees
from py_trees.trees import BehaviourTree
from py_trees import Blackboard
from py_trees.composites import Sequence, Selector
from pygoal.utils.bt import goalspec2BT
from gym_minigrid.wrappers import ReseedWrapper, FullyObsWrapper

from pygoal.lib.bt import CompetentNode
from pygoal.utils.distribution import (
    compare_curve,
    sequence, selector,
    recursive_com, recursive_setup)

from examples.competent.multigoalgrid import (
    GenRecPropMultiGoal, MultiGoalGridExp)


def reset_env(env, seed=12345):
    # env.seed(12345)
    env.reset()


def find_key():
    env_name = 'MiniGrid-Goals-v0'
    env = gym.make(env_name)
    # env = ReseedWrapper(env, seeds=[3])   # Easy
    env = ReseedWrapper(env, seeds=[5])     # Medium
    # env = ReseedWrapper(env, seeds=[7])     # Hard
    env = FullyObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.agent_view_size = 1
    env.reset()
    # env.render(mode='human')
    # time.sleep(10)
    # state, reward, done, _ = env.step(2)
    # print(state['image'].shape, reward, done, _)
    # Find the key
    goalspec = 'F P_[KE][1,none,==]'
    # keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    allkeys = [
        'LO', 'FW', 'KE', 'DR',
        'BOB', 'BOR', 'BAB', 'BAR',
        'LV', 'GO', 'CK',
        'CBB', 'CBR', 'CAB', 'CAR',
        'DO', 'RM']

    keys = [
        'LO', 'FW', 'KE']

    actions = [0, 1, 2]

    root = goalspec2BT(goalspec, planner=None, node=CompetentNode)
    behaviour_tree = BehaviourTree(root)
    child = behaviour_tree.root

    planner = GenRecPropMultiGoal(
        env, keys, child.name, dict(), actions=actions,
        max_trace=50, seed=None, allkeys=allkeys)

    def run(pepoch=50, iepoch=10):
        # pepoch = 50
        child.setup(0, planner, True, pepoch)
        # Train
        for i in range(pepoch):
            behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )
        # Inference
        child.train = False
        child.planner.epoch = iepoch
        child.planner.tcount = 0
        for i in range(iepoch):
            behaviour_tree.tick(
                pre_tick_handler=reset_env(env)
            )
    competency = []
    epochs = [(80, 10)] * 2
    datas = []
    for i in range(2):
        run(epochs[i][0], epochs[i][1])
        datas.append(
            np.mean(
                planner.blackboard.shared_content[
                    'ctdata'][planner.goalspec], axis=0))
        competency.append(planner.compute_competency())
    print(competency)
    compare_curve(competency, datas)
    # print(
    #     planner.compute_competency(),
    #     planner.blackboard.shared_content['curve'])


def carry_key():
    env_name = 'MiniGrid-Goals-v0'
    env = gym.make(env_name)
    env = ReseedWrapper(env, seeds=[3])
    env = FullyObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.agent_view_size = 1
    env.reset()
    # env.render(mode='human')
    state, reward, done, _ = env.step(2)

    # Find the key
    goalspec = 'F P_[KE][1,none,==] U F P_[CK][1,none,==]'
    allkeys = [
        'LO', 'FW', 'KE', 'DR',
        'BOB', 'BOR', 'BAB', 'BAR',
        'LV', 'GO', 'CK',
        'CBB', 'CBR', 'CAB', 'CAR',
        'DO', 'RM']

    keys = [
        'LO', 'FW', 'KE', 'CK']

    actions = [0, 1, 2, 3, 4, 5]

    root = goalspec2BT(goalspec, planner=None, node=CompetentNode)
    behaviour_tree = BehaviourTree(root)
    epoch = 80

    def fn_c(child):
        pass

    def fn_eset(child):
        planner = GenRecPropMultiGoal(
            env, keys, child.name, dict(), actions=actions,
            max_trace=40, seed=None, allkeys=allkeys)

        child.setup(0, planner, True, epoch)

    def fn_einf(child):
        child.train = False
        child.planner.epoch = 5
        child.planner.tcount = 0

    def fn_ecomp(child):
        child.planner.compute_competency()

    recursive_setup(behaviour_tree.root, fn_eset, fn_c)
    # recursive_setup(behaviour_tree.root, fn_c, fn_c)
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    # py_trees.display.print_ascii_tree(behaviour_tree.root)

    # Train
    for i in range(100):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Training', behaviour_tree.root.status)

    # Inference
    recursive_setup(behaviour_tree.root, fn_einf, fn_c)
    for i in range(5):
        behaviour_tree.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Inference', behaviour_tree.root.status)
    recursive_setup(behaviour_tree.root, fn_ecomp, fn_c)
    # recursive_setup(behaviour_tree.root, fn_c, fn_c)
    blackboard = Blackboard()
    print(recursive_com(behaviour_tree.root, blackboard))


def test_comptency():
    # import py_trees
    # behaviour_tree = BehaviourTree(root)
    # # Remember to comment set_state in GenRecProp before
    # # running this test case
    one = BehaviourTree(Sequence(name=str(1)))
    two = Sequence(name=str(2))
    three = Sequence(name=str(3))
    four = Selector(name=str(4))
    five = Sequence(name=str(5))
    six = Sequence(name=str(6))
    # seven = Parallel(name=str(7))
    seven = Selector(name=str(7))
    exenodes = [CompetentNode(
        name=chr(ord('A')+i), planner=None) for i in range(0, 11)]
    three.add_children(exenodes[:3])
    four.add_children(exenodes[3:6])
    six.add_children(exenodes[6:9])
    seven.add_children(exenodes[9:])
    two.add_children([three, four])
    five.add_children([six, seven])
    one.root.add_children([two, five])
    # py_trees.logging.level = py_trees.logging.Level.DEBUG
    # py_trees.display.print_ascii_tree(one.root)
    blackboard = Blackboard()
    env_name = 'MiniGrid-Goals-v0'
    env = gym.make(env_name)
    env = ReseedWrapper(env, seeds=[3])
    env = FullyObsWrapper(env)
    env.max_steps = min(env.max_steps, 200)
    env.agent_view_size = 1
    env.reset()
    # env.render(mode='human')
    state, reward, done, _ = env.step(2)
    # print(state['image'].shape, reward, done, _)
    # Find the key
    goalspec = 'F P_[KE][1,none,==]'
    # keys = ['L', 'F', 'K', 'D', 'C', 'G', 'O']
    allkeys = [
        'LO', 'FW', 'KE', 'DR',
        'BOB', 'BOR', 'BAB', 'BAR',
        'LV', 'GO', 'CK',
        'CBB', 'CBR', 'CAB', 'CAR',
        'DO', 'RM']

    keys = [
        'LO', 'FW', 'KE']

    actions = [0, 1, 2, 3, 4, 5]

    def fn_c(child):
        pass

    def fn_eset(child):
        planner = GenRecPropMultiGoal(
            env, keys, goalspec, dict(), actions=actions,
            max_trace=40, seed=None, allkeys=allkeys, id=child.name)

        child.setup(0, planner, True, 50)

    def fn_einf(child):
        child.train = False
        child.planner.epoch = 5
        child.planner.tcount = 0

    def fn_ecomp(child):
        child.planner.compute_competency()
        print(
            child.name,
            child.planner.blackboard.shared_content['curve'][child.name])

    recursive_setup(one.root, fn_eset, fn_c)
    # Train
    for i in range(100):
        one.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Training', one.root.status)

    # Inference
    recursive_setup(one.root, fn_einf, fn_c)
    for i in range(5):
        one.tick(
            pre_tick_handler=reset_env(env)
        )
    print(i, 'Inference', one.root.status)
    recursive_setup(one.root, fn_ecomp, fn_c)

    # Manually setting the competency
    ckeys = [chr(ord('A')+i) for i in range(0, 11)]
    manval = [
        np.array([0.84805786, 4.76735384, 0.20430223]),
        np.array([0.54378425, 4.26958399, 3.50727315]),
        np.array([0.50952059, 5.54225945, 5.28025611])
        ]
    j = 0
    for c in ckeys:
        blackboard.shared_content['curve'][c] = manval[j % 3]
        j += 1
    # Recursively compute competency for control nodes
    recursive_com(one.root, blackboard)
    # print(exenodes[0].planner.blackboard.shared_content['curve'])

    # Manually compare the recursively computed competency values
    # for the control
    # First sub-tree
    a = exenodes[0].planner.blackboard.shared_content['curve']['A']
    b = exenodes[0].planner.blackboard.shared_content['curve']['B']
    c = exenodes[0].planner.blackboard.shared_content['curve']['C']
    threec = sequence([a, b, c])
    # print('three', threec)
    # print(
    # 'three', exenodes[0].planner.blackboard.shared_content['curve']['3'])
    assert threec == exenodes[
        0].planner.blackboard.shared_content['curve']['3']
    # Second sub-tree
    d = exenodes[0].planner.blackboard.shared_content['curve']['D']
    e = exenodes[0].planner.blackboard.shared_content['curve']['E']
    f = exenodes[0].planner.blackboard.shared_content['curve']['F']
    fourc = selector([d, e, f])
    # print(
    # 'four', exenodes[0].planner.blackboard.shared_content['curve']['4'])
    assert fourc == exenodes[
        0].planner.blackboard.shared_content['curve']['4']
    # Third sub-tree
    g = exenodes[0].planner.blackboard.shared_content['curve']['G']
    h = exenodes[0].planner.blackboard.shared_content['curve']['H']
    i = exenodes[0].planner.blackboard.shared_content['curve']['I']
    sixc = sequence([g, h, i])
    # print(
    # 'six', exenodes[0].planner.blackboard.shared_content['curve']['6'])
    assert sixc == exenodes[
        0].planner.blackboard.shared_content['curve']['6']
    # Fourth sub-tree
    j = exenodes[0].planner.blackboard.shared_content['curve']['J']
    k = exenodes[0].planner.blackboard.shared_content['curve']['K']
    sevenc = selector([j, k])
    # print(
    # 'seven', exenodes[0].planner.blackboard.shared_content['curve']['7'])
    assert sevenc == exenodes[
        0].planner.blackboard.shared_content['curve']['7']

    twoc = sequence([threec, fourc])
    assert twoc == exenodes[
        0].planner.blackboard.shared_content['curve']['2']

    fivec = sequence([sixc, sevenc])
    assert fivec == exenodes[
        0].planner.blackboard.shared_content['curve']['5']

    onec = sequence([twoc, fivec])
    assert onec == exenodes[
        0].planner.blackboard.shared_content['curve']['1']

    print(onec)


def expriment1():
    goalspec = 'F P_[KE][1,none,==] U F P_[CK][1,none,==]'
    keys = [
        'LO', 'FW', 'KE', 'CK']
    exp = MultiGoalGridExp('carrykey', goalspec, keys)
    exp.run()
    exp.draw_plot(['F(P_[KE][1,none,==])', 'F(P_[CK][1,none,==])'])


if __name__ == "__main__":
    # find_key()
    # carry_key()
    # test_comptency()
    expriment1()
