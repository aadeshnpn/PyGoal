
from pygoal.utils.bt import goalspec2BT


def test_goal_0():
    goalspec0 = 'F(P_[IC][True,none,==]) U F(P_[L][13,none,==])'
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)
    assert rootname == 'Sequence'
    assert name == 'U'
    assert num_child == 2


def test_goal_1():
    goalspec0 = 'P_[IC][True,none,==]'
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)
    assert rootname == 'GoalNode'
    assert name == 'P_[IC][True,none,==]'
    assert num_child == 0


def test_goal_2():
    goalspec0 = 'F P_[IC][True,none,==]'
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'GoalNode'
    assert name == 'F(P_[IC][True,none,==])'
    assert num_child == 0


def test_goal_3():
    goalspec0 = 'G P_[IC][True,none,==]'
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'GoalNode'
    assert name == 'G(P_[IC][True,none,==])'
    assert num_child == 0


def test_goal_4():
    goalspec0 = """((F(P_[IC][True,none,==]) U
        G(P_[L][13,none,==]))) U (F(P_[L][23,none,==]))"""
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Sequence'
    assert name == 'U'
    assert num_child == 3


def test_goal_5():
    goalspec0 = """((F(P_[IC][True,none,==]) U
        G(P_[L][13,none,==]))) R (F(P_[L][23,none,==]))"""
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Selector'
    assert name == 'R'
    assert num_child == 2


def test_goal_6():
    goalspec0 = """((F(P_[IC][True,none,==]) U F(P_[L][13,none,==]))
        U F(P_[L][23,none,==])) U F(P_[L][33,none,==])"""
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Sequence'
    assert name == 'U'
    assert num_child == 4


def test_goal_7():
    goalspec0 = '(F P_[IC][True,none,==]) & (G P_[L][13,none,==])'
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Parallel'
    assert name == '&'
    assert num_child == 2


def test_goal_8():
    goalspec0 = """(F P_[IC][True,none,==]) &
        (G P_[L][13,none,==]) & (G P_[L][13,none,==])"""
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Parallel'
    assert name == '&'
    assert num_child == 3


def test_goal_9():
    goalspec0 = '(F P_[IC][True,none,==]) | (G P_[L][13,none,==])'
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Selector'
    assert name == '|'
    assert num_child == 2


def test_goal_10():
    goalspec0 = """(F P_[IC][True,none,==]) | (
        G P_[L][13,none,==]) | (G P_[L][13,none,==])"""
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Selector'
    assert name == '|'
    assert num_child == 3


def test_goal_11():
    goalspec0 = """((F P_[IC][True,none,==]) | (
        G P_[L][13,none,==])) & (G P_[L][13,none,==])"""
    root = goalspec2BT(goalspec0)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)

    assert rootname == 'Parallel'
    assert name == '&'
    assert num_child == 2


def test_goal_paper():
    atomic = 'F(P_[IC][True,none,==])'
    goalspec = '('+atomic+' & '+atomic+') U ('+atomic+' U '+atomic+')'
    print(goalspec)
    root = goalspec2BT(goalspec)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)
    print(root.children)
    assert rootname == 'Sequence'
    assert name == 'U'
    assert num_child == 3


def test_goal_standard():
    psi1 = 'F P_[A][True,none,==]'
    psi2 = 'F P_[B][True,none,==]'
    psi3 = 'F P_[C][True,none,==]'
    psi4 = 'F P_[D][True,none,==]'
    psi5 = 'F P_[E][True,none,==]'
    psi6 = 'F P_[F][True,none,==]'
    psi7 = 'F P_[G][True,none,==]'
    goalspec = '('+psi1+' & '+psi2+') U ('+psi3+' U '+psi4+') U ('+psi5+' U '+psi6+' U '+psi7+')'   # noqa: E501

    print(goalspec)
    root = goalspec2BT(goalspec)
    rootname = type(root).__name__
    name, num_child = root.name, len(root.children)
    print(root.children, [node.name for node in root.children])
    assert rootname == 'Sequence'
    assert name == 'U'
    assert num_child == 6
