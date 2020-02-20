# -*- coding: utf-8 -*-
"""
LTL based Goal Framework

"""
import datetime

from .pygoal.lib.genrecprop import GenRecProp
from .pygoal.lib.bt import Policy, PolicyNode, DummyNode
from .pygoal.lib.mdplib import GridMDP
from .pygoal.utils.bt import goalspec2BT


__all__ = [
    "GenRecProp", "Policy", "PolicyNode", "DummyNode",
    "GridMDP", "goalspec2BT"]

__title__ = 'PyGoal'
__version__ = '0.0.1'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright %s Project PyGoal Team' % datetime.date.today().year
