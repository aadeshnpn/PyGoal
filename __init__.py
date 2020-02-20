# -*- coding: utf-8 -*-
"""
LTL based Goal Framework

"""
import datetime

from .pygoal.genrecprop import GenRecProp
from .pygoal.bt import Policy, PolicyNode, DummyNode
from .pygoal.mdplib import GridMDP

__all__ = [
    "GenRecProp", "Policy", "PolicyNode", "DummyNode",
    "GridMDP"]

__title__ = 'PyGoal'
__version__ = '0.0.1'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright %s Project PyGoal Team' % datetime.date.today().year
