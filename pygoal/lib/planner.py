"""Planners."""

from enum import Enum
from pygoal.lib.genrecprop import GenRecProp, GenRecPropMDP


class Planner(Enum):
    DEFAULT = GenRecProp
    GENRECPROPMDP = GenRecPropMDP
    GENRECPROPTAXI = 'TAXI'
    GENRECPROPKEY = 'KEYDOOR'
    MOVEIT = 'MOVEIT'
