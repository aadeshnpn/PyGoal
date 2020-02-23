"""Planners."""

from enum import Enum
from pygoal.lib.genrecprop import GenRecProp


class Planner(Enum):
    DEFAULT = GenRecProp
    GENRECPROPMDP = GenRecProp
    GENRECPROPTAXI = 'TAXI'
    GENRECPROPKEY = 'KEYDOOR'
    MOVEIT = 'MOVEIT'
