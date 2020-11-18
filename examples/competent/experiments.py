"""Competency experiments with MultiGoalGrid for the IJCAIpaper"""

from examples.competent.multigoalgrid import (
    MultiGoalGridExp)


def exp_find_key():
    goalspec = 'F P_[KE][1,none,==]'
    keys = [
        'LO', 'FW', 'KE']
    exp = MultiGoalGridExp('exp_find_key', goalspec, keys)
    exp.run()
    exp.draw_plot(['F(P_[KE][1,none,==])'])
    exp.save_data()


def exp_carry_key():
    goalspec = 'F P_[KE][1,none,==] U F P_[CK][1,none,==]'
    keys = [
        'LO', 'FW', 'KE', 'CK']
    exp = MultiGoalGridExp('exp_carry_key', goalspec, keys)
    exp.run()
    root = exp.blackboard.shared_content['curve']['U']
    exp.draw_plot(['F(P_[KE][1,none,==])', 'F(P_[CK][1,none,==])'], root)
    exp.save_data()


def main():
    exp_find_key()
    exp_carry_key()


if __name__ == "__main__":
    main()
