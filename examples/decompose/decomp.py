"""Verifying the decomposition of LTL into BT."""

# from examples.decompose.utils import DummyNode

from flloat.parser.ltlf import LTLfParser

# parse the formula
parser = LTLfParser()
# formula = "r U (t U (b U p))"  # F(b) U F(p)"
# formula = "p U (b U (t U r))"
formula = "((F r U t) U F b) U F p"
parsed_formula = parser(formula)
print(parsed_formula)
# evaluate over finite traces
t1 = [
    {"r": True, "t": False, "b": False, "p": False},
    {"r": True, "t": False, "b": False, "p": False},
    {"r": False, "t": True, "b": False, "p": False},
    {"r": False, "t": False, "b": True, "p": False},
    {"r": False, "t": False, "b": False, "p": True},
]
# t1 = [
#     {"r": False, "t": True, "b": False, "p": False},
#     {"r": True, "t": False, "b": False, "p": False},
#     {"r": False, "t": False, "b": False, "p": False},
#     {"r": False, "t": False, "b": True, "p": False},
#     {"r": False, "t": False, "b": False, "p": True}
#     ]

print('t1', parsed_formula.truth(t1, 0))

# t2 = [
#     {"a": False, "b": False},
#     {"a": True, "b": True},
#     {"a": False, "b": True},
# ]
# assert not parsed_formula.truth(t2, 0)

# # from LTLf formula to DFA
dfa = parsed_formula.to_automaton()
# print(dir(dfa))
print(dfa.get_transitions(), dfa.size)
# assert dfa.accepts(t1)
# assert not dfa.accepts(t2)

# # print the automaton
graph = dfa.to_graphviz()
graph.render("./1")  # requires Graphviz installed on your system.
