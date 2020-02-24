# PyGoal

A framework that allows to represent goal explicitly for learning algorithms using LTL formulation. This framework is based on python implementation of FLLOAT LTL farmework. Brief feature list:
* Easy way to represent goal for learning algorithms
* Various examples for different type of goals
* Learning algorithm to achieve goals specified by the LTL formula
* Behavior Tree implementation for learning of sequential goals

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
pip install matplotlib numpy scipy scikit-learn pandas gym-minigrid gym nose 
```

### Installing

This swarm framework depends on FLLOAT for Linear Temporal Logic (LTL) and py_trees for Behavior trees. You need to clone these framework from below location

PonyGE2

```
git clone https://github.com/aadeshnpn/flloat.git
cd flloat
git checkout predicate
pip install .
```

And for py_trees

```
git clone https://github.com/aadeshnpn/py_trees
cd py_trees
pip install .
```

Now all the dependencies are installed

clone the swarm repo
```
git clone https://github.com/aadeshnpn/PyGoal.git
cd PyGoal
pip install .
```
Now you have the PyGoal framework installed.

## Running the tests

All the tests files are located under test folder. Just traverse to the folder and run
```
nosetest test_mdp_goals.py
```
to test is the mdp related modules are working or not. But there are lots of
test files to test other modules

## Contributing

Just submit the pull requests to us. The code should be properly formatted and validated using all PEP8 rules and flask rules excluding ignore=F403,E501,E123,E128.  If you are using visual studio code you can add following to your setting files
```
"python.linting.flake8Enabled": true,
"python.linting.flake8Args": ["--ignore=F403,E501,E123,E128","--exclude=docs,build"]
```

## Authors

* **Aadesh Neupane** - *Initial work* - [Aadeshnpn](https://github.com/aadeshnpn)

See also the list of [contributors](https://github.com/aadeshnpn/swarm/contributors) who participated in this project.

## License
Mozilla Public License 2.0
