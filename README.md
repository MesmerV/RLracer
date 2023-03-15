## Introduction
This repository is the repo for the project of INF581 of the Ecole Polytechnique. It contains all the required scripts to start a training for autonomous agent in our custom environment.

## Installation
Frist of all, you must create a new conda environmnent. 

```
conda create -n RLgym
conda activate RLgym
```


Then install dependencies
```
pip install numpy matplotlib pygame moviepy git+https://github.com/carlosluis/stable-baselines3@fix_tests
```

Then you can clone this repository:
```
git clone git@github.com:MesmerV/RLracer.git 
``

Then you clone the fork of the repo of the environment:
```
git clone https://github.com/maumlima/highway-env.git
```

Rename the repo highway-env into "highway_env"

And you finally install it:

```
pip install -e ./highway_env
```