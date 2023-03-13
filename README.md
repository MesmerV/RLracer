## Introduction
This repository is the repo for the project of INF581 of the Ecole Polytechnique. It contains all the required scripts to start a training for autonomous agent in our custom environment.

## Installation
Frist of all, you must create a new conda environmnent. Secondly, install pytorch, numpy, matplotlib.

Then install pygame:
```
pip install pygame
```
Then:
```
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
```

```
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
```
```
pip install moviepy
```

Then you clone the fork of the repo of the environment:
```
https://github.com/maumlima/highway-env.git
```

Rename the repo highway-env into "highway_env"

And you finally install it:

```
pip install -e ./highway_env
```