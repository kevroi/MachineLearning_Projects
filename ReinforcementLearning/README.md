# Reinforcement Learning in Python

All examples in this repo will be based on the OpenAI `gym` python library, which can be installed using the pip package manager on your CLI:
```
pip install gym
```

`gym` has a bunch of different environments for RL experiments, as listed on the OpenAI [docs page](https://www.gymlibrary.dev).
For example, if I wanted to use `Box2D` for experiments with `Bipedal Walker` or `Lunar Lander`, I can install it as follows:
```
pip install `gym[Box2D]`
```