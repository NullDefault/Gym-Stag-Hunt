# gym-stag-hunt

The repository contains an implementation of a Markov Stag Hunt - a multi agent gridworld game as described in [this paper](https://arxiv.org/abs/1709.02865).

## Markov Stag Hunt

Two agents start off in the top two corners of a NxN grid. A stag is placed in the center of the grid. K plants are
then placed in K random unoccupied cells. In each time step, each agent moves up, down, left or right, and the stag 
moves towards the nearest agent. At the end of the time step, if an agent is on top of a plant, it harvests it getting
+h reinforcement. If an agent is on top of a stag, and the other agent is not, the victim agent loses -g points.
However, if both agents are on top of the stag, they both get +p reinforcement points. After a specified amount of
time steps (default is 1000), the game is considered done and the environment will reset the position of its entities.

<p align="center">
  <img src="https://raw.githubusercontent.com/NullDefault/gym-stag-hunt/master/gym_stag_hunt/assets/screenshot.png" />
</p>


# Installation

```bash
cd gym-stag-hunt
pip install -e .
```