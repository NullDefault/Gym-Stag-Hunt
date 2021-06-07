# Gym Stag Hunt

The repository contains an implementation of a Markov Stag Hunt - a multi agent grid-based game as described in [this paper](https://arxiv.org/abs/1709.02865). The core goal of the project is to offer a robust, efficient, and customizable environment for exploring prosocial behavior in multi agent reinforcement learning. Feedback and requests for features are welcome.

## Markov Stag Hunt

Two agents start off in the top two corners of a NxM grid. A stag is placed in the center of the grid. K plants are
then placed in K random unoccupied cells. In each time step, each agent moves up, down, left or right, and the stag 
either moves towards the nearest agent (default) or takes a random move. At the end of the time step, if an agent is on top of a plant, it **harvests** it getting
h reinforcement. If an agent is on top of a stag, and the other agent is not, the victim agent gets **mauled**, losing g points.
However, if both agents are on top of the stag, they **catch** it, both earning p reinforcement points. After a specified amount of
time steps (default is 1000), the game is considered done and the environment will reset the position of its entities.

**Observations**: RGB pixel array or coordinate array with boolean tuples signifying entity presence.  
**Actions**: Left, down, right or up on the grid. Encoding is ```LEFT=0, DOWN=1, RIGHT=2, UP=3```.

#### PyGame Rendering / Image Observation
<p align="center">
  <img src="https://github.com/NullDefault/gym-stag-hunt/blob/master/gym_stag_hunt/assets/screenshot.png?raw=true" />
</p>

#### Matrix Printout / Coordinate Observation
```
╔════════════════════════════╗
║ ·A   ·   P·    ·    · B  · ║
║ ·    ·    ·    ·    ·    · ║
║ ·    ·    ·  S ·    ·    · ║
║ ·    ·    ·    ·   P·    · ║
║ ·    ·    ·    ·    ·    · ║
╚════════════════════════════╝
```

### Config Parameters
**param** = format = default value:
> Description

**grid_size** = (N, M) = (5, 5): 
> Dimensions of the simulation grid. M x N should be equal to at least 3. 

**screen_size** = (X, Y) = (600, 600):
> Dimensions of the virtual display where the game will render. Irrelevant if pygame will not be used.
  
**obs_type** = 'image' or 'coords' = 'image': 
> What type of observation you want. Image gets you a coordinate array with RGB tuples corresponding to pixel color. Coords gets you a coordinate array with boolean tuples of size 4 signifying the presence of entities in that cell (index 0 is agent A, index 1 is agent B, index 2 is stag, index 3 is plant).  

**load_renderer** = bool = False: 
> Used if you want to render some iterations when using coordinate observations. Irrelevant when using image observations.  

**episodes_per_game** = int = 1000: 
> How many episodes (time steps) occur during a single game of Stag Hunt before entity positions are reset and the game is considered done.

**stag_reward** = int = 5:
> Reinforcement reward for when agents catch the stag by occupying the same cell as it at the same time. Expected to be positive.

**stag_follows** = bool = True:
> Whether or not the stag should seek out the nearest agent at each time step. If false, the stag will take a random action instead.

**run_away_after_maul** = bool = False:
> Whether or not the stag re-appears in a different cell after mauling an agent. When false, this can lead to difficulties with learning as agents will often get overwhelmed by the (very aggressive apparently) stag, who will relentlessly chase them continuously mauling them. An intelligent agent may never get caught, as by taking a step away from the stag at each time step they can avoid it entirely. Learning this initially, however, can be difficult.

**forage_quantity** = int = 2:
> How many plants should be at the grid at any given time step. Cannot be higher than N x M - 3. N x M is the total number of cells, and -3 is for the 3 cells occupied by the agents and stag.

**forage_reward** = int = 1:
> Reinforcement reward for harvesting a plant. Expected to be positive.

**mauling_punishment** = int = -5:
> Reinforcement reward (or, rather, punishment) for getting mauled by a stag. Expected to be negative.

## Classic Stag Hunt

A 2x2 Stag Hunt game as usually described in game theory literature.

**Observations**: No observations, although last taken agent actions are returned in place of the observation.  
**Actions**: Cooperate or Defect, encoding is ```COOPERATE=0, DEFECT=1```.

### Config Parameters
**param** = format = default value:
> Description

**stag_reward** = int = 5:
> Reinforcement reward for when agents catch the stag by occupying the same cell as it at the same time. Expected to be positive.

**forage_reward** = int = 1:
> Reinforcement reward for harvesting a plant. Expected to be positive.

**mauling_punishment** = int = -5:
> Reinforcement reward (or, rather, punishment) for getting mauled by a stag. Expected to be negative.

### Example Render

```
      B   
    C   D 
   ╔══╦══╗
 C ║AB║  ║
   ║  ║  ║
A  ╠══╬══╣
   ║  ║  ║
 D ║  ║  ║
   ╚══╩══╝
```
A and B are cooperating here.

# Installation

After cloning the repository:

```bash
cd Gym-Stag-Hunt
pip install -e .
```

# Minimal Example
```
import gym
import gym_stag_hunt
import time

env = gym.make("StagHunt-v0", obs_type='image', ...) # you can pass config parameters here
env.reset()
for iteration in range(1000):
  time.sleep(.2)
  obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
  env.render()
env.close()
```

