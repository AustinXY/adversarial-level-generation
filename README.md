## Requirements
python >=  3.6.8<br>

gym >= 0.2.3<br>
torch >= 1.7.1<br>
tqdm >= 4.32.1<br>
numpy >= 1.14.1<br>
IPython >= 6.4.0<br>
imageio >= 2.3.0<br>
stable_baselines3 >= 0.10.0<br>
(dependencies will be automatically installed using the installation instruction)

## Installation
```bash
git clone https://github.com/AustinXY/adversarial-level-generation.git
cd gym-sokoban
pip install -e .
```

## Directory structure
```
demo                  contains the demo jupyter notebook and demo utilities
  |_ temp             contains intermediate gifs to display in notebook
envs                  contains gym environments, demo wrappers, and relative utilities
  |_ surface          contains tile source images for human readable level display
training_scripts      contains scripts for trainig Sokoban agent and ALG.
demo_checkpoints      contains checkpoints for running the demo.
  |_ agent_v1b_0.zip  Sokoban agent checkpoint, trained for ~1e7 steps
  |_ agent_v1b_1.zip  Sokoban agent checkpoint, trained for ~1e7 steps
  |_ alg_v1b_v0.zip   ALG checkpoint, trained for ~1e7 steps
  |_ alg_v1b_v1.zip   ALG checkpoint, trained for ~1e7 steps
  |_ alg_v1b_v2.zip   ALG checkpoint, trained for ~1e7 steps
```

## Design
The project will be based on the OpenAI gym. The OpenAI gym is a reinforcement learning platform that has some environments and one of them is the gym-sokoban. The environment has a builtin random level generation algorithm based on random walk algorithm which this project will seek to replace with the adversarial level generator (ALG). <br>
The reinforcement learning framework is usually defined as such. An agent has a set of states and a set of available actions at each state. By assigning some reward functions to the states, the agent can learn to select appropriate actions at each state to maximize the reward. We can see that it’s easy to define a set of states and actions for the sokoban. However, it’s also possible to define a set of states and actions for the ALG so that it can be reinforcement trained. The Definition is as follows. Each map configuration is a state for the ALG. At each state, the ALG can substitute a wall tile with a floor, box target, box or player tile. There is a maximum of n box targets, n boxes and 1 player tiles. After the map contains n box targets, n boxes and 1 player tile, the ALG will have a finish generation action. <br>
This is designed to be an adversarial approach meaning that the game agent will be trained against the ALG adversarially. The adversarial framework is inspired by the generative adversarial network (GAN), in which a generator is trained against a discriminator. The generator and the discriminator will both improve gradually to ensure that neither the generator nor the discriminator will become too powerful at any given point to break the training. In the adversarial training approach proposed in this project, it’s easy to see that the game AI and the ALG will also need to both improve gradually and with respect to each other. If the ALG is too powerful and generate a map that’s extremely hard at the very beginning, the game agent wouldn’t be able to solve the game, and if the game agent is too powerful and crakes all the ALG generated maps with zero effort, the ALG wouldn’t be able to train as well.<br>

```
For n = 1 : 4
For epoch = 1 : maxEpoch
Map_is_solvable = False
While not Map_is_solvable
Initial_map = all wall tiles
ALG generate n-box map starting from the initial map
Using reverse play to check solvability of generated_map
If generated_map is not solvable
ALG_reward = some small penalty
Update ALG
If generated_map is solvable
Map_is_solvable = True
End While
Train game agent on generated_map
Update game agent
Calculate reward for ALG base on game agent’s training process
Update ALG
End for
End for
```

## Acknowledgement
The codebase is built on top of the gym-sokoban project.
