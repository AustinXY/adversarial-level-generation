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
