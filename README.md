## Requirements
gym >= 0.2.3<br>
tqdm >= 4.32.1<br>
torch >= 1.7.1<br>
numpy >= 1.14.1<br>
python >=  3.6.8<br>
IPython >= 6.4.0<br>
imageio >= 2.3.0<br>
stable_baselines3 >= 0.10.0<br>

## Installation
```bash
git clone https://github.com/AustinXY/adversarial-level-generation.git
cd gym-sokoban
pip install -e .
```

## Directory structure
```
training_scripts  contains scripts for trainig Sokoban agent and ALG.
demo_checkpoints  contains checkpoints for running the demo.
demo              contains the demo jupyter notebook and demo utilities
  |_ temp         contains intermediate gifs to display in notebook
envs              contains gym environments, demo wrappers, and relative utilities
  |_ surface      contains tile source images for human readable level display
```