import pickle
import os
from open_spiel.python import rl_environment
# from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import cfr
import open_spiel.python.games.spike_sabacc
import numpy as np
import pyspiel
from open_spiel.python.algorithms import get_all_states

# Create the environment
env = rl_environment.Environment("spike_sabacc")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

game = pyspiel.load_game("spike_sabacc")  # Load game properly

states = get_all_states.get_all_states(game, depth_limit=5)
print(states)