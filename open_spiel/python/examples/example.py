# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example."""

import random
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

FLAGS = flags.FLAGS

# Game strings can just contain the name or the name followed by parameters
# and arguments, e.g. "breakthrough(rows=6,columns=6)"
# flags.DEFINE_string("game_string", "spike_sabacc", "Game string")
flags.DEFINE_string("game_string", "spike_sabacc", "Game string")


def main(_):
  games_list = pyspiel.registered_games()
  # print("Registered games:")
  # print(games_list)

  action_string = None

  print("Creating game: " + FLAGS.game_string)
  game = pyspiel.load_game(FLAGS.game_string)

  # Create the initial state
  state = game.new_initial_state()

  # Print the initial state
  print(str(state))

  i = 0
  while not state.is_terminal() and i < 50:
    i += 1
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      print("\033[93mchance node\033[0m")
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      print("Chance node, got " + str(num_actions) + " outcomes: ")
      print(outcomes)
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
      
      print("\033[95mCurrent state: ", state.state_to_string(state.current_state), "\033[0m")
      print("\033[95mCurrent state: ", state.current_state, "\033[0m")
      print("\033[95mPlayer ", state.current_player(), ", randomly sampled action: ", state.action_to_string(state.current_player(), action), "\033[0m")
      print("\033[95mbefore applying action:\033[0m")
      print(str(state))

      state.apply_action(action)
      print("\033[95mafter applying action:\033[0m")
      print(str(state))

    elif state.is_simultaneous_node():
      # Simultaneous node: sample actions for all players.
      random_choice = lambda a: np.random.choice(a) if a else [0]
      chosen_actions = [
          random_choice(state.legal_actions(pid))
          for pid in range(game.num_players())
      ]
      print("Chosen actions: ", [
          state.action_to_string(pid, action)
          for pid, action in enumerate(chosen_actions)
      ])
      state.apply_actions(chosen_actions)
    else:
      print("\033[93mdecision node\033[0m")
      # Decision node: sample action for the single current player
      action = random.choice(state.legal_actions())
      action_string = state.action_to_string(state.current_player(), action)

      print("\033[91mCurrent state: ", state.state_to_string(state.current_state), "\033[0m")
      print("\033[91mCurrent state: ", state.current_state, "\033[0m")
      print("\033[91mPlayer ", state.current_player(), ", randomly sampled action: ", action_string, "\033[0m")
      print("\033[92mbefore applying action:\033[0m")
      print(str(state))
      state.apply_action(action)
      print("\033[94mafter applying action:\033[0m")
      print(str(state))
    


  # print("get bets: ", state.get_bets())

  # hand_sums, winning_score, winners, bets, winnings = state.returnReturns()
  # print("Hand sums: ", hand_sums)
  # print("Winning score: ", winning_score)
  # print("Winners: ", winners)
  # print("Bets: ", bets)
  # print("Winnings: ", winnings)
  # print("Returns: ", [winnings if i in winners else -bets[i] for i in range(2)])
  # Game is now done. Print utilities for each player
  returns = state.returns()
  print("returns: ", returns)
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)
