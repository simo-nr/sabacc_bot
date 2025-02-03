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

# Lint as python3
"""Corellian Spike Sabacc implemented in Python."""

import numpy as np
import random

import pyspiel


STAND_ACTION = 0
DRAW_ACTION = 1
BET_ACTION = 2
FOLD_ACTION = 3

_MAX_NUM_PLAYERS = 10
_MIN_NUM_PLAYERS = 2
_INITIAL_HAND_SIZE = 2
_INITIAL_FUNDS = 10
_NUM_ROUNDS = 3
_DECK = list(range(-6, 7))  # Cards from -6 to 6, including 0

_GAME_TYPE = pyspiel.GameType(
    short_name="spike_sabacc",
    long_name="Corellian Spike Sabacc",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_MAX_NUM_PLAYERS,
    min_num_players=_MIN_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=True,
    parameter_specification={
        "players": _MIN_NUM_PLAYERS,
    },
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(_DECK) + 2,  # Drawing, discarding, and betting
    max_chance_outcomes=len(_DECK),
    num_players=_MIN_NUM_PLAYERS,
    min_utility=-1,  # A player only loses their initial bet
    max_utility=_MIN_NUM_PLAYERS,  # Winner gets their own bet + opponents' bets
    utility_sum=0.0,
    max_game_length=_NUM_ROUNDS * _MIN_NUM_PLAYERS * 2,  # 3 rounds per player with actions
)


class SpikeSabacc(pyspiel.Game):
    """A Python version of Corellian Spike Sabacc."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        self.deck = _DECK.copy()
        self.num_rounds = _NUM_ROUNDS

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return SpikeSabaccState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return SpikeSabaccObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            self.num_players(),
            params,
        )
    

class SpikeSabaccState(pyspiel.State):
    """State of a Corellian Spike Sabacc game."""

    def __init__(self, game):
        super().__init__(game)
        self._num_players = game.num_players()
        self._deck = game.deck.copy()
        # print("deck: ", self._deck)
        random.shuffle(self._deck)
        # print("shuffled deck: " , self._deck)
        
        # Initialize player hands with two random cards
        self.hands = [[self._deck.pop(), self._deck.pop()] for _ in range(self._num_players)]
        
        # Each players funds
        self.funds = [_INITIAL_FUNDS] * self._num_players

        # Betting history
        self.bets = [0] * self._num_players
        
        # Round tracking
        self.current_round = 0
        # 0: Drawing phase, 1: Betting phase
        self.current_phase = 0
        self._current_player = 0
        self.game_over = False

    # not used
    def get_bets(self):
        return self.bets

    def __str__(self):
        """String representation for debugging."""
        return (
            f"Hands: {self.hands}, Current Player: {self.current_player()}, "
            f"Round: {self.current_round}, Bets: {self.bets}, Game Over: {self.game_over}"
        )
    
    def is_chance_node(self):
      """Returns whether the current node is a chance node."""
      # currently never, later add dice phase
      return 0

    def current_player(self):
        """Returns the current player."""
        if self.game_over:
            return pyspiel.PlayerId.TERMINAL
        return self._current_player

    def legal_actions(self, player):
        """Returns the list of legal actions for the given player."""
        if self.game_over or self.hands[player] == []:
            return []
        elif self.current_phase == 0:
            return [STAND_ACTION, DRAW_ACTION]
        else:
            return [BET_ACTION, FOLD_ACTION]

    def apply_action(self, action):
        """Applies the chosen action."""
        if action == DRAW_ACTION:
            self.hands[self.current_player()].append(self._deck.pop())
        elif action == BET_ACTION:
            # self.funds[self.current_player()] -= 1
            self.bets[self.current_player()] += 1
        elif action == FOLD_ACTION:
            # fold by removing the player's hand
            self.hands[self.current_player()] = []
        
        # Move to the next player
        self._current_player = (self.current_player() + 1) % self._num_players
        # Move to the next phase or round
        if self.current_player() == 0:
            if self.current_phase == 0:
                self.current_phase += 1
            elif self.current_phase == 1:
                self.current_round += 1
                self.current_phase = 0
        
        # Check if game is over
        if self.current_round >= _NUM_ROUNDS or self.one_player_left(self.hands):
            self.game_over = True

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal: {action}"
        elif action == STAND_ACTION:
            return "Stand"
        elif action == DRAW_ACTION:
            return "Draw"
        elif action == BET_ACTION:
            return "Bet"
        elif action == FOLD_ACTION:
            return "Fold"
        else:
            print("action: ", action)
            raise ValueError(f"Unknown action: {action}")
        
    def phase_to_string(self, phase):
        """Phase -> string."""
        return ["Drawing", "Betting"][phase]
    
    def chance_outcomes(self):
      """Returns possible chance outcomes and their probabilities."""
      if self.is_chance_node():
          probabilities = 1.0 / len(self._deck)
          return [(i, probabilities) for i in range(len(self._deck))]
      return []

    def is_terminal(self):
        """Returns whether the game is over."""
        # or all but one player folded
        return self.game_over
    
    def one_player_left(self, hands):
        """Returns whether only one player has a non-empty hand -> has not fold."""
        return sum([len(hand) > 0 for hand in hands]) == 1
    
    def one_player_left_in_game(self, hands):
        """Returns whether there is only one player left with chips."""
        return sum([fund > 0 for fund in self.funds]) == 1
    
    def returns(self):
        """Returns the final scores for all players."""
        if not self.game_over:
            return [0] * self._num_players
        
        # Compute final hand values
        hand_sums = [sum(hand) for hand in self.hands]
        # min score of player without empty hand
        winning_score = min([sum(hand) for hand in self.hands if hand != []])
        # Filter out players with empty hands
        valid_players = [i for i in range(self._num_players) if self.hands[i]] 
        winners = {i for i in valid_players if hand_sums[i] == winning_score}
        
        # Assign winnings
        winnings = sum(self.bets)/len(winners)
        return [winnings-self.bets[i] if i in winners else -self.bets[i] for i in range(self._num_players)]


class SpikeSabaccObserver:
    """Observer for Corellian Spike Sabacc, following the PyObserver interface."""

    def __init__(self, iig_obs_type, num_players, params=None):
        """Initializes an observation tensor."""
        del params
        self.num_players = num_players

        # Observation components
        pieces = [
            ("player", num_players, (num_players,)),  # One-hot encoding of current player
            ("current_round", 1, (1,)),  # Current round number
            ("current_phase", 1, (1,)),  # Drawing (0) or Betting (1)
        ]

        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            hand_size = _INITIAL_HAND_SIZE + _NUM_ROUNDS  # Maximum possible hand size
            pieces.append(("private_hand", hand_size, (hand_size,)))  # Player's hand

        if iig_obs_type.public_info:
            pieces.append(("bets", num_players, (num_players,)))  # Current bets of all players
            pieces.append(("game_over", 1, (1,)))  # Whether the game is over

        # Build observation tensor
        total_size = sum(size for _, size, _ in pieces)
        self.tensor = np.zeros(total_size, np.float32)

        # Create named views of tensor
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index : index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates the observation tensor based on the current state from `player`'s POV."""
        self.tensor.fill(0)

        if "player" in self.dict:
            self.dict["player"][player] = 1
        if "current_round" in self.dict:
            self.dict["current_round"][0] = state.current_round
        if "current_phase" in self.dict:
            self.dict["current_phase"][0] = state.current_phase
        if "private_hand" in self.dict:
            hand = np.full((_INITIAL_HAND_SIZE + _NUM_ROUNDS,), -999)  # Fill with -999
            hand[:len(state.hands[player])] = state.hands[player]  # Copy actual hand
            self.dict["private_hand"][:] = hand  # Store in observation tensor
        if "bets" in self.dict:
            self.dict["bets"][:] = state.get_bets()
        if "game_over" in self.dict:
            self.dict["game_over"][0] = int(state.is_terminal())

    def string_from(self, state, player):
        """Returns a string representation of the observation from `player`'s perspective."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f"p{player}")
        if "current_round" in self.dict:
            pieces.append(f"round:{state.current_round}")
        if "current_phase" in self.dict:
            phase_str = "Drawing" if state.current_phase == 0 else "Betting"
            pieces.append(f"phase:{phase_str}")
        if "private_hand" in self.dict:
            pieces.append(f"hand:{state.hands[player]}")
        if "bets" in self.dict:
            pieces.append(f"bets:{state.get_bets()}")
        if "game_over" in self.dict:
            pieces.append(f"game_over:{int(state.is_terminal())}")
        
        return " ".join(str(p) for p in pieces)



# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, SpikeSabacc)
