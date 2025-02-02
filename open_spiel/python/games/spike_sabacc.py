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
        print("deck: ", self._deck)
        random.shuffle(self._deck)
        print("shuffled deck: " , self._deck)
        
        # Initialize player hands with two random cards
        self.hands = [[self._deck.pop(), self._deck.pop()] for _ in range(self._num_players)]
        
        # Betting history
        self.bets = [0] * self._num_players
        
        # Round tracking
        self.current_round = 0
        # 0: Drawing phase, 1: Betting phase
        self.current_phase = 0
        self._current_player = 0
        self.game_over = False

    def __str__(self):
        """String representation for debugging."""
        return (
            f"Hands: {self.hands}, Current Player: {self.current_player()}, "
            f"Round: {self.current_round}, Bets: {self.bets}, Game Over: {self.game_over}"
        )
    
    def is_chance_node(self):
      """Returns whether the current node is a chance node."""
      return self.current_phase == 1

    def current_player(self):
        """Returns the current player."""
        if self.game_over:
            return pyspiel.PlayerId.TERMINAL
        return self._current_player

    def legal_actions(self, player):
        """Returns the list of legal actions for the given player."""
        if self.game_over:
            return []
        elif self.is_chance_node():
            return [STAND_ACTION, DRAW_ACTION]
        else:
            return [BET_ACTION, FOLD_ACTION]
        # print("deck list: ", list(range(len(_DECK))))
        # print("offset: ", BID_ACTION_OFFSET)
        # print("legal actions: ", list(range(len(_DECK))) + [BID_ACTION_OFFSET])
        # return list(range(len(_DECK))) + [BID_ACTION_OFFSET]  # Draw a card or place a bet

    def apply_action(self, action):
        """Applies the chosen action."""
        if action < len(_DECK):
            # Replace a card in hand with the chosen new card
            self.hands[self.current_player()].append(_DECK[action])
            self.hands[self.current_player()].pop(0)  # Remove the oldest card
        else:
            # Betting logic (for now, simple tracking)
            self.bets[self.current_player()] += 1
        
        # Move to the next player
        self._current_player = (self.current_player() + 1) % self._num_players
        if self.current_player() == 0:
            self.current_round += 1
        
        # Check if game is over
        if self.current_round >= _NUM_ROUNDS:
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
            raise ValueError(f"Unknown action: {action}")
    
    def chance_outcomes(self):
      """Returns possible chance outcomes and their probabilities."""
      if self.is_chance_node():
          probabilities = 1.0 / len(self._deck)
          return [(i, probabilities) for i in range(len(self._deck))]
      return []

    def is_terminal(self):
        """Returns whether the game is over."""
        return self.game_over

    def returns(self):
        """Returns the final scores for all players."""
        if not self.game_over:
            return [0] * self._num_players
        
        # Compute final hand values
        hand_sums = [sum(hand) for hand in self.hands]
        winner = min(range(self._num_players), key=lambda i: abs(hand_sums[i]))
        
        # Assign winnings
        winnings = sum(self.bets)
        return [winnings if i == winner else -self.bets[i] for i in range(self._num_players)]



class SpikeSabaccState2(pyspiel.State):
  """A python version of the Liars Poker state."""

  def winner(self):
    """Returns the id of the winner if the bid originator has won.

    -1 otherwise.
    """
    return self._winner

  def loser(self):
    """Returns the id of the loser if the bid originator has lost.

    -1 otherwise.
    """
    return self._loser

  def _is_challenge_possible(self):
    """A challenge is possible once the first bid is made."""
    return self._current_action != -1

  def _is_rebid_possible(self):
    """A rebid is only possible when all players have challenged the original bid."""
    return not self.is_rebid and self._num_challenges == self._num_players - 1

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    actions = []

    if self._is_challenge_possible():
      actions.append(CHALLENGE_ACTION)

    if player != self._bid_originator or self._is_rebid_possible():
      # Any move higher than the current bid is allowed.
      # Bids start at BID_ACTION_OFFSET (1) as 0 represents the challenge
      # action.
      for bid in range(
          max(BID_ACTION_OFFSET, self._current_action + 1), self._max_bid + 1
      ):
        actions.append(bid)

    return actions

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    probability = 1.0 / self._num_digits
    return [(digit, probability) for digit in self._deck]

  def _decode_bid(self, bid):
    """Turns a bid ID to a (count, number) tuple.

    For example, take 2 players each with 2 numbers from the deck of 1, 2, and
    3.
      - A bid of two 1's would correspond to a bid id 1.
        - Explanation: 1 is the lowest number, and the only lower bid would be
        zero 1's.
      - A bid of three 3's would correspond to a bid id 10.
        - Explanation: 1-4 1's take bid ids 0-3. 1-4 2's take bid ids 4-7. 1 and
        2 3's take bid ids 8 and 9.

    Args:
      bid: Bid ID in the range 0 to self._max_bid (non-inclusive).

    Returns:
      A tuple of (count, number). For example, (1, 2) represents one 2's.
    """
    number = bid % self._num_digits + 1
    count = bid // self._num_digits + 1
    return (count, number)

  def encode_bid(self, count, number):
    """Turns a count and number into a bid ID.

    Bid ID is in the range 0 to self._max_bid (non-inclusive).

    For example, take 2 players each with 2 numbers from the deck of 1, 2, and
    3.
      - A count of 2 and number of 1 would be a bid of two one's and a bid id 1.
        - Explanation: 1 is the lowest number, and the only lower bid would be
        zero 1's
          corresponding to bid id 0.

    Args:
      count: The count of the bid.
      number: The number of the bid.

    Returns:
      A single bid ID.
    """
    return (count - 1) * self._num_digits + number - 1

  def _counts(self):
    """Determines if the bid originator wins or loses."""
    bid_count, bid_number = self._decode_bid(
        self._current_action - BID_ACTION_OFFSET
    )

    # Count the number of bid_numbers from all players.
    matches = 0
    for player_id in range(self._num_players):
      for digit in self.hands[player_id]:
        if digit == bid_number:
          matches += 1

    # If the number of matches are at least the bid_count bid, then the bidder
    # wins. Otherwise everyone else wins.
    if matches >= bid_count:
      self._winner = self._bid_originator
    else:
      self._loser = self._bid_originator

  def _update_bid_history(self, bid, player):
    """Writes a player's bid into memory."""
    self.bid_history[bid][player] = 1

  def _update_challenge_history(self, bid, player):
    """Write a player's challenge for a bid into memory."""
    self.challenge_history[bid][player] = 1

  def _apply_action(self, action):
    """Applies an action and updates the state."""
    if self.is_chance_node():
      # If we are still populating hands, draw a number for the current player.
      self.hands[self._current_player].append(action)
    elif action == CHALLENGE_ACTION:
      assert self._is_challenge_possible()
      self._update_challenge_history(
          self._current_action - BID_ACTION_OFFSET, self._current_player
      )
      self._num_challenges += 1
      # If there is no ongoing rebid, check if all players challenge before
      # counting. If there is an ongoing rebid, count once all the players
      # except the bidder challenges.
      if (not self.is_rebid and self._num_challenges == self._num_players) or (
          self.is_rebid and self._num_challenges == self._num_players - 1
      ):
        self._counts()
    else:
      # Set the current bid to the action.
      self._current_action = action
      if self._current_player == self._bid_originator:
        # If the bid originator is bidding again, we have a rebid.
        self.is_rebid = True
      else:
        # Otherwise, we have a regular bid.
        self.is_rebid = False
      # Set the bid originator to the current player.
      self._bid_originator = self._current_player
      self._update_bid_history(
          self._current_action - BID_ACTION_OFFSET, self._current_player
      )
      self._num_challenges = 0
    self._current_player = (self._current_player + 1) % self._num_players

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal: {action}"
    elif action == CHALLENGE_ACTION:
      return "Challenge"
    else:
      count, number = self._decode_bid(action - BID_ACTION_OFFSET)
      return f"Bid: {count} of {number}"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._winner >= 0 or self._loser >= 0

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if self._winner != -1:
      bidder_reward = self._num_players - 1
      others_reward = -1.0
    elif self._loser != -1:
      bidder_reward = -1 * (self._num_players - 1)
      others_reward = 1.0
    else:
      # Game is not over.
      bidder_reward = 0.0
      others_reward = 0.0
    return [
        others_reward if player_id != self._bid_originator else bidder_reward
        for player_id in range(self._num_players)
    ]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    if self._current_action != -1:
      count, number = self._decode_bid(self._current_action - BID_ACTION_OFFSET)
    else:
      count, number = "None", "None"
    return (
        "Hands: {}, Bidder: {}, Current Player: {}, Current Bid: {} of {},"
        " Rebid: {}".format(
            self.hands,
            self._bid_originator,
            self.current_player(),
            count,
            number,
            self.is_rebid,
        )
    )


class SpikeSabaccObserver:
  """Observer, conforming to the PyObserver interface (see observation.py).

  An observation will consist of the following:
    - One hot encoding of the current player number: [0 0 0 1 0 0 0]
    - A vector of length hand_length containing the digits in a player's hand.
    - Two matrices each of size (hand_length * num_digits * num_players,
    num_players)
      will store bids and challenges respectively. Each row in the matrix
      corresponds
      to a particular bid (e.g. one 1, two 5s, or eight 3s). 0 will represent no
      action. 1 will represent a player's bid or a player's challenge.
    - One bit for whether we are rebidding: [1] rebid occuring, [0] otherwise
    - One bit for whether we are counting: [1] COUNTS called, [0] otherwise
  """

  def __init__(
      self, iig_obs_type, num_players, hand_length, num_digits, params=None
  ):
    """Initiliazes an empty observation tensor."""
    del params
    self.num_players = num_players
    self.hand_length = hand_length

    # Determine which observation pieces we want to include.
    # Pieces is a list of tuples containing observation pieces.
    # Pieces are described by their (name, number of elements, and shape).
    pieces = [(
        "player",
        num_players,
        (num_players,),
    )]  # One-hot encoding for the player id.
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      # Vector containing the digits in a player's hand
      pieces.append(("private_hand", hand_length, (hand_length,)))
    if iig_obs_type.public_info:
      pieces.append(("rebid_state", 1, (1,)))
      pieces.append(("counts_state", 1, (1,)))
      if iig_obs_type.perfect_recall:
        # One-hot encodings for players' moves at every round.
        total_possible_rounds = hand_length * num_digits * num_players
        pieces.append((
            "bid_history",
            total_possible_rounds * num_players,
            (total_possible_rounds, num_players),
        ))
        pieces.append((
            "challenge_history",
            total_possible_rounds * num_players,
            (total_possible_rounds, num_players),
        ))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index : index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if "player" in self.dict:
      self.dict["player"][player] = 1
    if (
        "private_hand" in self.dict
        and len(state.hands[player]) == self.hand_length
    ):
      self.dict["private_hand"] = np.asarray(state.hands[player])
    if "rebid_state" in self.dict:
      self.dict["rebid_state"][0] = int(state.is_rebid)
    if "counts_state" in self.dict:
      self.dict["counts_state"][0] = int(state.is_terminal())
    if "bid_history" in self.dict:
      self.dict["bid_history"] = state.bid_history
    if "challenge_history" in self.dict:
      self.dict["challenge_history"] = state.challenge_history

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if (
        "private_hand" in self.dict
        and len(state.hands[player]) == self.hand_length
    ):
      pieces.append(f"hand:{state.hands[player]}")
    if "rebid_state" in self.dict:
      pieces.append(f"rebid:{[int(state.is_rebid)]}")
    if "counts_state" in self.dict:
      pieces.append(f"counts:{[int(state.is_terminal())]}")
    if "bid_history" in self.dict:
      for bid in range(len(state.bid_history)):
        if np.any(state.bid_history[bid] == 1):
          pieces.append("b:{}.".format(bid))
    if "challenge_history" in self.dict:
      for bid in range(len(state.challenge_history)):
        if np.any(state.challenge_history[bid] == 1):
          pieces.append("c:{}.".format(bid))
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, SpikeSabacc)
