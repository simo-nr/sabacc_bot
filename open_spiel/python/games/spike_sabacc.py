
"""Corellian Spike Sabacc implemented in Python."""

import numpy as np
import random
import copy

import pyspiel


STAND_ACTION = 0
DRAW_ACTION = 1
BET_ACTION = 2
FOLD_ACTION = 3
WAIT_ACTION = 4

DECIDE_STATE = 0
APPLY_STATE = 1
BET_STATE = 2
DICE_STATE = 3
DEAL_STATE = 4

_MAX_NUM_PLAYERS = 10
_MIN_NUM_PLAYERS = 2
_INITIAL_HAND_SIZE = 2
_INITIAL_FUNDS = 3
_NUM_ROUNDS = 3
_DECK = list(range(-10, 11))  # Cards from -6 to 6, including 0

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
        self.deck = copy.deepcopy(_DECK)
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
        print("\033[91minit game \033[0m")
        super().__init__(game)
        self._num_players = game.num_players()

        self._full_deck = copy.deepcopy(game.deck)
        self.card_to_action = {card: idx for idx, card in enumerate(self._full_deck)}
        self.action_to_card = {idx: card for idx, card in enumerate(self._full_deck)}
        self._remaining_deck = copy.deepcopy(self._full_deck)
        
        # Initialize player hands with two random cards
        # random.shuffle(self._remaining_deck)        
        self.hands = [[self._remaining_deck.pop(), self._remaining_deck.pop()] for _ in range(self._num_players)]
        print("shuffled deck", self._remaining_deck)
        print("deck size", len(self._remaining_deck))
        print("initial hands ", self.hands)
        
        self._chance_event_type = None # "deck" or "dice"

        # Each players funds
        self.funds = [_INITIAL_FUNDS] * self._num_players
        # Betting history
        self.bets = [0] * self._num_players
        # Round tracking
        self._current_player = 0
        self._previous_starting_player = self._current_player
        self.current_round = 0
        self.current_state = DECIDE_STATE

        self.hand_over = False
        self.game_over = False
        print("end of init hands: ", self.hands)
        print(f"\033[93mDeck ID: {id(self._remaining_deck)}, Hands ID: {id(self.hands)}\033[0m")

    def __str__(self):
        """String representation for debugging."""
        return (
            f"Hands: {self.hands}, Current Player: {self.current_player()}, "
            f"Round: {self.current_round}, Phase: {self.state_to_string(self.current_state)}, "
            f"Bets: {self.bets}, Funds: {self.funds}, Round Over: {self.hand_over}, Game Over: {self.game_over}"
        )
    
    def is_chance_node(self):
      """Returns whether the current node is a chance node."""
      print("chance node hands: ", self.hands)
      return (self.current_state == APPLY_STATE 
              or self.current_state == DICE_STATE 
              or self.current_state == DEAL_STATE)

    def current_player(self):
        """Returns the current player."""
        if self.game_over:
            return pyspiel.PlayerId.TERMINAL
        return self._current_player

    def legal_actions(self):
        """Returns the list of legal actions for the given player."""
        print("legal actions hands: ", self.hands)        
        player = self.current_player()
        if self.game_over or self.hand_over or self.hands[player] == []:
            return [WAIT_ACTION]
        elif self.current_state == DECIDE_STATE:
            return [STAND_ACTION, DRAW_ACTION]
        elif self.funds[player] > 0:
            return [BET_ACTION, FOLD_ACTION]
        else:
            return [FOLD_ACTION, WAIT_ACTION]

    def _apply_action(self, action):
        """Applies the chosen action."""
        print("apply action hands: ", self.hands)
        if self.is_chance_node():
            # dice phase for every round
            if self.current_state == DICE_STATE:
                double_dice = False
                if double_dice == True:
                    self.current_state = DEAL_STATE
                    self._chance_event_type = "deck"
                else:
                    self.current_state = DECIDE_STATE
            if self.current_state == DEAL_STATE:
                dealt_card = self.action_to_card[action]

                print(f"Dealt card: {dealt_card}, Remaining deck: {self._remaining_deck}")
                if dealt_card not in self._remaining_deck:
                    raise ValueError(f"Attempted to deal a card {dealt_card} not in the deck: {self._remaining_deck}")
                
                self.hands[self._current_player].append(dealt_card)
                self._remaining_deck.remove(dealt_card)

                if len([hand for hand in self.hands if len(hand) == 2]) == self._num_players:
                    # everyone has 2 cards, move on to decide state
                    self.current_state = DECIDE_STATE
                else:
                    # move on to next player but dont change any state
                    self._current_player = (self.current_player() + 1) % self._num_players
                    # self._current_player = 50

                
            elif self.current_state == APPLY_STATE:
                drawn_card = self.action_to_card[action]

                print(f"Drawn card: {drawn_card}, Remaining deck: {self._remaining_deck}")
                if drawn_card not in self._remaining_deck:
                    raise ValueError(f"Card {drawn_card} is not in the remaining deck: {self._remaining_deck}")
                
                self.hands[self._current_player].append(drawn_card)
                self._remaining_deck.remove(drawn_card)
                # DECIDE state for next player
                self._current_player = (self.current_player() + 1) % self._num_players
                if self._current_player == 0:
                    self.current_state = BET_STATE
                else:
                    self.current_state = DECIDE_STATE

        elif action == DRAW_ACTION: # decision node with draw action
            # DECIDE STATE to APPLY STATE
            self.current_state = APPLY_STATE
            self._chance_event_type = "deck"
        else:
            if action == BET_ACTION:
                self.funds[self.current_player()] -= 1
                self.bets[self.current_player()] += 1
                
            elif action == FOLD_ACTION:
                print("\033[94m fold player ", self.current_player(), "\033[0m")
                self.hands[self.current_player()] = []
            
            self._current_player = (self.current_player() + 1) % self._num_players
            if self.current_player() == 0:
                if self.current_state == DECIDE_STATE:
                    self.current_state = BET_STATE
                elif self.current_state == BET_STATE:
                    self.current_state = DICE_STATE
                    self._chance_event_type = "dice"

            # check if round has ended
            if self.current_round + 1 >= _NUM_ROUNDS or self.one_player_left_in_round(self.hands):
                self.hand_over = True
        
        action = None
        if self.hand_over:
            print("hand over")
            self.hand_over = False
            # winner gets all bets
            self.funds = [fund + winning for fund, winning in zip(self.funds, self.calc_winnings())]
            self.bets = [0] * self._num_players
            self.hands = [[] for _ in range(self._num_players)]
            self._remaining_deck = copy.deepcopy(self._full_deck)
            print("\033[92m remaining deck after reset: ", self._remaining_deck, "\033[0m")
            self.current_round = 0
            self.current_state = DEAL_STATE
            self._chance_event_type = "deck"

        # Check if game is over
        if self.one_player_left_in_game():
            self.game_over = True

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal: {action}"
        elif self.is_chance_node():
            return f"Take {action}"
        else:
            return ["Stand", "Draw", "Bet", "Fold", "Wait"][action]
        
    def state_to_string(self, phase):
        """Phase -> string."""
        return ["Decide", "Apply", "Bet", "Dice", "Deal"][phase]
    
    def chance_outcomes(self):
        print("chance outcomes hand: ", self.hands)
        if self._chance_event_type == "deck":
            remaining_cards = len(self._remaining_deck)
            probability = 1.0 / remaining_cards
            return [(self.card_to_action[card], probability) for card in self._remaining_deck]
        elif self._chance_event_type == "dice":
            return [(i, 1.0 / 6) for i in range(1, 7)]  # 6-sided die
        if self._chance_event_type is None:
            raise ValueError("Chance event type is not set!")

    def is_terminal(self):
        """Returns whether the game is over."""
        return self.game_over
    
    def one_player_left_in_round(self, hands):
        """Returns whether only one player has a non-empty hand -> has not fold."""
        return sum([len(hand) > 0 for hand in hands]) == 1
    
    def one_player_left_in_game(self):
        """Returns whether there is only one player left with chips, in the pot or in their funds."""
        return sum([(fund > 0 or self.bets[i] > 0) for i, fund in enumerate(self.funds)]) == 1
    
    def returns(self):
        return self.funds
    
    def calc_winnings(self):
        # Compute final hand values
        hand_sums = [sum(hand) for hand in self.hands]
        # min score of player without empty hand
        winning_score = abs(min([sum(hand) for hand in self.hands if hand != []]))
        # Filter out players with empty hands
        valid_players = [i for i in range(self._num_players) if self.hands[i]] 
        winners = {i for i in valid_players if abs(hand_sums[i]) == winning_score}

        if not winners:
            # no winners, everybody gets money back
            return self.bets
        else:
            # Assign winnings
            winnings_pp = sum(self.bets)/len(winners)
            return [winnings_pp if i in winners else 0 for i in range(self._num_players)]


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
