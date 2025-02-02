from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
import open_spiel.python.games.spike_sabacc as spike_sabacc
import open_spiel.python.games.liars_poker as liars_poker
import open_spiel.python.games


# Create the environment
env = rl_environment.Environment("spike_sabacc")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

# Create the agents
agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
]

# Train the Q-learning agents in self-play.
for cur_episode in range(1000):
  if cur_episode % 100 == 0:
    print(f"Episodes: {cur_episode}")
  time_step = env.reset()
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    agent_output = agents[player_id].step(time_step)
    time_step = env.step([agent_output.action])
  # Episode is over, step all agents with final info state.
  for agent in agents:
    agent.step(time_step)
print("Done!")

# # Evaluate the Q-learning agent against a random agent.
# from open_spiel.python.algorithms import random_agent
# eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000") ]

# time_step = env.reset()
# while not time_step.last():
#   print("")
#   print(env.get_state)
#   player_id = time_step.observations["current_player"]
#   # Note the evaluation flag. A Q-learner will set epsilon=0 here.
#   agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
#   print(f"Agent {player_id} chooses {env.get_state.action_to_string(agent_output.action)}")
#   time_step = env.step([agent_output.action])

# print("")
# print(env.get_state)
# print(time_step.rewards)
