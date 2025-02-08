from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
import open_spiel.python.games.spike_sabacc as spike_sabacc
# import open_spiel.python.games.liars_poker as liars_poker
# import open_spiel.python.games


# Create the environment
env = rl_environment.Environment("spike_sabacc")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

def trainEasyAgents(num_episodes):
    # Create two Q-learning agents
    easyAgents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    # Train the agents in self-play
    for cur_episode in range(num_episodes):
        if cur_episode % 100 == 0:
            print(f"Episodes: {cur_episode}")
        
        time_step = env.reset()
        
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = easyAgents[player_id].step(time_step)  # Each agent makes a move
            time_step = env.step([agent_output.action])

        # Episode is over, both agents update their Q-values
        for agent in easyAgents:
            agent.step(time_step)
    print("Easy agents trained!")
    return easyAgents

def trainHardAgents(num_episodes):
    # Create two Q-learning agents
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    for cur_episode in range(num_episodes):
        if cur_episode % 100 == 0:
            print(f"Episodes: {cur_episode}")
        
        time_step = env.reset()
        
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)  # Each agent makes a move
            time_step = env.step([agent_output.action])

        # Episode is over, both agents update their Q-values
        for agent in agents:
            agent.step(time_step)

    print("Hard agents trained!")
    return agents


def test_agents(agent1, agent2, env, num_episodes=100):
    """Tests two trained agents against each other and calculates win rates."""
    eval_agents = [agent1, agent2]
    
    wins = [0, 0]  # Wins for agent1 and agent2
    
    for _ in range(num_episodes):
        time_step = env.reset()
        information = ""
        while not time_step.last():
            information += "\n" + str(env.get_state)

            player_id = time_step.observations["current_player"]
            agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)

            information += f", Agent {player_id} chooses {env.get_state.action_to_string(agent_output.action)}"

            time_step = env.step([agent_output.action])
        
        information += "\n" + str(env.get_state)

        # Reward indicates the winner
        if time_step.rewards[0] > time_step.rewards[1]:
            # print("\033[91mrewards: ", time_step.rewards, "\033[0m")
            wins[0] += 1
        elif time_step.rewards[1] > time_step.rewards[0]:
            # print("\033[92mrewards: ", time_step.rewards, "\033[0m")
            wins[1] += 1
        # else:
            # print(information)
            # print("rewards: ", time_step.rewards)

    win_rates = [wins[0] / num_episodes, wins[1] / num_episodes]
    return win_rates


# Train the easy agents
easyAgents = trainEasyAgents(num_episodes=100)
# Train the hard agents
agents = trainHardAgents(num_episodes=1000)
# Test the two trained agents against each other
win_rates1 = test_agents(easyAgents[0], agents[0], env, num_episodes=100)
print(f"Easy agent win rate: {win_rates1[0]}")
print(f"Hard agent win rate: {win_rates1[1]}")

win_rates2 = test_agents(agents[0], easyAgents[0], env, num_episodes=100)
print(f"Easy agent win rate: {win_rates2[1]}")
print(f"Hard agent win rate: {win_rates2[0]}")

print("")

total = [(win_rates1[0] + win_rates2[1])/2, (win_rates1[1] + win_rates2[0])/2]
print(f"Easy agent win rate: {total[0]}")
print(f"Hard agent win rate: {total[1]}")
