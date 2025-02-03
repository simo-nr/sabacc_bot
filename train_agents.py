import pickle
import os
from open_spiel.python import rl_environment
# from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import cfr
import open_spiel.python.games.spike_sabacc
import numpy as np
import pyspiel

# Create the environment
env = rl_environment.Environment("spike_sabacc")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

game = pyspiel.load_game("spike_sabacc")  # Load game properly

def save_agents(agents, filename="trained_agents.pkl"):
    """Saves the trained agents to a file."""
    with open(filename, "wb") as f:
        pickle.dump(agents, f)
    print(f"Agents saved to {filename}")

def load_agents(filename="trained_agents.pkl"):
    """Loads trained agents from a file if it exists."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            agents = pickle.load(f)
        print(f"Agents loaded from {filename}")
        return agents
    return None  # Return None if file does not exist

def trainQAgents(num_episodes, save_path):
    """Train agents or load them if already trained."""
    agents = load_agents(save_path)
    if agents:  # If agents are already trained, return them
        return agents

    # Otherwise, train new agents
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    for cur_episode in range(num_episodes):
        if cur_episode % 100 == 0:
            print(f"Training Easy Agents - Episode: {cur_episode}")

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])

        for agent in agents:
            agent.step(time_step)

    print("Agents trained!")
    save_agents(agents, save_path)
    return agents

def trainAgents(num_iterations, save_path):
    """Train CFR agents or load them if already trained."""
    
    # Try to load pre-trained agents
    try:
        with open(save_path, "rb") as f:
            solver = pickle.load(f)
        print("Loaded pre-trained CFR agents.")
        return solver
    except FileNotFoundError:
        print("No pre-trained agents found. Training new CFR agents...")

    # Initialize CFR solver
    solver = cfr.CFRSolver(game)

    # Train for the specified number of iterations
    for iteration in range(num_iterations):
        if iteration % 100 == 0:
            print(f"Training CFR Agents - Iteration: {iteration}")
        solver.evaluate_and_update_policy()

    # Save trained agent
    with open(save_path, "wb") as f:
        pickle.dump(solver, f)

    print("CFR Agents trained and saved!")
    return solver

def test_agents(solver, agent, env, num_episodes):
    """Tests two trained agents against each other and calculates win rates."""
    eval_agents = [solver, agent]
    
    wins = [0, 0]  # Wins for agent1 and agent2
    print("Testing agents...")
    
    for _ in range(num_episodes):
        time_step = env.reset()
        # information = ""
        while not time_step.last():
            # information += "\n" + str(env.get_state)

            player_id = time_step.observations["current_player"]
            if player_id == 0:
                strategy = solver.average_policy()  
                # Get legal actions for the current player
                legal_actions = env.get_legal_actions(player_id)
                # Get the probability distribution over actions from CFR
                action_probs = [strategy.action_probabilities(player_id)[a] for a in legal_actions]
                # Choose an action based on CFR probabilities
                chosen_action = np.random.choice(legal_actions, p=action_probs)
                # Apply the chosen action to the environment
                time_step = env.step([chosen_action])
            elif player_id == 1:
                agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])

            # information += f", Agent {player_id} chooses {env.get_state.action_to_string(agent_output.action)}"
            # print(str(env.get_state))
        
        # information += "\n" + str(env.get_state)

        # Reward indicates the winner
        if time_step.rewards[0] > time_step.rewards[1]:
            print("\033[91mrewards: ", time_step.rewards, "\033[0m")
            wins[0] += 1
        elif time_step.rewards[1] > time_step.rewards[0]:
            print("\033[92mrewards: ", time_step.rewards, "\033[0m")
            wins[1] += 1
        # else:
            # print(information)
            # print("rewards: ", time_step.rewards)

    win_rates = [wins[0] / num_episodes, wins[1] / num_episodes]
    return win_rates

# Train or load the easy agents
easyAgents = trainQAgents(num_episodes=5, save_path="easy_agents_5.pkl")

# Train or load the hard agents
solver = trainAgents(num_iterations=10, save_path="hard_agents_cfr_10.pkl")

print("training done")

# Test the two trained agents against each other
# win_rates1 = test_agents(easyAgents[0], solver, env, num_episodes=10)
# win_rates2 = test_agents(solver, easyAgents[0], env, num_episodes=10)
# print(f"Easy agent win rate: {(win_rates1[0] + win_rates2[1])/2}")
# print(f"Hard agent win rate: {(win_rates1[1] + win_rates2[0])/2}")
