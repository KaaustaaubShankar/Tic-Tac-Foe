import numpy as np
import random
import concurrent.futures
import multiprocessing
from game import TicTacFoeEnv  # Assuming your TicTacFoeEnv class is in the "game" module

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = {}  # Maps state-action pairs to values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair"""
        return self.q_table.get((self.hash_state(state), action), 0)

    def set_q_value(self, state, action, value):
        """Sets the Q-value for a given state-action pair"""
        self.q_table[(self.hash_state(state), action)] = value

    def hash_state(self, state):
        """Hashes the board state to be used as a dictionary key"""
        board, replace_count, player = state
        return str(board) + str(replace_count) + player

    def choose_action(self, state, available_actions):
        """Choose the action based on the exploration/exploitation strategy (epsilon-greedy)"""
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_value = max(q_values)
            best_actions = [available_actions[i] for i, v in enumerate(q_values) if v == max_value]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done, available_actions):
        """Update Q-values based on the agent's experience"""
        current_q = self.get_q_value(state, action)
        max_future_q = 0 if done else max([self.get_q_value(next_state, a) for a in available_actions])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.set_q_value(state, action, new_q)

    def decay_exploration(self):
        """Decay the exploration rate over time"""
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, 0.2)  


def evaluate_agent(agent, env, episodes=50):
    """Evaluate the trained agent against a static opponent."""
    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_reward": 0
    }

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            available_actions = [(i, j) for i in range(5) for j in range(5) if env.board[i][j] == " " or env.replace_count[i][j] < 2]
            
            if env.current_player == "X":
                action = agent.choose_action(state, available_actions)
            else:
                action = random.choice(available_actions)
            
            next_state, reward, done = env.step(action)
            results["total_reward"] += reward
            
            state = next_state

            if done:
                if reward > 0:
                    results["wins"] += 1 if env.current_player == "X" else 0
                    results["losses"] += 1 if env.current_player == "O" else 0
                else:
                    results["draws"] += 1

    print(f"Results after {episodes} episodes:")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    print(f"Win Rate: {results['wins'] / episodes * 100:.2f}%")


def calculate_random_moves(episode, total_episodes=100000, min_moves=0, max_moves=999, step_size=10000):
    """Calculate the number of random moves based on the current episode in steps of 10,000 episodes."""
    step = episode // step_size
    moves = min(min_moves + step, max_moves)
    return moves


def run_single_episode(episode, agent_x, agent_o, env):
    """Runs a single episode of Q-learning."""
    last_random_start_moves = calculate_random_moves(episode)

    if random.uniform(0, 1) < 0.25:
        state = env.reset(random_start_moves=last_random_start_moves)
    else:
        state = env.reset()

    done = False
    while not done:
        available_actions = [(i, j) for i in range(5) for j in range(5) if env.board[i][j] == " " or env.replace_count[i][j] < 2]

        if env.current_player == "X":
            action = agent_x.choose_action(state, available_actions)
        else:
            action = agent_o.choose_action(state, available_actions)

        next_state, reward, done = env.step(action)

        if env.current_player == "X":
            agent_x.learn(state, action, reward, next_state, done, available_actions)
        else:
            agent_o.learn(state, action, -reward, next_state, done, available_actions)

        state = next_state

    agent_x.decay_exploration()
    agent_o.decay_exploration()

    return None


def parallel_training(episodes, env, agent_x, agent_o, num_workers=26, batch_size=10000):
    """Parallel training of the agents in batches of `batch_size` episodes"""
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for start in range(0, episodes, batch_size):
            end = min(start + batch_size, episodes)
            futures = [executor.submit(run_single_episode, episode, agent_x, agent_o, env) for episode in range(start, end)]
            
            # Wait for all the episodes in the current batch to finish
            for future in concurrent.futures.as_completed(futures):
                future.result()

            # Evaluate after each batch
            print(f"Completed episodes: {end}/{episodes}")
            evaluate_agent(agent_x, env, episodes=100)  # Evaluating the agent

            print(f"Proceeding with the next batch...")


def play_vs_agent(env, agent_o):
    """Play a game against the Q-learning agent (you are X, agent is O)"""
    state = env.reset()
    done = False
    while not done:
        env.render()
        available_actions = [(i, j) for i in range(5) for j in range(5) if env.board[i][j] == " " or env.replace_count[i][j] < 2]

        if env.current_player == "X":
            row = int(input("Enter row (0-4): "))
            col = int(input("Enter col (0-4): "))
            action = (row, col)
        else:
            action = agent_o.choose_action(state, available_actions)
            print(f"Agent O chooses {action}")

        try:
            next_state, reward, done = env.step(action)
        except ValueError as e:
            print(e)
            continue

        if done:
            env.render()
            if reward > 0:
                print("You (X) win!" if env.current_player == "X" else "Agent O wins!")
            else:
                print("It's a draw!")
            break

        state = next_state


# Simulation setup
env = TicTacFoeEnv()
agent_x = QLearningAgent()
agent_o = QLearningAgent()

# Parallel training with multiprocessing
episodes = 100000
parallel_training(episodes, env, agent_x, agent_o)

# After training, play against the agent
play_vs_agent(env, agent_o)