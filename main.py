import numpy as np
import random
import torch
import torch.nn.functional as F
from game import TicTacFoeEnv  # Assuming your TicTacFoeEnv class is in the "game" module

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = {}  # Maps state-action pairs to values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair"""
        return self.q_table.get((self.hash_state(state), action), torch.tensor(0.0, device=device))

    def set_q_value(self, state, action, value):
        """Sets the Q-value for a given state-action pair"""
        self.q_table[(self.hash_state(state), action)] = value

    def hash_state(self, state):
        """Hashes the board state to be used as a dictionary key"""
        board, replace_count, player = state
        return str(board) + str(replace_count) + player

    def choose_action(self, state, available_actions):
        """Choose the action based on the exploration/exploitation strategy (epsilon-greedy)"""
        if not available_actions:  # No available actions, return None or handle this case
            return None
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in available_actions]
            q_values_tensor = torch.stack(q_values)
            max_value = torch.max(q_values_tensor).item()
            best_actions = [available_actions[i] for i, v in enumerate(q_values) if v == max_value]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done, available_actions):
        """Update Q-values based on the agent's experience"""
        current_q = self.get_q_value(state, action)
        if done:
            max_future_q = torch.tensor(0.0, device=device)
        else:
            future_q_values = [self.get_q_value(next_state, a) for a in available_actions]
            max_future_q = torch.max(torch.stack(future_q_values))

        reward_tensor = torch.tensor(reward, device=device)
        new_q = current_q + self.learning_rate * (reward_tensor + self.discount_factor * max_future_q - current_q)
        self.set_q_value(state, action, new_q)

    def decay_exploration(self):
        """Decay the exploration rate over time"""
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, 0.2)


def run_batch_episodes(batch_size, agent_x, agent_o, env):
    """Runs a batch of episodes of Q-learning in parallel."""
    states = []
    dones = []

    # Initialize multiple environments for the batch
    for _ in range(batch_size):
        states.append(env.reset())
    
    dones = [False] * batch_size

    while not all(dones):
        for i in range(batch_size):
            if dones[i]:
                continue

            available_actions = [(r, c) for r in range(5) for c in range(5) if env.board[r][c] == " " or env.replace_count[r][c] < 2]
            
            if not available_actions:  # No actions left, continue
                dones[i] = True
                continue

            if env.current_player == "X":
                action = agent_x.choose_action(states[i], available_actions)
            else:
                action = agent_o.choose_action(states[i], available_actions)

            if action is None:  # No valid action, end the episode
                dones[i] = True
                continue

            next_state, reward, done = env.step(action)

            agent = agent_x if env.current_player == "X" else agent_o
            agent.learn(states[i], action, reward, next_state, done, available_actions)

            states[i] = next_state
            dones[i] = done

        # Decay exploration after each batch
        agent_x.decay_exploration()
        agent_o.decay_exploration()



def batch_training(episodes, env, agent_x, agent_o, batch_size=100000):
    """Batched training of the agents on the GPU."""
    for episode in range(0, episodes, batch_size):
        run_batch_episodes(batch_size, agent_x, agent_o, env)
        # Print progress after each batch
        print(f"Completed {episode + batch_size}/{episodes} episodes")


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

# Train using GPU and batch training
episodes = 1000000
batch_training(episodes, env, agent_x, agent_o)
print("Agent O")
evaluate_agent(agent_o, env, episodes=50)
print()
print("Agent X")
print()
evaluate_agent(agent_x, env, episodes=50)
# After training, play against the agent
play_vs_agent(env, agent_o)
play_vs_agent(env, agent_x)
