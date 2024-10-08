import numpy as np
import random

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
        board, player = state
        return str(board) + player

    def choose_action(self, state, available_actions):
        """Choose the action based on the exploration/exploitation strategy (epsilon-greedy)"""
        # Force random action with a small probability (e.g., 10%)
        if random.uniform(0, 1) < 0.1:
            return random.choice(available_actions)
        
        # Standard epsilon-greedy strategy
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in available_actions]
            return available_actions[np.argmax(q_values)]


    def learn(self, state, action, reward, next_state, done, available_actions):
        """Update Q-values based on the agent's experience"""
        current_q = self.get_q_value(state, action)
        max_future_q = 0 if done else max([self.get_q_value(next_state, a) for a in available_actions])
        
        # Q-learning update rule: Q(s,a) ← Q(s,a) + α * (r + γ * maxQ(s',a') - Q(s,a))
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
            
            if env.current_player == "X":  # Agent's turn
                action = agent.choose_action(state, available_actions)
            else:  # Static opponent's turn
                action = random.choice(available_actions)  # Or implement a basic strategy
            
            next_state, reward, done = env.step(action)
            results["total_reward"] += reward
            
            state = next_state

            if done:
                if reward == 1:  # Agent won
                    results["wins"] += 1
                elif reward == -1:  # Agent lost
                    results["losses"] += 1
                else:  # Draw
                    results["draws"] += 1

    # Print evaluation results
    print(f"Results after {episodes} episodes:")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    print(f"Win Rate: {results['wins'] / episodes * 100:.2f}%")

def calculate_random_moves(episode, total_episodes=100000, min_moves=3, max_moves=75, step_size=10000):
    """Calculate the number of random moves based on the current episode in steps of 10,000 episodes."""
    # Calculate which step we're in (each step is 10,000 episodes)
    step = episode // step_size

    # Cap the step to ensure it does not exceed the range for max_moves
    moves = min(min_moves + step, max_moves)
    
    return moves


# Simulation
from game import TicTacFoeEnv  # Assuming your TicTacFoeEnv class is in the "game" module
env = TicTacFoeEnv()
agent_x = QLearningAgent()
agent_o = QLearningAgent()


# Initialize variables
episodes = 100000
last_random_start_moves = None  # To store the last calculated random start moves

for episode in range(episodes):
    # Check if the random start moves should be recalculated
    current_step = episode // 1000
    if last_random_start_moves is None or current_step != (episode - 1) // 1000:
        # Recalculate random_start_moves only if the episode has crossed into a new step
        last_random_start_moves = calculate_random_moves(episode, episodes,step_size=1000)

    # 25% of games start with random board positions, scaling random moves with episodes
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

        if done:
            agent_x.decay_exploration()
            agent_o.decay_exploration()

    # Periodically evaluate the agent
    if episode % 100 == 0:
        print(f"Episode {episode}/{episodes}")
    
    if episode % 1000 == 0:
        evaluate_agent(agent_x, env, episodes=100)


def play_vs_agent(env, agent_o):
    """Play a game against the Q-learning agent (you are X, agent is O)"""
    state = env.reset()
    done = False
    while not done:
        env.render()
        available_actions = [(i, j) for i in range(5) for j in range(5) if env.board[i][j] == " " or env.replace_count[i][j] < 2]

        if env.current_player == "X":
            # Human's turn
            row = int(input("Enter row (0-4): "))
            col = int(input("Enter col (0-4): "))
            action = (row, col)
        else:
            # Agent O's turn
            action = agent_o.choose_action(state, available_actions)
            print(f"Agent O chooses {action}")

        try:
            next_state, reward, done = env.step(action)
        except ValueError as e:
            print(e)
            continue

        if done:
            env.render()
            if reward == 1:
                print("You (X) win!")
            elif reward == -1:
                print("Agent O wins!")
            else:
                print("It's a draw!")
            break

        state = next_state


# Play against the trained agent
play_vs_agent(env, agent_o)
