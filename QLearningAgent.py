import random
import numpy as np
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
  