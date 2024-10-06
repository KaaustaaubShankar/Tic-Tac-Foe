from game import TicTacFoeEnv 
from QLearningAgent import QLearningAgent  
from MinimaxAgent import MinimaxAgent  
import random
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

# Initialize the environment
env = TicTacFoeEnv()

# Initialize Q-learning agents and Minimax agent
agent_x = QLearningAgent()  # Q-learning agent for X
agent_o = MinimaxAgent(env, depth=3)  # Minimax agent for O (set depth to control search depth)

# Initialize variables
episodes = 100000
last_random_start_moves = None  # To store the last calculated random start moves

# Function to calculate the number of random start moves
def calculate_random_moves(episode, total_episodes, step_size):
    # Example logic to calculate the number of random moves based on episode progress
    max_random_moves = 10  # Set max random moves
    return min((episode // step_size) % max_random_moves, 5)

for episode in range(episodes):
    # Check if the random start moves should be recalculated
    current_step = episode // 1000
    if last_random_start_moves is None or current_step != (episode - 1) // 1000:
        # Recalculate random_start_moves only if the episode has crossed into a new step
        last_random_start_moves = calculate_random_moves(episode, episodes, step_size=1000)

    # 25% of games start with random board positions, scaling random moves with episodes
    if random.uniform(0, 1) < 0.25:
        state = env.reset(random_start_moves=last_random_start_moves)
    else:
        state = env.reset()

    done = False
    while not done:
        available_actions = [(i, j) for i in range(5) for j in range(5) if env.board[i][j] == " " or env.replace_count[i][j] < 2]

        # Player X (QLearning Agent) makes a move
        if env.current_player == "X":
            action = agent_x.choose_action(state, available_actions)
        # Player O (Minimax Agent) makes a move
        else:
            action = agent_o.get_best_move()

        next_state, reward, done = env.step(action)

        # Update Q-learning agent (Player X) only
        if env.current_player == "X":
            agent_x.learn(state, action, reward, next_state, done, available_actions)
        
        state = next_state

        # If the game is over, reduce exploration rate for Q-learning agent
        if done:
            agent_x.decay_exploration()

    # Periodically evaluate the agent
    print(f"Episode {episode}/{episodes}")
    
    # Periodically evaluate Q-learning agent performance (every 1000 episodes)
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
