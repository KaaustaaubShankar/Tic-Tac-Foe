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

import multiprocessing
from game import TicTacFoeEnv
from QLearningAgent import QLearningAgent
from MinimaxAgent import MinimaxAgent
import random

def worker(episodes_per_worker, shared_q_table, lock):
    env = TicTacFoeEnv()
    agent_x = QLearningAgent(q_table=shared_q_table)  # Pass the shared Q-table
    agent_o = MinimaxAgent(env, depth=3)

    for episode in range(episodes_per_worker):
        state = env.reset()
        done = False
        while not done:
            available_actions = [(i, j) for i in range(5) for j in range(5)
                                 if env.board[i][j] == " " or env.replace_count[i][j] < 2]

            if env.current_player == "X":
                action = agent_x.choose_action(state, available_actions)
            else:
                action = agent_o.get_best_move()

            next_state, reward, done = env.step(action)

            if env.current_player == "X":
                with lock:
                    agent_x.learn(state, action, reward, next_state, done, available_actions)

            state = next_state

            if done:
                agent_x.decay_exploration()
        print(f"Worker {multiprocessing.current_process().name} finished episode {episode}")
if __name__ == "__main__":
    episodes = 100000
    num_processes = multiprocessing.cpu_count()
    episodes_per_worker = episodes // num_processes

    manager = multiprocessing.Manager()
    shared_q_table = manager.dict()
    lock = manager.Lock()

    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(episodes_per_worker, shared_q_table, lock))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # After training, evaluate the agent
    env = TicTacFoeEnv()
    agent_x = QLearningAgent(q_table=dict(shared_q_table))  # Convert back to a normal dict
    evaluate_agent(agent_x, env, episodes=100)

