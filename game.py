import numpy as np
import random

class TicTacFoeEnv:
    def __init__(self):
        self.board = np.full((5, 5), " ")
        self.replace_count = np.zeros((5, 5))  # Keeps track of how many times a position is replaced
        self.current_player = "X"
        self.turns = 0
        self.done = False

    def reset(self, random_start_moves=0):
        """Reset the environment to the initial state, optionally start with a random number of moves"""
        self.board = np.full((5, 5), " ")
        self.replace_count = np.zeros((5, 5))
        self.current_player = "X"
        self.turns = 0
        self.done = False

        # If random_start_moves > 0, initialize the board with random moves
        if random_start_moves > 0:
            self.random_start(random_start_moves)

        return self.get_state()

    def random_start(self, random_start_moves):
        """Randomly place a given number of marks for both players on the board, alternating turns,
        ensuring that no moves lead to an immediate win condition."""
        players = ["X", "O"]  # Player X always starts, then alternate with O

        # Initialize available positions
        available_positions = [(r, c) for r in range(5) for c in range(5)
                            if self.board[r][c] == " " or self.replace_count[r][c] < 2]

        # Loop through the number of moves to randomly place marks
        for move in range(random_start_moves):
            if not available_positions:
                break  # No more available positions to place a mark

            player = players[move % 2]  # Determine the current player (X or O)

            max_attempts = 100  # Limit the number of attempts to find a valid placement
            attempts = 0
            valid_move = False

            while attempts < max_attempts:
                # Randomly pick an available position
                position = random.choice(available_positions)
                row, col = position

                # Temporarily place the mark to check for winning state
                self.board[row][col] = player

                # Check if placing the mark results in a win for the current player
                if not self.check_winner(player):
                    # If not a win, finalize the placement
                    if self.board[row][col] == " ":
                        # If the position is empty, place the mark
                        self.board[row][col] = player
                    elif self.replace_count[row][col] < 2:
                        # If the position is not empty but can be replaced, replace the mark
                        self.board[row][col] = player
                        self.replace_count[row][col] += 1

                    valid_move = True
                    break  # Move was valid, exit the attempt loop
                else:
                    # Reset the position if it was a winning move
                    self.board[row][col] = " "  # Undo the move
                    # Remove this position from available_positions
                    available_positions.remove(position)

                attempts += 1  # Increment the attempts

            if not valid_move:
                print(f"Could not place a valid move after {max_attempts} attempts for player {player}.")
                # Optionally, you could add logic here to ensure that the game doesn't get stuck.

        # Update the number of turns to reflect the initial random moves
        self.turns = random_start_moves

        # Set the current player for the next turn (X starts if even moves, O if odd)
        self.current_player = "X" if random_start_moves % 2 == 0 else "O"

    def get_state(self):
        """Return the current state as a tuple (board state and current player)"""
        return (self.board.copy(), self.current_player)

    def step(self, action):
        """Execute an action and return the new state, reward, done, and info"""
        row, col = action
        if self.board[row][col] == " ":
            self.board[row][col] = self.current_player
            reward = 0  # No immediate reward for placing a mark
        elif self.replace_count[row][col] < 2:
            self.board[row][col] = self.current_player
            self.replace_count[row][col] += 1
            reward = -0.1  # Slight penalty for replacing opponent's mark
        else:
            raise ValueError(f"Invalid move at row {row}, col {col}")

        self.turns += 1
        reward, done = self.evaluate_game()

        if not done:
            self.switch_player()

        return self.get_state(), reward, done

    def can_win_next_turn(self, player):
        """Check if the player can win in the next turn"""
        for row in range(5):
            for col in range(5):
                if self.board[row][col] == " ":
                    self.board[row][col] = player
                    if self.is_winner(player):
                        self.board[row][col] = " "  # Undo the move
                        return True
                    self.board[row][col] = " "  # Undo the move
        return False

    def evaluate_game(self):
        """Evaluate if the game is over and assign rewards"""
        reward = 0

        # Check if the current player has won
        if self.check_winner(self.current_player):
            return (1 if self.current_player == "X" else -1), True

        # Check if the opponent has won (after the current move)
        opponent = "O" if self.current_player == "X" else "X"
        if self.check_winner(opponent):
            return (-1 if self.current_player == "X" else 1), True

        # Check for a draw
        if self.is_draw():
            return -0.5, True

        # Slight reward for having longer lines (bonus points for potential winning setups)
        reward += self.evaluate_lines(self.current_player)

        # Check if current player can win in the next turn (to give slight reward for strategic advantage)
        if self.can_win_next_turn(self.current_player):
            reward += 0.5  # Small reward for setting up a winning move

        return reward, False

    def check_winner(self, player):
        """Check if the given player has a winning line"""
        # Check rows and columns
        for i in range(5):
            if all(self.board[i][j] == player for j in range(5)):  # Row check
                return True
            if all(self.board[j][i] == player for j in range(5)):  # Column check
                return True

        # Check diagonals
        if all(self.board[i][i] == player for i in range(5)):  # Main diagonal
            return True
        if all(self.board[i][4 - i] == player for i in range(5)):  # Anti-diagonal
            return True

        return False

    def evaluate_lines(self, player):
        """Evaluate the board for potential line advantages (bonus for longer lines)"""
        reward = 0
        lines_to_check = []

        # Check rows and columns
        for i in range(5):
            lines_to_check.append(self.board[i])  # Rows
            lines_to_check.append(self.board[:, i])  # Columns

        # Check diagonals
        lines_to_check.append([self.board[i][i] for i in range(5)])  # Main diagonal
        lines_to_check.append([self.board[i][4 - i] for i in range(5)])  # Anti-diagonal

        # Reward player for having multiple consecutive marks
        for line in lines_to_check:
            count = np.count_nonzero(np.array(line) == player)
            if count == 4:
                reward += 0.45  # Almost winning
            elif count == 3:
                reward += 0.3  # Setting up
            elif count == 2:
                reward += 0.2  # Small advantage

        return reward

    def switch_player(self):
        """Switch between players X and O"""
        self.current_player = "O" if self.current_player == "X" else "X"

    def is_winner(self, player):
        """Check if the current player has won"""
        for row in self.board:
            if list(row).count(player) == 5:
                return True

        for col in range(5):
            if all(self.board[row][col] == player for row in range(5)):
                return True

        # Check diagonals
        if all(self.board[i][i] == player for i in range(5)):
            return True
        if all(self.board[i][4 - i] == player for i in range(5)):
            return True

        return False

    def is_draw(self):
        """Check if the game is a draw"""
        return np.all(self.board != " ") and np.all(self.replace_count == 2)

    def render(self):
        """Print the current game board"""
        for row in self.board:
            print("|".join(row))
            print("-" * 9)
