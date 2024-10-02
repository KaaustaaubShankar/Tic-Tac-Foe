import numpy as np

class TicTacFoeEnv:
    def __init__(self):
        self.board = np.full((5, 5), " ")
        self.replace_count = np.zeros((5, 5))  # Keeps track of how many times a position is replaced
        self.current_player = "X"
        self.turns = 0
        self.done = False

    def reset(self):
        """Reset the environment to the initial state"""
        self.board = np.full((5, 5), " ")
        self.replace_count = np.zeros((5, 5))
        self.current_player = "X"
        self.turns = 0
        self.done = False
        return self.get_state()

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
        if self.is_winner(self.current_player):
            return 1 if self.current_player == "X" else -1, True
        if self.is_draw():
            return -0.5, True

        # Check for longer lines and give a slight reward
        reward = 0
        for row in range(5):
            for col in range(5):
                if self.board[row][col] == self.current_player:
                    # Check horizontal line
                    if col <= 1 and all(self.board[row][col + i] == self.current_player for i in range(5 - col)):
                        reward += 0.1
                    # Check vertical line
                    if row <= 1 and all(self.board[row + i][col] == self.current_player for i in range(5 - row)):
                        reward += 0.1
                    # Check diagonal lines
                    if row <= 1 and col <= 1:
                        if all(self.board[row + i][col + i] == self.current_player for i in range(5 - max(row, col))):
                            reward += 0.1
                        if all(self.board[row + i][col - i] == self.current_player for i in range(5 - max(row, 4 - col))):
                            reward += 0.1
        
        opponent = "O" if self.current_player == "X" else "X"

        if self.is_winner(opponent):
            return -1, True

        # Encourage making a winning move
        if self.can_win_next_turn(self.current_player):
            reward += 0.5  # Slight reward for a winning opportunity

        return reward, False


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
        return self.turns == 25  # All positions filled

    def render(self):
        """Print the current game board"""
        for row in self.board:
            print("|".join(row))
            print("-" * 9)
