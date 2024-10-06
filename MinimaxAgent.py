from game import TicTacFoeEnv

class MinimaxAgent:
    def __init__(self, env, depth=3):
        """Initialize the Minimax agent with a given environment and depth for recursion."""
        self.env = env
        self.depth = depth  # Depth controls how deep the Minimax agent searches the game tree

    def get_best_move(self):
        """Get the best move using the Minimax algorithm."""
        board, current_player = self.env.get_state()  # Unpack state directly here
        best_move = None
        best_value = -float('inf') if current_player == "X" else float('inf')

        for move in self.get_available_moves(board):  # Pass the board only
            simulated_env = self.simulate_move(board, move, current_player)
            move_value = self.minimax(simulated_env, self.depth, -float('inf'), float('inf'), False)
            
            if current_player == "X" and move_value > best_value:
                best_value = move_value
                best_move = move
            elif current_player == "O" and move_value < best_value:
                best_value = move_value
                best_move = move

        return best_move

    def minimax(self, env, depth, alpha, beta, is_maximizing):
        """Recursive Minimax algorithm with alpha-beta pruning."""
        board, current_player = env.get_state()  # Unpack state here
        reward, done = env.evaluate_game()

        # If the game is over or we've reached the depth limit, return the evaluation
        if done or depth == 0:
            return reward

        if is_maximizing:  # Player X (Maximizer)
            max_eval = -float('inf')
            for move in self.get_available_moves(board):  # Pass the board only
                simulated_env = self.simulate_move(board, move, "X")
                eval = self.minimax(simulated_env, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:  # Player O (Minimizer)
            min_eval = float('inf')
            for move in self.get_available_moves(board):  # Pass the board only
                simulated_env = self.simulate_move(board, move, "O")
                eval = self.minimax(simulated_env, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_available_moves(self, board):
        """Return a list of available moves (positions) for the current board state."""
        available_moves = []
        for row in range(5):
            for col in range(5):
                if board[row][col] == " " or self.env.replace_count[row][col] < 2:
                    available_moves.append((row, col))
        return available_moves

    def simulate_move(self, board, move, player):
        """Simulate placing a move on the board for the given player, returning a new simulated environment."""
        new_env = TicTacFoeEnv()
        new_env.board = board.copy()  # Copy the board state
        new_env.current_player = player
        new_env.replace_count = self.env.replace_count.copy()  # Copy replace counts

        # Make the move in the simulated environment
        new_env.board[move[0]][move[1]] = player
        if self.env.replace_count[move[0]][move[1]] > 0:
            new_env.replace_count[move[0]][move[1]] += 1

        return new_env
