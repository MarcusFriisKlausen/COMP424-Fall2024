import helpers as h

class Agent:
    def __init__(self):
        """
        Initialize the agent, add a name which is used to register the agent
        """
        self.name = "DummyAgent"
        # Flag to indicate whether the agent can be used to autoplay
        self.autoplay = True

    def __str__(self) -> str:
        return self.name

    def step(self, chess_board, player, opponent):
        """
        Main decision logic of the agent, which is called by the simulator.
        Extend this method to implement your own agent to play the game.

        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        player : int
            The current player (1 for black, 2 for white).
        opponent : int
            The opponent player (1 for black, 2 for white).

        Returns
        -------
        move_pos : tuple of int
            The position (x, y) where the player places the disc.
        """
        valid_moves = h.get_valid_moves(chess_board, player)

        pass

    def alpha_beta_minimax(self, chess_board, player, opponent, depth, alpha, beta, max_player):
        """
        Minimax algorithm making use of alpha-beta-pruning. Used for finding the best evaluated move for the current player.
        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        player : int
            The current player (1 for black, 2 for white).
        opponent : int
            The opponent player (1 for black, 2 for white).
        depth : int
            The max depth of the search.
        alpha : float
            The lower bound for alpha parameter.
        beta : float
            The upper bound for beta parameter.
        max_player : bool
            Boolean value for if player parameter is the maximizing player.
        Returns
        -------
        evaluation_best_move : (int, (int, int))
            Best evaluation for the player and the move correlating to the evalutaion.
        """
        pass

    def evaluation(self, chess_board, player, opponent):
        """
        Evaluation function for board state.
        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        player : int
            The current player (1 for black, 2 for white).
        opponent : int
            The opponent player (1 for black, 2 for white).
        Returns
        -------
        evaluation : int
            The evaluation of the current board for the player.
        """
        pass
    
    def tree_depth(self, chess_board):
        """
        Function for deciding how far tree depth should be for evaluation based on amount of pieces currently on the board.
        Parameters
        ----------
        chess_board : numpy.ndarray of shape (board_size, board_size)
            The chess board with 0 representing an empty space, 1 for black (Player 1),
            and 2 for white (Player 2).
        Returns
        -------
        depth : int
            Depth for next evaluation.
        """
        pass