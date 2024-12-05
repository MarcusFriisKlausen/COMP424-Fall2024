# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import copy
import time
import helpers as h
import psutil
@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def step(self, chess_board, player, opponent):
      """
      Implement the step function of your agent here.
      You can use the following variables to access the chess board:
      - chess_board: a numpy array of shape (board_size, board_size)
        where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
        and 2 represents Player 2's discs (Brown).
      - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
      - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

      You should return a tuple (r,c), where (r,c) is the position where your agent
      wants to place the next disc. Use functions in helpers to determine valid moves
      and more helpful tools.

      Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
      """
      start_time = time.time()

      is_max_player = True if player == 2 else False

      evaluation_best_move = self.iterative_deepening_minimax(chess_board, player, opponent, is_max_player, start_time, 1.9)

      _, best_move = evaluation_best_move

      time_taken = time.time() - start_time
      print("My AI's turn took ", time_taken, "seconds.")

      process = psutil.Process()
      mem_info = process.memory_info()
      # Uncomment the next line to print memory usage
      print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")
      
      return best_move
  
  def iterative_deepening_minimax(self, chess_board, player, opponent, max_player, start_time, max_time_seconds):
    """Iterative Deepening Minimax with Alpha-Beta Pruning and Time Constraint.
    
    This method uses iterative deepening to search progressively deeper levels 
    of the game tree while respecting a time limit. It combines minimax 
    with alpha-beta pruning to find the best move for the current player.
    
    Parameters
    ----------
    chess_board : numpy.ndarray of shape (board_size, board_size)
        The chess board with 0 representing an empty space, 1 for black (Player 1),
        and 2 for white (Player 2).
    player : int
        The current player (1 for black, 2 for white).
    opponent : int
        The opponent player (1 for black, 2 for white).
    max_player : bool
        Boolean value indicating if the current player is the maximizing player.
    start_time : float
        The time (in seconds) when the search started, obtained from time.time().
    max_time_seconds : float
        The maximum time (in seconds) allowed for the search.

    Returns
    -------
    eval : float
        The evaluation score of the best move found within the time limit.
    best_move : (int, int) or None
        The coordinates of the best move for the player, or None if no valid move is found.
    """
    
    best_move = None
    m = chess_board.shape[0]
    depth = 1

    while True:
        current_time = time.time()
        if current_time - start_time > max_time_seconds:
            break

        try:
            eval, move = self.alpha_beta_minimax(
                chess_board,
                player,
                opponent,
                depth,
                -float('inf'),
                float('inf'),
                max_player,
                start_time,
                max_time_seconds
            )
            best_move = move
        except TimeoutError:
            break

        depth += 1
    print(eval, " ", best_move)
    return eval, best_move


  def alpha_beta_minimax(self, chess_board, player, opponent, depth, alpha, beta, max_player, start_time, max_time_seconds):
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
          Best evaluation for the player and the move correlating to the evaluation.
      """
      if time.time() - start_time > max_time_seconds:
        raise TimeoutError("Time limit exceeded")
      
      valid_moves = h.get_valid_moves(chess_board, player)
      valid_moves = self.prioritize_moves(chess_board, valid_moves)

      max_time = (time.time() - start_time) > 1.97
      if depth == 0 or len(valid_moves) == 0 or max_time:
          return self.evaluation(chess_board), None
      
      best_move = None
      if max_player:
          best_evaluation = -float('inf')
          for move in valid_moves:
              board_copy = copy.deepcopy(chess_board)
              h.execute_move(board_copy, move, player)
              evaluation, _ = self.alpha_beta_minimax(board_copy, player, opponent, depth - 1, alpha, beta, False, start_time, max_time_seconds)
              if evaluation > best_evaluation:
                  best_evaluation = evaluation
                  best_move = move
              alpha = max(alpha, evaluation)
              if beta <= alpha:
                  break
          return best_evaluation, best_move
          
      else:
          best_evaluation = float('inf')
          for move in valid_moves:
              board_copy = copy.deepcopy(chess_board)
              h.execute_move(board_copy, move, player)
              evaluation, _ = self.alpha_beta_minimax(board_copy, player, opponent, depth - 1, alpha, beta, True, start_time, max_time_seconds)
              if evaluation < best_evaluation:
                  best_evaluation = evaluation
                  best_move = move
              beta = min(beta, evaluation)
              if beta <= alpha:
                  break
          return best_evaluation, best_move
  
  def evaluation(self, chess_board):
    """
    Evaluation function for the board state.
    The evaluates the given board state based on the number of pieces 
    owned by the player 1 and player 2, and assigns higher weights to critical positions.
    Also evaluates a weighted difference of available moves between the players.
    
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
    evaluation : float
        The evaluation score of the current board for the player. A positive 
        value indicates a favorable position for the player, while a negative 
        value indicates a favorable position for the opponent.
    """
    m = chess_board.shape[0]
    
    corner_weight = 100
    adjacent_to_corner_penalty = -25
    edge_weight = 10
    inner_weight = 5

    weight_matrix = np.zeros((m, m))
    
    corners = [(0, 0), (0, m-1), (m-1, 0), (m-1, m-1)]
    
    adj_to_corners = [
        (0, 1), (1, 0), (1, 1),            # Top-left corner
        (0, m-2), (1, m-1), (1, m-2),      # Top-right corner
        (m-1, 1), (m-2, 0), (m-2, 1),      # Bottom-left corner
        (m-2, m-1), (m-1, m-2), (m-2, m-2) # Bottom-right corner
    ]
    
    for corner in corners:
        weight_matrix[corner] = corner_weight

    for adj in adj_to_corners:
        weight_matrix[adj] = adjacent_to_corner_penalty

    for i in range(1, m-1):
        weight_matrix[0, i] = edge_weight   # Top edge
        weight_matrix[m-1, i] = edge_weight # Bottom edge
        weight_matrix[i, 0] = edge_weight   # Left edge
        weight_matrix[i, m-1] = edge_weight # Right edge

    for i in range(1, m-1):
        for j in range(1, m-1):
            if weight_matrix[i, j] == 0:
                weight_matrix[i, j] = inner_weight

    player_score = np.sum(weight_matrix[chess_board == 2])
    opponent_score = np.sum(weight_matrix[chess_board == 1])
    
    mobility = 5 * (len(h.get_valid_moves(chess_board, 2)) - len(h.get_valid_moves(chess_board, 1)))

    stability = None
    
    pieces_placed = np.sum(chess_board != 0)

    if (m * m) / 2 > player_score + opponent_score:
        player_score = 0.2 * player_score
        opponent_score = 0.2 * opponent_score
    elif (2 * (m * m)) / 3 > player_score + opponent_score:
        player_stability = self.stable_discs(chess_board, 2)
        opponent_stability = self.stable_discs(chess_board, 1)
        stability = player_stability - opponent_stability
        player_score = 0.7 * player_score
        opponent_score = 0.7 * opponent_score
    elif (m * m) - 6 < player_score + opponent_score:
        player_score = 1.5 * player_score
        opponent_score = 1.5 * opponent_score

    if stability:
        return stability + mobility + (player_score - opponent_score)
    else:
        return mobility + (player_score - opponent_score)
  
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

      empty_spaces = np.sum(chess_board == 0)
      if empty_spaces > 40:
          depth = 50
      elif empty_spaces > 20:
          depth = 40
      else:
          depth = 20
      return empty_spaces
  
  def prioritize_moves(self, chess_board, valid_moves):
    """
    Sorts move in order of highest strategic importance.
    
    Moves are prioritized in the following order:
    1. Corner moves
    2. Edge moves
    3. Center moves
    4. Moves adjacent to corners
    
    Parameters
    ----------
    chess_board : numpy.ndarray of shape (board_size, board_size)
        The current chess board.
    valid_moves : list of tuples
        List of valid moves (row, col) for the current player.

    Returns
    -------
    sorted_moves : list of tuples
        List of valid moves sorted by priority.
    """
    m = chess_board.shape[0]
    
    corners = {(0, 0), (0, m-1), (m-1, 0), (m-1, m-1)}
    
    adj_to_corners = {
        (0, 1), (1, 0), (1, 1),           # Top-left corner
        (0, m-2), (1, m-1), (1, m-2),     # Top-right corner
        (m-1, 1), (m-2, 0), (m-2, 1),     # Bottom-left corner
        (m-2, m-1), (m-1, m-2), (m-2, m-2)  # Bottom-right corner
    }
    
    corner_moves = []
    edge_moves = []
    center_moves = []
    adjacent_corner_moves = []
    
    for move in valid_moves:
        if move in corners:
            corner_moves.append(move)
        elif move in adj_to_corners:
            adjacent_corner_moves.append(move)
        elif move[0] == 0 or move[0] == m-1 or move[1] == 0 or move[1] == m-1:
            edge_moves.append(move)
        else:
            center_moves.append(move)
    
    sorted_moves = corner_moves + edge_moves + center_moves + adjacent_corner_moves
    return sorted_moves

  def stable_discs(self, chess_board, player):
    """
    Calculates the stability score for the given player on the board.
    
    Stable discs are discs that can't be flipped for the rest of the game.
    
    Parameters
    ----------
    chess_board : numpy.ndarray of shape (board_size, board_size)
        Current board state.
    player : int
        The current player's ID.
    opponent : int
        The opponent player's ID.

    Returns
    -------
    stability_score : int
        The stability score for the given player.
    """
    m = chess_board.shape[0]
    stability_score = 0
    
    def is_stable(x, y):
        """
        Determines if a piece at (x, y) is stable.
        A piece is stable if it cannot be flipped under any circumstances.
        """
        if chess_board[x, y] != player:
            return False
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < m and 0 <= ny < m:
                if chess_board[nx, ny] != player:
                    break
                nx, ny = nx + dx, ny + dy
            else:
                return True
        return False

    for x in range(m):
        for y in range(m):
            if chess_board[x, y] == player and is_stable(x, y):
                stability_score += 1

    return stability_score
  