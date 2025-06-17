import random
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import time
import multiprocessing

FINAL_STATE_CONFIG = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
INITIAL_STATE_CONFIG = [[5, 6, 4], [7, 8, 0], [2, 1, 3]]

class PuzzleState:
    """Represents the state of the 8-puzzle board."""
    
    def __init__(self, board: List[List[int]]):
        """
        Initialize the puzzle state.
        
        Args:
            board: A 3x3 list of lists representing the puzzle state.
                   Use 0 to represent the empty space.
        """
        self.board = board
        self.size = len(board)
        self.empty_pos = self._find_empty_position()
        
    def _find_empty_position(self) -> Tuple[int, int]:
        """Find the position of the empty space (0)."""
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return (i, j)
        raise ValueError("Invalid board state - no empty space found")
    
    def __eq__(self, other: 'PuzzleState') -> bool:
        """Check if two states are equal."""
        return self.board == other.board
    
    def __hash__(self) -> int:
        """Make the state hashable for use in sets/dictionaries."""
        return hash(tuple(tuple(row) for row in self.board))
    
    def __str__(self) -> str:
        """String representation of the board."""
        return '\n'.join([' '.join(map(str, row)) for row in self.board])
    
    def is_goal(self, goal_state: 'PuzzleState') -> bool:
        """Check if this state matches the goal state."""
        return self == goal_state
    
    def copy(self) -> 'PuzzleState':
        """Create a deep copy of this state."""
        return PuzzleState([row.copy() for row in self.board])
    
    def get_possible_moves(self) -> List['PuzzleState']:
        """Generate all possible next states from the current state."""
        moves = []
        i, j = self.empty_pos
        
        # Possible movement directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for di, dj in directions:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < self.size and 0 <= new_j < self.size:
                new_board = self.copy()
                # Swap empty space with adjacent tile
                new_board.board[i][j], new_board.board[new_i][new_j] = \
                    new_board.board[new_i][new_j], new_board.board[i][j]
                new_board.empty_pos = (new_i, new_j)
                moves.append(new_board)
                
        return moves
    
    def manhattan_distance(self, goal_state: 'PuzzleState') -> int:
        """Calculate the Manhattan distance heuristic to the goal state."""
        distance = 0
        goal_positions = {}
        
        # Create a mapping of value to position in the goal state
        for i in range(self.size):
            for j in range(self.size):
                goal_positions[goal_state.board[i][j]] = (i, j)
        
        # Calculate Manhattan distance for each tile
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value != 0:  # Skip the empty space
                    goal_i, goal_j = goal_positions[value]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        
        return distance

    def linear_conflict(self, goal_state: 'PuzzleState') -> int:
        """Calculate the Manhattan distance with linear conflict heuristic."""
        distance = 0
        goal_positions = {}

        # Step 1: Build goal position lookup
        for i in range(self.size):
            for j in range(self.size):
                goal_positions[goal_state.board[i][j]] = (i, j)

        # Step 2: Compute Manhattan distance
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value != 0:
                    goal_i, goal_j = goal_positions[value]
                    distance += abs(i - goal_i) + abs(j - goal_j)

        # Step 3: Add linear conflicts for rows
        for row in range(self.size):
            max_col = [-1] * self.size  # Used to detect conflicts
            for col1 in range(self.size):
                tile1 = self.board[row][col1]
                if tile1 != 0:
                    goal_row1, goal_col1 = goal_positions[tile1]
                    if goal_row1 == row:  # Same row
                        for col2 in range(col1 + 1, self.size):
                            tile2 = self.board[row][col2]
                            if tile2 != 0:
                                goal_row2, goal_col2 = goal_positions[tile2]
                                if goal_row2 == row and goal_col1 > goal_col2:
                                    distance += 2

        # Step 4: Add linear conflicts for columns
        for col in range(self.size):
            for row1 in range(self.size):
                tile1 = self.board[row1][col]
                if tile1 != 0:
                    goal_row1, goal_col1 = goal_positions[tile1]
                    if goal_col1 == col:  # Same column
                        for row2 in range(row1 + 1, self.size):
                            tile2 = self.board[row2][col]
                            if tile2 != 0:
                                goal_row2, goal_col2 = goal_positions[tile2]
                                if goal_col2 == col and goal_row1 > goal_row2:
                                    distance += 2

        return distance

    
    def misplaced_tiles(self, goal_state: 'PuzzleState') -> int:
        """Count the number of misplaced tiles compared to the goal state."""
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != goal_state.board[i][j] and self.board[i][j] != 0:
                    count += 1
        return count


class SearchNode:
    """Represents a node in the search tree."""
    
    def __init__(self, state: PuzzleState, parent: Optional['SearchNode'] = None, 
                 action: Optional[str] = None, cost: int = 0):
        """
        Initialize a search node.
        
        Args:
            state: The puzzle state associated with this node.
            parent: The parent node in the search tree.
            action: The action taken to reach this node from the parent.
            cost: The total cost from the initial state to this node.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        
    def __lt__(self, other: 'SearchNode') -> bool:
        """Compare nodes based on cost (for priority queues)."""
        return self.cost < other.cost
    
    def path(self) -> List['SearchNode']:
        """Return the path from the initial state to this node."""
        node, path = self, []
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class SearchMethod(ABC):
    """Abstract base class for search methods."""
    
    @abstractmethod
    def solve(self, initial_state: PuzzleState, goal_state: PuzzleState) -> Optional[SearchNode]:
        """Solve the puzzle and return the solution node if found."""
        pass
    
    def get_search_name(self) -> str:
        """Return the name of the search method."""
        return self.__class__.__name__


class BreadthFirstSearch(SearchMethod):
    """Breadth-first search implementation."""
    
    def solve(self, initial_state: PuzzleState, goal_state: PuzzleState) -> Optional[SearchNode]:
        frontier = [SearchNode(initial_state)]
        explored = set()
        
        while frontier:
            node = frontier.pop(0)
            
            if node.state.is_goal(goal_state):
                return node
                
            explored.add(node.state)
            
            for move in node.state.get_possible_moves():
                if move not in explored and not any(n.state == move for n in frontier):
                    frontier.append(SearchNode(move, node, "move", node.cost + 1))
        
        return None


class DepthFirstSearch(SearchMethod):
    """Depth-first search implementation."""
    
    def solve(self, initial_state: PuzzleState, goal_state: PuzzleState) -> Optional[SearchNode]:
        # Store both the node and its current depth in the frontier
        frontier = [(SearchNode(initial_state), 0)]  # (node, depth)
        explored = set()
        depth_limit = 100
        
        while frontier:
            node, current_depth = frontier.pop()
            
            if node.state.is_goal(goal_state):
                return node
                
            explored.add(node.state)
            
            # Only expand if we haven't reached the depth limit
            if current_depth < depth_limit:
                for move in reversed(node.state.get_possible_moves()):
                    if move not in explored and not any(n.state == move for n, _ in frontier):
                        new_node = SearchNode(move, node, "move", node.cost + 1)
                        frontier.append((new_node, current_depth + 1))
        
        return None
    
class UniformCostSearch(SearchMethod):
    """Uniform Cost Search implementation (Dijkstra's algorithm)."""
    
    def solve(self, initial_state: PuzzleState, goal_state: PuzzleState) -> Optional[SearchNode]:
        # Priority queue where nodes are sorted by path cost
        frontier = [SearchNode(initial_state, cost=0)]
        explored = set()
        
        # To keep track of the best cost to reach each state
        cost_so_far = {initial_state: 0}
        
        while frontier:
            # Get node with lowest cost
            frontier.sort(key=lambda n: n.cost)
            node = frontier.pop(0)
            
            if node.state.is_goal(goal_state):
                return node
                
            explored.add(node.state)
            
            for move in node.state.get_possible_moves():
                new_cost = node.cost + 1  # Each move has cost 1
                
                if move not in explored and not any(n.state == move for n in frontier):
                    frontier.append(SearchNode(move, node, "move", new_cost))
                    cost_so_far[move] = new_cost
                elif any(n.state == move for n in frontier):
                    # Find the existing node in frontier
                    existing_node = next(n for n in frontier if n.state == move)
                    if new_cost < existing_node.cost:
                        # Update with better path
                        existing_node.parent = node
                        existing_node.cost = new_cost
                        existing_node.action = "move"
                        cost_so_far[move] = new_cost
        
        return None


class GreedyBestFirstSearch(SearchMethod):
    """Greedy Best-First Search implementation."""
    
    def __init__(self, heuristic: str = 'manhattan'):
        """
        Initialize Greedy Search with a specified heuristic.
        
        Args:
            heuristic: Either 'manhattan', linear_conflict or 'misplaced' for the heuristic function.
        """
        self.heuristic = heuristic
        
    def solve(self, initial_state: PuzzleState, goal_state: PuzzleState) -> Optional[SearchNode]:
        # Priority queue where nodes are sorted by heuristic value only
        frontier = [SearchNode(initial_state)]
        explored = set()
        
        while frontier:
            # Sort frontier by heuristic value (greedy choice)
            frontier.sort(key=lambda n: self._heuristic(n.state, goal_state))
            node = frontier.pop(0)
            
            if node.state.is_goal(goal_state):
                return node
                
            explored.add(node.state)
            
            for move in node.state.get_possible_moves():
                if move not in explored and not any(n.state == move for n in frontier):
                    frontier.append(SearchNode(move, node, "move", node.cost + 1))
        
        return None
    
    def _heuristic(self, state: PuzzleState, goal_state: PuzzleState) -> int:
        """Calculate heuristic value based on the selected method."""
        if self.heuristic == 'manhattan':
            return state.manhattan_distance(goal_state)
        elif self.heuristic == 'misplaced':
            return state.misplaced_tiles(goal_state)
        elif self.heuristic == 'linear_conflict':
            return state.linear_conflict(goal_state)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")
    
    def get_search_name(self) -> str:
        """Return the name of the search method with heuristic."""
        return f"{super().get_search_name()} ({self.heuristic} heuristic)"


class AStarSearch(SearchMethod):
    """A* search implementation."""
    
    def __init__(self, heuristic: str = 'manhattan'):
        """
        Initialize A* search with a specified heuristic.
        
        Args:
            heuristic: Either 'manhattan', linear_conflict or 'misplaced' for the heuristic function.
        """
        self.heuristic = heuristic
        
    def solve(self, initial_state: PuzzleState, goal_state: PuzzleState) -> Optional[SearchNode]:
        open_set = [SearchNode(initial_state)]
        explored = set()
        
        # For node scoring
        g_score = {initial_state: 0}
        f_score = {initial_state: self._heuristic(initial_state, goal_state)}
        
        while open_set:
            # Get node with lowest f_score
            open_set.sort(key=lambda n: f_score[n.state])
            node = open_set.pop(0)
            
            if node.state.is_goal(goal_state):
                return node
                
            explored.add(node.state)
            #print(node.state.board)
            
            for move in node.state.get_possible_moves():
                if move in explored:
                    continue
                    
                tentative_g_score = g_score[node.state] + 1
                
                if not any(n.state == move for n in open_set):
                    new_node = SearchNode(move, node, "move", tentative_g_score)
                    open_set.append(new_node)
                elif tentative_g_score >= g_score.get(move, float('inf')):
                    continue
                    
                # This path is the best so far
                g_score[move] = tentative_g_score
                f_score[move] = g_score[move] + self._heuristic(move, goal_state)
        
        return None
    
    def _heuristic(self, state: PuzzleState, goal_state: PuzzleState) -> int:
        """Calculate heuristic value based on the selected method."""
        if self.heuristic == 'manhattan':
            return state.manhattan_distance(goal_state)
        elif self.heuristic == 'misplaced':
            return state.misplaced_tiles(goal_state)
        elif self.heuristic == 'linear_conflict':
            return state.linear_conflict(goal_state)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")
    
    def get_search_name(self) -> str:
        """Return the name of the search method with heuristic."""
        return f"{super().get_search_name()} ({self.heuristic} heuristic)"


class EightPuzzleGame:
    """The 8-puzzle game with different search methods."""
    
    def __init__(self, initial_state: Optional[PuzzleState] = None):
        """
        Initialize the 8-puzzle game.
        
        Args:
            initial_state: Optional initial state. If None, a random solvable state is generated.
        """
        self.goal_state = PuzzleState(FINAL_STATE_CONFIG)
        
        if initial_state:
            self.initial_state = initial_state
        else:
            self.initial_state = self._generate_random_solvable_state()
        
        self.search_methods = {
            'bfs': BreadthFirstSearch(),
            'dfs': DepthFirstSearch(),
            'ucs': UniformCostSearch(),
            'greedy_manhattan': GreedyBestFirstSearch('manhattan'),
            'greedy_misplaced': GreedyBestFirstSearch('misplaced'),
            'greedy_linear': GreedyBestFirstSearch('linear_conflict'),
            'astar_manhattan': AStarSearch('manhattan'),
            'astar_misplaced': AStarSearch('misplaced'),
            'astar_linear':    AStarSearch('linear_conflict')
        }
    
    def _generate_random_solvable_state(self) -> PuzzleState:
        """Generate a random solvable initial state."""
        while True:
            numbers = list(range(9))
            random.shuffle(numbers)
            board = [numbers[i*3:(i+1)*3] for i in range(3)]
            state = PuzzleState(board)
            if self._is_solvable(state):
                return state
    
    def _is_solvable(self, state: PuzzleState) -> bool:
        """Check if a given state is solvable."""
        # Flatten the board and count inversions
        flat_board = [num for row in state.board for num in row if num != 0]
        inversions = 0
        n = len(flat_board)
        
        for i in range(n):
            for j in range(i + 1, n):
                if flat_board[i] > flat_board[j]:
                    inversions += 1
        
        # For a 3x3 puzzle, it's solvable if the number of inversions is even
        return inversions % 2 == 0
    
    def add_search_method(self, name: str, method: SearchMethod):
        """Add a custom search method to the game."""
        self.search_methods[name] = method
    
    def solve_with_method(self, method_name: str) -> Dict[str, Any]:
        """
        Solve the puzzle using a specified search method.
        
        Args:
            method_name: Name of the search method to use.
            
        Returns:
            A dictionary containing solution information:
            - 'method': Name of the search method
            - 'solution': List of states from initial to goal
            - 'actions': List of actions taken
            - 'nodes_expanded': Number of nodes expanded
            - 'time_taken': Time taken to solve in seconds
            - 'path_cost': Cost of the solution path
        """
        if method_name not in self.search_methods:
            raise ValueError(f"Unknown search method: {method_name}")
        
        method = self.search_methods[method_name]
        start_time = time.time()
        solution_node = method.solve(self.initial_state, self.goal_state)
        time_taken = time.time() - start_time
        
        if solution_node:
            path = solution_node.path()
            solution = [node.state for node in path]
            actions = [node.action for node in path[1:]]  # First node has no action
            
            # Estimate nodes expanded (this is a simplification)
            # In a proper implementation, we'd track this during the search
            nodes_expanded = len(solution) * 4  # Approximation
            
            return {
                'method': method.get_search_name(),
                'solution': solution,
                'actions': actions,
                'nodes_expanded': nodes_expanded,
                'time_taken': time_taken,
                'path_cost': solution_node.cost
            }
        else:
            return {
                'method': method.get_search_name(),
                'solution': None,
                'message': 'No solution found'
            }
    
    def print_solution(self, solution_info: Dict[str, Any], print_steps: bool=False):
        """Print the solution information in a readable format."""
        print(f"\nSolution using {solution_info['method']}:")
        
        if solution_info['solution'] is None:
            print("No solution found!")
            return
        
        print(f"Path cost: {solution_info['path_cost']}")
        print(f"Nodes expanded: {solution_info['nodes_expanded']}")
        print(f"Time taken: {solution_info['time_taken']:.4f} seconds")
        


        if print_steps:
            print("\nSolution path:")
            for i, state in enumerate(solution_info['solution']):
                print(f"\nStep {i}:")
                print(state)
                if i < len(solution_info['actions']):
                    print(f"Action: {solution_info['actions'][i]}")

def solve_with_timeout(game, method, timeout):
    # Create a queue to communicate the result
    result_queue = multiprocessing.Queue()
    
    # Define the worker function
    def worker():
        solution_info = game.solve_with_method(method)
        result_queue.put(solution_info)
    
    # Start the process
    p = multiprocessing.Process(target=worker)
    p.start()
    
    # Wait for timeout or completion
    p.join(timeout=timeout)
    
    # If process is still alive after timeout, terminate it
    if p.is_alive():
        p.terminate()
        p.join()
        return None  # or some timeout indicator
    else:
        return result_queue.get()
    

def solve_game(game, method, timeout):
    timeout_seconds = 5
    start_time = time.time()
    solution_info = solve_with_timeout(game, method, timeout)

    if solution_info is None: return None
    
    #game.print_solution(solution_info)
    #elapsed = time.time() - start_time
    #print(f"Method {method} completed in {elapsed:.2f} seconds")

    return solution_info


# Example usage
if __name__ == "__main__":
    # Create a game with a specific initial state
    initial_board = INITIAL_STATE_CONFIG
    game = EightPuzzleGame(PuzzleState(initial_board))
    
    # Or create a game with a random initial state
    # game = EightPuzzleGame()
    
    print("Initial state:")
    print(game.initial_state)
    
    print("\nGoal state:")
    print(game.goal_state)
    
    # Solve using different methods
    methods = ['bfs', 'dfs', 'ucs', 'greedy_manhattan', 'greedy_misplaced', 'greedy_linear', 'astar_manhattan', 'astar_misplaced', 'astar_linear']
    #methods = ['astar_manhattan', 'astar_misplaced', 'astar_linear']
    
    timeout_seconds = 5
    for method in methods:
        start_time = time.time()
        solution_info = solve_with_timeout(game, method, timeout_seconds)
        
        if solution_info is None:
            print(f"\nMethod {method} timed out after {timeout_seconds} seconds")
            continue
        
        game.print_solution(solution_info)
        elapsed = time.time() - start_time
        print(f"Method {method} completed in {elapsed:.2f} seconds")