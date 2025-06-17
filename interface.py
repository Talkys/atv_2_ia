import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QTextEdit, QSpinBox, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from puzzle_solver import EightPuzzleGame, PuzzleState, FINAL_STATE_CONFIG
import puzzle_solver

class PuzzleBoardWidget(QWidget):
    """Widget to display the 8-puzzle board."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 300)
        self.cells = []
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(2)
        
        # Create 3x3 grid of cells
        for i in range(3):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(2)
            for j in range(3):
                cell = QLabel()
                cell.setAlignment(Qt.AlignCenter)
                cell.setStyleSheet("""
                    QLabel {
                        background-color: #f0f0f0;
                        border: 2px solid #333;
                        font-size: 24px;
                        font-weight: bold;
                    }
                """)
                cell.setFixedSize(90, 90)
                row_layout.addWidget(cell)
                self.cells.append(cell)
            self.layout.addLayout(row_layout)
    
    def update_board(self, board):
        """Update the display with a new board state."""
        for i in range(3):
            for j in range(3):
                value = board[i][j]
                cell = self.cells[i*3 + j]
                if value == 0:
                    cell.setText("")
                    cell.setStyleSheet("""
                        QLabel {
                            background-color: #d0d0d0;
                            color: #000000;
                            border: 2px solid #333;
                        }
                    """)
                else:
                    cell.setText(str(value))
                    cell.setStyleSheet("""
                        QLabel {
                            background-color: #f0f0f0;
                            color: #000000;
                            border: 2px solid #333;
                            font-size: 24px;
                            font-weight: bold;
                        }
                    """)


class EightPuzzleGUI(QMainWindow):
    """Main GUI window for the 8-puzzle solver."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("8-Puzzle Solver")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize puzzle game
        self.puzzle_game = None
        self.current_solution = None
        self.current_step = 0
        
        self.init_ui()
        self.reset_puzzle()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - puzzle display and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Puzzle board display
        self.puzzle_board = PuzzleBoardWidget()
        left_layout.addWidget(self.puzzle_board, alignment=Qt.AlignCenter)
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Breadth-First Search (BFS)",
            "Depth-First Search (DFS)",
            "Uniform Cost Search (UCS)",
            "Greedy Best-First (Manhattan)",
            "Greedy Best-First (Misplaced)",
            "Greedy Best-First (Linear Conflict)",
            "A* Search (Manhattan)",
            "A* Search (Misplaced)",
            "A* Search (Linear Conflict)"
        ])
        control_layout.addWidget(self.method_combo)
        
        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve_puzzle)
        control_layout.addWidget(self.solve_button)
        
        left_layout.addLayout(control_layout)
        
        # Step controls
        step_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("Previous Step")
        self.prev_button.clicked.connect(self.prev_step)
        step_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next Step")
        self.next_button.clicked.connect(self.next_step)
        step_layout.addWidget(self.next_button)
        
        left_layout.addLayout(step_layout)
        
        # Puzzle generation
        gen_layout = QHBoxLayout()
        
        self.random_button = QPushButton("Random Puzzle")
        self.random_button.clicked.connect(self.generate_random_puzzle)
        gen_layout.addWidget(self.random_button)
        
        self.reset_button = QPushButton("Reset Puzzle")
        self.reset_button.clicked.connect(self.reset_puzzle)
        gen_layout.addWidget(self.reset_button)
        
        left_layout.addLayout(gen_layout)
        
        # Right panel - solution info
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.solution_info = QTextEdit()
        self.solution_info.setReadOnly(True)
        self.solution_info.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                color: #000000;
                border: 1px solid #ccc;
                font-family: monospace;
            }
        """)
        right_layout.addWidget(self.solution_info)
        
        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Method", "Time (s)", "Nodes", "Cost"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.stats_table)
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)
        
        # Timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_step)
        
    def reset_puzzle(self):
        """Reset to the default initial state."""
        self.puzzle_game = EightPuzzleGame(PuzzleState([[5, 6, 4], [7, 8, 0], [2, 1, 3]]))
        self.update_display()
        self.clear_solution()
        
    def generate_random_puzzle(self):
        """Generate a new random puzzle."""
        self.puzzle_game = EightPuzzleGame()
        self.update_display()
        self.clear_solution()
        
    def update_display(self):
        """Update the board display with current state."""
        if self.puzzle_game:
            self.puzzle_board.update_board(self.puzzle_game.initial_state.board)
        
    def clear_solution(self):
        """Clear the current solution and reset step tracking."""
        self.current_solution = None
        self.current_step = 0
        self.solution_info.clear()
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        
    def solve_puzzle(self):
        """Solve the puzzle using the selected method."""
        if not self.puzzle_game:
            return
            
        method_map = {
            0: 'bfs',
            1: 'dfs',
            2: 'ucs',
            3: 'greedy_manhattan',
            4: 'greedy_misplaced',
            5: 'greedy_linear',
            6: 'astar_manhattan',
            7: 'astar_misplaced',
            8: 'astar_linear'
        }
        
        method_idx = self.method_combo.currentIndex()
        method_key = method_map.get(method_idx, 'astar_manhattan')
        
        try:

            #solution_info = self.puzzle_game.solve_with_method(method_key)
            solution_info = puzzle_solver.solve_game(self.puzzle_game, method_key, 5)

            if solution_info is None:
                QMessageBox.information(self, "No Solution", "This method timed out before finding a solution")
                return
            
            if solution_info['solution'] is None:
                QMessageBox.information(self, "No Solution", "No solution was found for this puzzle.")
                return
                
            self.current_solution = solution_info
            self.current_step = 0
            self.display_solution_info()
            self.update_solution_step()
            
            # Enable step buttons
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(len(self.current_solution['solution']) > 1)
            
            # Add to statistics table
            self.add_to_stats_table(solution_info)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while solving:\n{str(e)}")

    
            
    
    def display_solution_info(self):
        """Display the solution information in the text box."""
        if not self.current_solution:
            return
            
        info = self.current_solution
        text = f"Solution Method: {info['method']}\n"
        text += f"Path Cost: {info['path_cost']}\n"
        text += f"Nodes Expanded: {info['nodes_expanded']}\n"
        text += f"Time Taken: {info['time_taken']:.4f} seconds\n"
        text += f"\nSolution Path ({len(info['solution'])} steps):\n"
        
        for i, state in enumerate(info['solution']):
            if i < len(info['actions']):
                text += f"\nStep {i}: {info['actions'][i]}\n"
            else:
                text += f"\nStep {i}:\n"
            text += str(state) + "\n"
            
        self.solution_info.setPlainText(text)
    
    def update_solution_step(self):
        """Update the board display to show the current step in the solution."""
        if not self.current_solution or not self.current_solution['solution']:
            return
            
        if self.current_step < len(self.current_solution['solution']):
            state = self.current_solution['solution'][self.current_step]
            self.puzzle_board.update_board(state.board)
            
            # Highlight the moved tile (if not the first step)
            if self.current_step > 0 and self.current_step <= len(self.current_solution['actions']):
                prev_state = self.current_solution['solution'][self.current_step-1]
                # Find the position that changed
                for i in range(3):
                    for j in range(3):
                        if state.board[i][j] != prev_state.board[i][j] and state.board[i][j] != 0:
                            # Highlight this cell
                            cell = self.puzzle_board.cells[i*3 + j]
                            cell.setStyleSheet("""
                                QLabel {
                                    background-color: #a0e0a0;
                                    color: #000000;
                                    border: 2px solid #333;
                                    font-size: 24px;
                                    font-weight: bold;
                                }
                            """)
                            break
    
    def prev_step(self):
        """Go to the previous step in the solution."""
        if self.current_solution and self.current_step > 0:
            self.current_step -= 1
            self.update_solution_step()
            self.next_button.setEnabled(True)
            if self.current_step == 0:
                self.prev_button.setEnabled(False)
    
    def next_step(self):
        """Go to the next step in the solution."""
        if self.current_solution and self.current_step < len(self.current_solution['solution']) - 1:
            self.current_step += 1
            self.update_solution_step()
            self.prev_button.setEnabled(True)
            if self.current_step == len(self.current_solution['solution']) - 1:
                self.next_button.setEnabled(False)
    
    def add_to_stats_table(self, solution_info):
        """Add the solution statistics to the table."""
        row = self.stats_table.rowCount()
        self.stats_table.insertRow(row)
        
        items = [
            QTableWidgetItem(solution_info['method']),
            QTableWidgetItem(f"{solution_info['time_taken']:.4f}"),
            QTableWidgetItem(str(solution_info['nodes_expanded'])),
            QTableWidgetItem(str(solution_info['path_cost']))
        ]
        
        for i, item in enumerate(items):
            item.setTextAlignment(Qt.AlignCenter)
            self.stats_table.setItem(row, i, item)
        
        # Highlight the best (lowest cost) solution
        if row > 0:
            best_row = 0
            best_cost = float('inf')
            for r in range(row + 1):
                cost = int(self.stats_table.item(r, 3).text())
                if cost < best_cost:
                    best_cost = cost
                    best_row = r
            
            for c in range(4):
                if best_row == row:  # Current row is the best
                    self.stats_table.item(row, c).setBackground(QColor(220, 255, 220))
                    self.stats_table.item(row, c).setForeground(QColor(0, 0, 0))
                else:
                    # Reset previous best highlighting
                    self.stats_table.item(best_row, c).setBackground(QColor(220, 255, 220))
                    self.stats_table.item(best_row, c).setForeground(QColor(0, 0, 0))
                    # Reset current row if not best
                    if r == row:
                        self.stats_table.item(row, c).setBackground(QColor(255, 255, 255))
                        self.stats_table.item(row, c).setForeground(QColor(0, 0, 0))


def main():
    app = QApplication(sys.argv)
    window = EightPuzzleGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()