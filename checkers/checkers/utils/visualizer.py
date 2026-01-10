"""
Real-time training visualization with pygame.
"""

import pygame
import numpy as np
import csv
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict
from collections import deque

from .visualizer_types import MetricsUpdate, GameStateUpdate, EvaluationUpdate, StatusUpdate
from .live_plots import LineChart
from .board_renderer import BoardRenderer


class RealTimeVisualizer:
    """
    Real-time training visualizer using pygame.
    
    Runs in a separate thread and displays:
    - Live metric charts (loss, win rate)
    - Current game board state
    - Training statistics and progress
    """
    
    def __init__(
        self,
        window_width: int = 1400,
        window_height: int = 900,
        log_file: Optional[str] = None,
        max_history: int = 100
    ):
        """
        Initialize visualizer.
        
        Args:
            window_width: Window width in pixels
            window_height: Window height in pixels
            log_file: Path to CSV log file to load historical data
            max_history: Maximum iterations to show in charts
        """
        self.window_width = window_width
        self.window_height = window_height
        self.log_file = log_file
        self.max_history = max_history
        
        # Threading
        self.update_queue = Queue()
        self.running = False
        self.thread = None
        
        # Pygame initialized flag
        self.initialized = False
        
        # State
        self.current_iteration = 0
        self.current_game_state: Optional[np.ndarray] = None
        self.current_policy: Optional[np.ndarray] = None
        self.current_status = "Initializing..."
        self.current_phase = ""
        
        # Statistics
        self.total_iterations = 0
        self.start_time = time.time()
        self.last_eval_win_rate = 0.0
        self.best_win_rate = 0.0
        self.buffer_size = 0
        
        # Recent messages
        self.messages = deque(maxlen=5)
        
    def start(self):
        """
        Initialize pygame (must be called from main thread).
        Does NOT start a separate thread - visualizer runs via update() calls.
        """
        if self.initialized:
            return
        
        # Initialize pygame
        pygame.init()
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("AlphaZero Checkers Training - Live Monitor")
        
        self.clock = pygame.time.Clock()
        
        # Initialize fonts
        try:
            self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
            self.stat_font = pygame.font.SysFont('Arial', 16)
            self.small_font = pygame.font.SysFont('Arial', 12)
        except:
            self.title_font = pygame.font.Font(None, 32)
            self.stat_font = pygame.font.Font(None, 20)
            self.small_font = pygame.font.Font(None, 16)
        
        # Initialize components
        self._init_components()
        
        # Load historical data if available
        if self.log_file:
            self._load_historical_data()
        
        self.initialized = True
        self.running = True
    
    
    def stop(self):
        """Stop the visualizer."""
        self.running = False
        if self.initialized:
            pygame.quit()
            self.initialized = False
    
    def update(self, data):
        """
        Queue an update to the visualizer.
        
        Args:
            data: Update object (MetricsUpdate, GameStateUpdate, etc.)
        """
        self.update_queue.put(data)
    
    def poll_events(self):
        """
        Poll for pygame events and process updates.
        Must be called regularly from main thread (e.g., in a loop).
        Returns False if user wants to quit.
        """
        if not self.initialized:
            return True
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_s:
                    self._save_screenshot()
        
        # Process updates from queue
        self._process_updates()
        
        # Render
        self._render()
        
        # Cap at 30 FPS - REMOVED to avoid slowing down training loop
        # self.clock.tick(30)
        
        return True
    
    def _init_components(self):
        """Initialize visualization components."""
        margin = 20
        
        # Loss chart (top left)
        loss_chart_width = 600
        loss_chart_height = 300
        self.loss_chart = LineChart(
            rect=pygame.Rect(margin, 80, loss_chart_width, loss_chart_height),
            title="Training Loss",
            max_points=self.max_history,
            y_label="Loss",
            colors={
                "Total": (66, 135, 245),
                "Policy": (245, 66, 66),
                "Value": (66, 245, 135)
            }
        )
        
        # Win rate chart (bottom left)
        win_chart_y = 80 + loss_chart_height + margin
        win_chart_height = self.window_height - win_chart_y - margin
        self.win_chart = LineChart(
            rect=pygame.Rect(margin, win_chart_y, loss_chart_width, win_chart_height),
            title="Evaluation Win Rate",
            max_points=self.max_history // 10,  # Eval is every 10 iterations
            y_label="Win Rate",
            colors={"Win Rate": (245, 135, 66)}
        )
        
        # Board renderer (top right)
        board_x = margin + loss_chart_width + margin
        board_y = 80
        self.board_renderer = BoardRenderer(square_size=45)
        self.board_offset = (board_x + 30, board_y + 30)
        
        # Stats panel coordinates
        self.stats_x = board_x
        self.stats_y = board_y + self.board_renderer.total_size + 30
    
    def _load_historical_data(self):
        """Load historical training data from CSV log."""
        log_path = Path(self.log_file)
        if not log_path.exists():
            return
        
        try:
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    iteration = int(row['iteration'])
                    
                    # Add loss data
                    self.loss_chart.add_data_point("Total", iteration, float(row['total_loss']))
                    self.loss_chart.add_data_point("Policy", iteration, float(row['policy_loss']))
                    self.loss_chart.add_data_point("Value", iteration, float(row['value_loss']))
                    
                    # Add win rate data (only when evaluated)
                    win_rate = float(row['eval_win_rate'])
                    if win_rate >= 0:
                        self.win_chart.add_data_point("Win Rate", iteration, win_rate)
                        self.last_eval_win_rate = win_rate
                        if win_rate > self.best_win_rate:
                            self.best_win_rate = win_rate
                    
                    self.current_iteration = iteration
                    self.buffer_size = int(row['buffer_size'])
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
    
    def _process_updates(self):
        """Process all pending updates from the queue."""
        try:
            while True:
                update = self.update_queue.get_nowait()
                self._handle_update(update)
        except Empty:
            pass
    
    def _handle_update(self, update):
        """Handle a single update."""
        if isinstance(update, MetricsUpdate):
            # Update charts
            self.loss_chart.add_data_point("Total", update.iteration, update.total_loss)
            self.loss_chart.add_data_point("Policy", update.iteration, update.policy_loss)
            self.loss_chart.add_data_point("Value", update.iteration, update.value_loss)
            
            self.current_iteration = update.iteration
            self.buffer_size = update.buffer_size
            
        elif isinstance(update, GameStateUpdate):
            # Update current game state
            self.current_game_state = update.game_array
            self.current_policy = update.policy
            
        elif isinstance(update, EvaluationUpdate):
            # Update win rate chart
            self.win_chart.add_data_point("Win Rate", update.iteration, update.win_rate)
            self.last_eval_win_rate = update.win_rate
            
            if update.win_rate > self.best_win_rate:
                self.best_win_rate = update.win_rate
            
            if update.is_best:
                self.messages.append(f"⭐ New best model! Win rate: {update.win_rate:.1%}")
            
        elif isinstance(update, StatusUpdate):
            self.current_status = update.message
            self.current_phase = update.phase
            if update.iteration > 0:
                self.current_iteration = update.iteration
    
    def _render(self):
        """Render all visualization components."""
        # Background
        self.screen.fill((245, 245, 245))
        
        # Title bar
        self._render_title()
        
        # Charts
        self.loss_chart.render(self.screen)
        self.win_chart.render(self.screen)
        
        # Board
        self._render_board()
        
        # Stats panel
        self._render_stats()
        
        # Messages
        self._render_messages()
        
        # Update display
        pygame.display.flip()
    
    def _render_title(self):
        """Render title bar."""
        title_bg = pygame.Rect(0, 0, self.window_width, 70)
        pygame.draw.rect(self.screen, (50, 50, 70), title_bg)
        
        title_text = self.title_font.render(
            "🎮 AlphaZero Checkers Training - Live Monitor",
            True,
            (255, 255, 255)
        )
        self.screen.blit(title_text, (20, 15))
        
        # Status
        status_text = self.stat_font.render(
            f"Status: {self.current_status}",
            True,
            (200, 200, 200)
        )
        self.screen.blit(status_text, (20, 45))
        
        # Iteration in top right
        iter_text = self.stat_font.render(
            f"Iteration: {self.current_iteration}",
            True,
            (255, 255, 255)
        )
        self.screen.blit(iter_text, (self.window_width - 200, 25))
    
    def _render_board(self):
        """Render the current game board."""
        # Board background
        board_bg = pygame.Rect(
            self.board_offset[0] - 30,
            self.board_offset[1] - 30,
            self.board_renderer.total_size + 60,
            self.board_renderer.total_size + 90
        )
        pygame.draw.rect(self.screen, (255, 255, 255), board_bg)
        pygame.draw.rect(self.screen, (150, 150, 150), board_bg, 2)
        
        # Title
        board_title = self.stat_font.render("Live Game State", True, (0, 0, 0))
        self.screen.blit(board_title, (self.board_offset[0], self.board_offset[1] - 22))
        
        # Render board
        self.board_renderer.render_board(self.screen, offset=self.board_offset)
        self.board_renderer.render_coordinates(self.screen, offset=self.board_offset)
        
        # Render pieces if we have a game state
        if self.current_game_state is not None:
            self.board_renderer.render_pieces(
                self.screen,
                self.current_game_state,
                offset=self.board_offset
            )
            
            # Render policy heatmap if available
            if self.current_policy is not None:
                self.board_renderer.render_heatmap(
                    self.screen,
                    self.current_policy,
                    offset=self.board_offset,
                    alpha=80
                )
        else:
            # No game state yet
            no_game_text = self.small_font.render(
                "Waiting for game data...",
                True,
                (150, 150, 150)
            )
            center_x = self.board_offset[0] + self.board_renderer.total_size // 2
            center_y = self.board_offset[1] + self.board_renderer.total_size // 2
            text_rect = no_game_text.get_rect(center=(center_x, center_y))
            self.screen.blit(no_game_text, text_rect)
    
    def _render_stats(self):
        """Render statistics panel."""
        stats_bg = pygame.Rect(self.stats_x, self.stats_y, 700, 220)
        pygame.draw.rect(self.screen, (255, 255, 255), stats_bg)
        pygame.draw.rect(self.screen, (150, 150, 150), stats_bg, 2)
        
        # Title
        stats_title = self.stat_font.render("Training Statistics", True, (0, 0, 0))
        self.screen.blit(stats_title, (self.stats_x + 10, self.stats_y + 10))
        
        # Stats
        y_offset = self.stats_y + 45
        line_height = 25
        
        stats = [
            f"Current Iteration: {self.current_iteration}",
            f"Buffer Size: {self.buffer_size:,} samples",
            f"Last Eval Win Rate: {self.last_eval_win_rate:.1%}",
            f"Best Win Rate: {self.best_win_rate:.1%}",
            f"Training Time: {self._format_time(time.time() - self.start_time)}",
            f"Current Phase: {self.current_phase or 'N/A'}",
        ]
        
        for stat in stats:
            stat_text = self.small_font.render(stat, True, (0, 0, 0))
            self.screen.blit(stat_text, (self.stats_x + 20, y_offset))
            y_offset += line_height
        
        # Controls
        y_offset += 10
        controls_text = self.small_font.render(
            "Controls: Q=Quit | S=Screenshot",
            True,
            (100, 100, 100)
        )
        self.screen.blit(controls_text, (self.stats_x + 20, y_offset))
    
    def _render_messages(self):
        """Render recent messages."""
        if not self.messages:
            return
        
        msg_y = self.stats_y + 230
        
        for i, msg in enumerate(reversed(list(self.messages))):
            alpha = max(50, 255 - i * 40)  # Fade older messages
            msg_text = self.small_font.render(msg, True, (0, 100, 0))
            msg_text.set_alpha(alpha)
            self.screen.blit(msg_text, (self.stats_x + 20, msg_y + i * 20))
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds / 86400)
            hours = int((seconds % 86400) / 3600)
            return f"{days}d {hours}h"
    
    def _save_screenshot(self):
        """Save a screenshot of the current visualization."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"training_viz_{timestamp}.png"
        
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        
        filepath = screenshots_dir / filename
        pygame.image.save(self.screen, str(filepath))
        
        print(f"Screenshot saved: {filepath}")
        self.messages.append(f"Screenshot saved: {filename}")


if __name__ == "__main__":
    # Test the visualizer with mock data
    print("Starting visualizer test...")
    print("Press Q to quit, S to save screenshot")
    
    viz = RealTimeVisualizer(log_file="logs/training_log.csv")
    viz.start()
    
    # Simulate some updates
    import random
    
    for i in range(10):
        time.sleep(1)
        
        # Send metrics update
        metrics = MetricsUpdate(
            iteration=i + 1,
            total_loss=10.0 - i * 0.5 + random.random(),
            policy_loss=7.0 - i * 0.3 + random.random() * 0.5,
            value_loss=3.0 - i * 0.2 + random.random() * 0.3,
            buffer_size=(i + 1) * 5000
        )
        viz.update(metrics)
        
        # Send status update
        status = StatusUpdate(
            message=f"Training iteration {i + 1}",
            iteration=i + 1,
            phase="training"
        )
        viz.update(status)
        
        # Send game state every few iterations
        if i % 3 == 0:
            # Create a simple test board
            board = np.zeros((10, 10), dtype=np.int8)
            for row in range(4):
                for col in range(10):
                    if (row + col) % 2 == 1:
                        board[row, col] = 1
            for row in range(6, 10):
                for col in range(10):
                    if (row + col) % 2 == 1:
                        board[row, col] = -1
            
            game_state = GameStateUpdate(
                game_array=board,
                policy=np.random.rand(150) * 0.1,
                move_count=i * 5
            )
            viz.update(game_state)
        
        # Send eval update every 10 iterations (when i+1 is divisible by 10)
        if (i + 1) % 10 == 0:
            eval_update = EvaluationUpdate(
                iteration=i + 1,
                win_rate=0.5 + i * 0.02,
                wins=30 + i,
                draws=10,
                losses=10 - i // 2,
                is_best=(i + 1) % 20 == 0
            )
            viz.update(eval_update)
    
    # Keep running until user closes
    print("Test updates complete. Visualizer still running...")
    print("Close the pygame window or press Ctrl+C to exit")
    
    try:
        while viz.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping visualizer...")
    
    viz.stop()
    print("✓ Visualizer test complete!")
