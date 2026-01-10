"""
Real-time line chart plotting for pygame.
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class LineChart:
    """
    Scrolling line chart for real-time data visualization.
    
    Supports multiple series with automatic scaling and legend.
    """
    
    def __init__(
        self,
        rect: pygame.Rect,
        title: str,
        max_points: int = 100,
        y_label: str = "",
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ):
        """
        Initialize line chart.
        
        Args:
            rect: Rectangle defining chart position and size
            title: Chart title
            max_points: Maximum number of points to display (scrolling)
            y_label: Label for Y axis
            colors: Dictionary mapping series names to RGB colors
        """
        self.rect = rect
        self.title = title
        self.max_points = max_points
        self.y_label = y_label
        
        # Data storage: {series_name: deque of (x, y) tuples}
        self.data: Dict[str, deque] = {}
        
        # Colors for each series
        self.colors = colors or {}
        self.default_colors = [
            (66, 135, 245),   # Blue
            (245, 66, 66),    # Red
            (66, 245, 135),   # Green
            (245, 200, 66),   # Yellow
            (200, 66, 245),   # Purple
            (245, 135, 66),   # Orange
        ]
        self.color_index = 0
        
        # Fonts
        try:
            self.title_font = pygame.font.SysFont('Arial', 16, bold=True)
            self.label_font = pygame.font.SysFont('Arial', 11)
        except:
            self.title_font = pygame.font.Font(None, 20)
            self.label_font = pygame.font.Font(None, 14)
        
        # Margins
        self.margin_left = 60
        self.margin_right = 20
        self.margin_top = 40
        self.margin_bottom = 40
        
    def add_data_point(self, series_name: str, x: float, y: float):
        """
        Add a data point to a series.
        
        Args:
            series_name: Name of the data series
            x: X coordinate (e.g., iteration number)
            y: Y value (e.g., loss)
        """
        if series_name not in self.data:
            self.data[series_name] = deque(maxlen=self.max_points)
            
            # Assign color if not specified
            if series_name not in self.colors:
                self.colors[series_name] = self.default_colors[self.color_index % len(self.default_colors)]
                self.color_index += 1
        
        self.data[series_name].append((x, y))
    
    def clear(self):
        """Clear all data."""
        self.data.clear()
    
    def render(self, surface: pygame.Surface):
        """
        Render the chart to a surface.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw background
        pygame.draw.rect(surface, (255, 255, 255), self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        
        # Draw title
        title_surf = self.title_font.render(self.title, True, (0, 0, 0))
        title_rect = title_surf.get_rect(centerx=self.rect.centerx, top=self.rect.top + 10)
        surface.blit(title_surf, title_rect)
        
        # Calculate plot area
        plot_left = self.rect.left + self.margin_left
        plot_right = self.rect.right - self.margin_right
        plot_top = self.rect.top + self.margin_top
        plot_bottom = self.rect.bottom - self.margin_bottom
        
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        
        if plot_width <= 0 or plot_height <= 0:
            return
        
        # Get all data points to determine ranges
        all_x = []
        all_y = []
        for series in self.data.values():
            for x, y in series:
                all_x.append(x)
                all_y.append(y)
        
        if not all_x:
            # No data yet
            no_data_surf = self.label_font.render("No data yet", True, (150, 150, 150))
            surface.blit(no_data_surf, (self.rect.centerx - 40, self.rect.centery))
            return
        
        # Determine ranges
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add padding to y range
        y_range = y_max - y_min
        if y_range < 0.001:
            y_range = 1.0
            
        # Calculate nice ticks
        ticks = self._calculate_nice_ticks(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Adjust min/max to include ticks
        if ticks:
            y_min = min(y_min, ticks[0])
            y_max = max(y_max, ticks[-1])
            y_range = y_max - y_min
        
        # Ensure x range is at least 1
        if x_max - x_min < 1:
            x_max = x_min + 1
        
        # Draw grid lines
        for val in ticks:
            # Map value to screen Y
            # Note: y goes down in screen coords, so we flip
            ratio = (val - y_min) / y_range
            y = plot_bottom - ratio * plot_height
            
            # Choose color - highlight 0 and 1
            if abs(val - 1.0) < 0.001 or abs(val) < 0.001:
                color = (150, 150, 150) # Darker for 0 and 1
                width = 2
            else:
                color = (230, 230, 230)
                width = 1
                
            pygame.draw.line(surface, color, (plot_left, y), (plot_right, y), width)
            
            # Y axis labels
            # Format: integer if close to integer, else float
            if abs(val - round(val)) < 0.001:
                label_text = f"{int(round(val))}"
            else:
                label_text = f"{val:.1f}"
                
            label = self.label_font.render(label_text, True, (100, 100, 100))
            # Align right
            surface.blit(label, (plot_left - label.get_width() - 5, y - 7))
        
        # Draw axes
        pygame.draw.line(surface, (0, 0, 0), (plot_left, plot_top), (plot_left, plot_bottom), 2)
        pygame.draw.line(surface, (0, 0, 0), (plot_left, plot_bottom), (plot_right, plot_bottom), 2)


        
        # Draw Y axis label
        if self.y_label:
            y_label_surf = self.label_font.render(self.y_label, True, (0, 0, 0))
            # Rotate would require more complex rendering, just place it
            surface.blit(y_label_surf, (self.rect.left + 5, self.rect.centery - 30))
        
        # Draw data series
        for series_name, points in self.data.items():
            if len(points) < 2:
                continue
            
            color = self.colors.get(series_name, (0, 0, 0))
            
            # Convert points to screen coordinates
            screen_points = []
            for x, y in points:
                screen_x = plot_left + ((x - x_min) / (x_max - x_min)) * plot_width
                screen_y = plot_bottom - ((y - y_min) / (y_max - y_min)) * plot_height
                screen_points.append((screen_x, screen_y))
            
            # Draw lines
            if len(screen_points) >= 2:
                pygame.draw.lines(surface, color, False, screen_points, 2)
                
                # Draw points
                for point in screen_points:
                    pygame.draw.circle(surface, color, (int(point[0]), int(point[1])), 3)
        
        # Draw legend
        legend_x = plot_right - 150
        legend_y = plot_top + 10
        
        for i, (series_name, color) in enumerate(self.colors.items()):
            if series_name not in self.data or not self.data[series_name]:
                continue
            
            y_pos = legend_y + i * 20
            
            # Color box
            pygame.draw.rect(surface, color, (legend_x, y_pos, 15, 15))
            pygame.draw.rect(surface, (0, 0, 0), (legend_x, y_pos, 15, 15), 1)
            
            # Label
            label_surf = self.label_font.render(series_name, True, (0, 0, 0))
            surface.blit(label_surf, (legend_x + 20, y_pos))

    def _calculate_nice_ticks(self, v_min: float, v_max: float, max_ticks: int = 6) -> List[float]:
        """Calculate human-readable tick values."""
        if v_max <= v_min:
            return [v_min]
            
        rng = v_max - v_min
        raw_step = rng / max_ticks
        
        # Find order of magnitude
        mag = 10 ** np.floor(np.log10(raw_step))
        norm_step = raw_step / mag
        
        # Snap to nice steps: 1, 2, 5
        if norm_step <= 1: step = 1 * mag
        elif norm_step <= 2: step = 2 * mag
        elif norm_step <= 5: step = 5 * mag
        else: step = 10 * mag
        
        # Generate ticks
        start = np.ceil(v_min / step) * step
        end = np.floor(v_max / step) * step
        
        # Use simple epsilon loop to avoid float errors
        ticks = []
        curr = start
        while curr <= end + step * 0.01:
            ticks.append(curr)
            curr += step
            
        return ticks


if __name__ == "__main__":
    # Test the line chart
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Line Chart Test")
    
    clock = pygame.time.Clock()
    
    chart = LineChart(
        rect=pygame.Rect(50, 50, 700, 500),
        title="Training Loss",
        y_label="Loss"
    )
    
    # Add some test data
    for i in range(50):
        chart.add_data_point("Total", i, 10 - i * 0.15 + np.random.randn() * 0.5)
        chart.add_data_point("Policy", i, 7 - i * 0.1 + np.random.randn() * 0.3)
        chart.add_data_point("Value", i, 3 - i * 0.05 + np.random.randn() * 0.2)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        screen.fill((240, 240, 240))
        chart.render(screen)
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("✓ Line chart test passed!")
