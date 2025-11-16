"""
Friction detector for wheel-track collision detection.

This module provides accurate polygon-based geometry for detecting which track
tiles each wheel is touching, replacing the Box2D contact listener with spatial
geometry queries.
"""

import numpy as np


class FrictionDetector:
    """
    Detects wheel-track collisions using accurate polygon-based geometry.

    Replaces Box2D contact listener with spatial geometry queries.
    Uses spatial partitioning for performance: only checks tiles near the car
    (~61 tiles instead of all 300), reducing computational cost by 80%.

    Performance Optimization:
    - Two-stage search: coarse (every 10th tile) then fine refinement
    - Spatial range: ±30 tiles around car position (wraps around for circular track)
    - Per-step cost: ~244 polygon checks (4 wheels × 61 tiles)
    - Tolerance: 0.3 units outside polygon edge for wheel-on-track detection

    Methods:
    - update_contacts(): Main entry point, called each physics step
    - _point_in_polygon(): Ray casting algorithm for point-in-polygon test
    - _distance_to_polygon_edge(): Minimum distance from point to polygon edges
    """
    def __init__(self, env, lap_complete_percent):
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def _point_in_polygon(self, px, py, vertices):
        """
        Check if point (px, py) is inside polygon using ray casting algorithm.

        Args:
            px, py: Point coordinates
            vertices: List of (x, y) tuples defining polygon vertices

        Returns:
            True if point is inside polygon, False otherwise
        """
        n = len(vertices)
        inside = False

        x1, y1 = vertices[0]
        for i in range(1, n + 1):
            x2, y2 = vertices[i % n]
            # Ray casting: count intersections with edges
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                inside = not inside
            x1, y1 = x2, y2

        return inside

    def _distance_to_polygon_edge(self, px, py, vertices):
        """
        Calculate minimum distance from point to polygon edges.

        Args:
            px, py: Point coordinates
            vertices: List of (x, y) tuples defining polygon vertices

        Returns:
            Minimum distance from point to any polygon edge
        """
        min_dist = float('inf')

        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]

            # Vector from edge start to end
            dx = x2 - x1
            dy = y2 - y1
            len_sq = dx * dx + dy * dy

            if len_sq < 1e-10:
                # Degenerate edge (two vertices at same position)
                dist = np.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                # Project point onto edge line segment
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

            min_dist = min(min_dist, dist)

        return min_dist

    def update_contacts(self, car, road_tiles):
        """
        Update wheel-tile contacts based on accurate polygon geometry.
        Uses spatial partitioning to only check nearby tiles (~61 instead of 300).
        Called each step to determine which tiles each wheel is touching.

        Performance: ~2000 geometric operations per step (4 wheels × 61 tiles × 8 ops)
        Still very fast due to simple arithmetic and spatial partitioning.
        """
        # Small tolerance for wheels just barely off the track edge
        # This accounts for wheel radius and numerical precision
        NEAR_TRACK_THRESHOLD = 0.3  # Allow 0.3 units outside polygon edge
        SPATIAL_CHECK_RANGE = 30  # Only check tiles within ±30 indices of car position

        # Clear old contacts
        for wheel in car.wheels:
            wheel.tiles.clear()

        # Get car position to determine nearby tiles
        car_x, car_y = car.hull.position

        # Find closest tile to car using cached tile centers
        # Two-stage coarse-then-fine search for efficiency
        min_dist_sq = float('inf')
        closest_tile_idx = 0

        # Coarse search: check every 10th tile
        for i in range(0, len(road_tiles), 10):
            tile = road_tiles[i]
            dx = car_x - tile.center_x
            dy = car_y - tile.center_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_tile_idx = i

        # Fine search: refine around coarse result
        search_start = max(0, closest_tile_idx - 10)
        search_end = min(len(road_tiles), closest_tile_idx + 11)
        for i in range(search_start, search_end):
            tile = road_tiles[i]
            dx = car_x - tile.center_x
            dy = car_y - tile.center_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_tile_idx = i

        # Build list of nearby tiles to check (wrap around for circular track)
        num_tiles = len(road_tiles)
        tile_indices_to_check = []
        for offset in range(-SPATIAL_CHECK_RANGE, SPATIAL_CHECK_RANGE + 1):
            idx = (closest_tile_idx + offset) % num_tiles
            tile_indices_to_check.append(idx)

        # Check each wheel against nearby track tiles using polygon geometry
        for wheel_idx, wheel in enumerate(car.wheels):
            # Use actual wheel position (set in Car._update_hull())
            wheel_world_x = wheel.position[0]
            wheel_world_y = wheel.position[1]

            for tile_idx in tile_indices_to_check:
                tile = road_tiles[tile_idx]

                # Check if wheel center is inside tile polygon
                inside = self._point_in_polygon(wheel_world_x, wheel_world_y, tile.vertices)

                if not inside:
                    # Wheel is outside polygon - check if it's close to the edge
                    dist_to_edge = self._distance_to_polygon_edge(
                        wheel_world_x, wheel_world_y, tile.vertices
                    )
                    if dist_to_edge > NEAR_TRACK_THRESHOLD:
                        continue  # Wheel is too far from this tile

                # Wheel is on track (inside polygon or very close to edge)
                wheel.tiles.add(tile)

                # Get car ID (multi-car support)
                car_id = car.car_id if hasattr(car, 'car_id') else 0

                # Initialize per-car tracking on first use (multi-car support)
                if not hasattr(tile, 'visited_by_cars'):
                    tile.visited_by_cars = set()
                    tile.road_visited = False  # Keep for backward compatibility

                # Handle tile visitation (per-car tracking)
                if car_id not in tile.visited_by_cars:
                    tile.visited_by_cars.add(car_id)
                    tile.road_visited = True  # Backward compatibility

                    # Update per-car tile count
                    if self.env.num_cars > 1:
                        self.env.car_tile_visited_counts[car_id] += 1
                    else:
                        self.env.tile_visited_count += 1

                    # Update furthest tile reached (for progress tracking)
                    # Anti-exploit: Only update if car is moving forward (prevents backward driving exploits)
                    car_forward_velocity = car.vx if hasattr(car, 'vx') else 0.0
                    is_moving_forward = car_forward_velocity > 0.1  # Must be moving forward at > 0.1 m/s

                    if is_moving_forward and tile.idx > self.env.furthest_tile_idx:
                        self.env.furthest_tile_idx = tile.idx
                    elif not is_moving_forward and self.env.verbose and tile.idx > self.env.furthest_tile_idx:
                        # Debug: car reached new tile while moving backward
                        car_str = f"Car {car_id} " if self.env.num_cars > 1 else ""
                        print(f"  ⚠ {car_str}Tile {tile.idx} reached while moving BACKWARD "
                              f"(vx={car_forward_velocity:.2f} m/s) - NO PROGRESS UPDATE")

                    # Lap completion check (per-car)
                    if tile.idx == 0:
                        if self.env.num_cars > 1:
                            progress = self.env.car_tile_visited_counts[car_id] / len(self.env.track)
                        else:
                            progress = self.env.tile_visited_count / len(self.env.track)

                        if progress > self.lap_complete_percent:
                            self.env.new_lap = True
