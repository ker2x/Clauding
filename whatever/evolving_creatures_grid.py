import pygame
import numpy as np
import random
import math

pygame.init()

# Grid-based world
GRID_WIDTH, GRID_HEIGHT = 60, 60
CELL_SIZE = 10  # pixels per grid cell
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 60
UPDATES_PER_FRAME = 10  # Run 10 evolution steps per rendered frame
MAX_FOOD = 100

class DNA:
    """Genetic code for a creature"""
    def __init__(self, genes=None):
        if genes is None:
            # 26 genes: 3 body traits + 12 weights (6→2) + 10 weights (2→5)
            # 6 inputs: energy + N/E/S/W neighbors + movement_success
            # 5 outputs: N, E, S, W, Stay
            self.genes = np.random.uniform(-1, 1, 26).astype(np.float32)
        else:
            self.genes = genes.copy()

    def mutate(self, mutation_rate=0.1):
        """Randomly mutate some genes"""
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] += random.gauss(0, 0.3)
                self.genes[i] = np.clip(self.genes[i], -1, 1)

    def crossover(self, other):
        """Create offspring DNA from two parents"""
        offspring_genes = np.zeros(26, dtype=np.float32)
        for i in range(26):
            offspring_genes[i] = self.genes[i] if random.random() < 0.5 else other.genes[i]
        offspring = DNA(offspring_genes)
        offspring.mutate()
        return offspring


class Creature:
    """A creature on a grid"""
    def __init__(self, x, y, dna=None, initial_energy=40):
        self.x = x  # integer grid coordinate
        self.y = y
        self.dna = dna if dna else DNA()

        # Physical traits encoded in DNA
        self.size = 0.3 + abs(self.dna.genes[0]) * 0.5  # affects visibility
        self.max_speed = 1  # all creatures move 1 cell per frame max

        # Color based on DNA
        r = int(128 + 127 * np.tanh(self.dna.genes[1]))
        g = int(128 + 127 * np.tanh(self.dna.genes[2]))
        b = int(128 + 127 * np.tanh(self.dna.genes[3]))
        self.color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

        # Energy system
        self.energy = initial_energy
        self.max_energy = 150
        self.age = 0

        # Sensory inputs (neighborhood based)
        self.neighbor_north = 0  # what's in N neighbor
        self.neighbor_east = 0   # what's in E neighbor
        self.neighbor_south = 0  # what's in S neighbor
        self.neighbor_west = 0   # what's in W neighbor
        self.movement_success = 1.0  # 1=moved successfully, 0=blocked
        self.energy_level = 1.0

    def brain_forward(self):
        """Neural network: 6 inputs -> 2 hidden -> 5 outputs (N, E, S, W, Stay)"""
        # Inputs: energy + 4 neighbors + movement_success
        inputs = np.array([
            self.energy_level,
            self.neighbor_north,
            self.neighbor_east,
            self.neighbor_south,
            self.neighbor_west,
            self.movement_success
        ], dtype=np.float32)

        # First layer: 6 inputs -> 2 hidden (genes[3:15])
        h1 = np.tanh(inputs @ self.dna.genes[3:15].reshape(6, 2))

        # Second layer: 2 hidden -> 5 outputs (genes[15:25])
        output = np.tanh(h1 @ self.dna.genes[15:25].reshape(2, 5))

        # Pick action with highest activation
        action = np.argmax(output)
        return action  # 0=N, 1=E, 2=S, 3=W, 4=Stay

    def update_senses(self, creatures, food):
        """Update sensory information - local neighborhood detection"""
        self.energy_level = self.energy / self.max_energy

        # Check each of 4 neighbors: 0=empty, 1=food, 0.5=smaller creature, -0.5=larger creature
        neighbors = {
            0: (self.x, self.y - 1),  # North
            1: (self.x + 1, self.y),  # East
            2: (self.x, self.y + 1),  # South
            3: (self.x - 1, self.y)   # West
        }

        for direction, (nx, ny) in neighbors.items():
            # Skip out of bounds
            if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
                neighbor_val = 0  # boundary = empty
            else:
                # Check if there's food
                food_here = False
                for fx, fy in food:
                    if fx == nx and fy == ny:
                        food_here = True
                        break

                if food_here:
                    neighbor_val = 1.0  # food
                else:
                    # Check if there's a creature
                    creature_here = None
                    for other in creatures:
                        if other is not self and other.x == nx and other.y == ny:
                            creature_here = other
                            break

                    if creature_here:
                        # Creature present: encode size comparison
                        if creature_here.size < self.size:
                            neighbor_val = 0.5  # smaller creature
                        else:
                            neighbor_val = -0.5  # larger creature
                    else:
                        neighbor_val = 0.0  # empty

            # Assign to appropriate neighbor
            if direction == 0:
                self.neighbor_north = neighbor_val
            elif direction == 1:
                self.neighbor_east = neighbor_val
            elif direction == 2:
                self.neighbor_south = neighbor_val
            elif direction == 3:
                self.neighbor_west = neighbor_val

    def update(self, creatures, food):
        """Turn-based grid movement: decide action, move 1 square"""
        self.update_senses(creatures, food)

        # Get brain output (action)
        action = self.brain_forward()

        # Compute new position
        new_x, new_y = self.x, self.y
        moved = False

        if action == 0:  # North
            new_y = max(0, self.y - 1)
            moved = True
        elif action == 1:  # East
            new_x = min(GRID_WIDTH - 1, self.x + 1)
            moved = True
        elif action == 2:  # South
            new_y = min(GRID_HEIGHT - 1, self.y + 1)
            moved = True
        elif action == 3:  # West
            new_x = max(0, self.x - 1)
            moved = True
        elif action == 4:  # Stay
            moved = False

        # Check if target square is occupied
        occupant = None
        if moved:
            for other in creatures:
                if other is not self and other.x == new_x and other.y == new_y:
                    occupant = other
                    break

        if occupant:
            # Combat: larger creature wins
            if self.size > occupant.size:
                # I win, eat the occupant
                self.energy = min(self.energy + 15, self.max_energy)
                occupant.energy = 0  # Kill them
                self.x, self.y = new_x, new_y
                self.movement_success = 1.0  # Move succeeded
                self.energy -= 0.3  # cost of combat
            else:
                # They win, I can't move (stay in place, lose some energy)
                self.movement_success = 0.0  # Blocked by creature
                self.energy -= 0.3
        elif moved:
            # Check if actually moved (not blocked by wall)
            if new_x != self.x or new_y != self.y:
                # Square is free, move normally
                self.x, self.y = new_x, new_y
                self.movement_success = 1.0  # Move succeeded
                self.energy -= 0.5
            else:
                # Hit boundary (tried to move but clamped to same position)
                self.movement_success = 0.0  # Blocked by wall
                self.energy -= 0.3
        else:
            # Stay action
            self.movement_success = 1.0  # Stay is always "successful"
            self.energy -= 0.1

        self.age += 1

    def eat(self, food_list):
        """Eat food at current grid cell"""
        for i in range(len(food_list) - 1, -1, -1):
            fx, fy = food_list[i]
            if fx == self.x and fy == self.y:
                self.energy = min(self.energy + 25, self.max_energy)
                food_list.pop(i)

    def reproduce(self):
        """Create offspring"""
        if self.energy > 80:
            self.energy -= 50
            offspring_dna = self.dna.crossover(self.dna)
            # Spawn nearby with hard boundaries
            ox = max(0, min(GRID_WIDTH - 1, self.x + random.randint(-2, 2)))
            oy = max(0, min(GRID_HEIGHT - 1, self.y + random.randint(-2, 2)))
            offspring = Creature(
                ox,
                oy,
                offspring_dna,
                initial_energy=40
            )
            return offspring
        return None

    def is_alive(self):
        return self.energy > 0

    def draw(self, screen):
        """Draw creature as square on grid"""
        px = self.x * CELL_SIZE
        py = self.y * CELL_SIZE
        brightness = self.energy / self.max_energy
        draw_color = tuple(int(c * (0.5 + 0.5 * brightness)) for c in self.color)
        pygame.draw.rect(screen, draw_color, (px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2))


class EvolvingCreaturesGrid:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Evolving Creatures (Grid): Bootstrap Evolution")
        self.clock = pygame.time.Clock()
        self.running = True

        self.creatures = []
        self.food = []

        # Initialize with some creatures
        for _ in range(5):
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            self.creatures.append(Creature(x, y, initial_energy=40))

        # Spawn initial food
        for _ in range(30):
            self.food.append((random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)))

        self.time = 0
        self.paused = False
        self.updates_per_frame = UPDATES_PER_FRAME

    def spawn_food(self):
        """Spawn new food"""
        if len(self.food) < MAX_FOOD and random.random() < 0.5:
            self.food.append((random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)))

    def spawn_creatures(self):
        """Continuously spawn random creatures (genetic lottery)"""
        spawn_rate = 0.5 * (1.0 - len(self.creatures) / 150.0)

        if random.random() < spawn_rate:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            self.creatures.append(Creature(x, y, initial_energy=40))

    def update(self):
        """Update all creatures"""
        if self.paused:
            return

        # Update creatures
        for creature in self.creatures:
            creature.update(self.creatures, self.food)
            creature.eat(self.food)

        # Reproduction
        new_creatures = []
        for creature in self.creatures:
            offspring = creature.reproduce()
            if offspring:
                new_creatures.append(offspring)

        self.creatures.extend(new_creatures)

        # Remove dead creatures
        self.creatures = [c for c in self.creatures if c.is_alive()]

        # Spawn food
        self.spawn_food()

        # Spawn new random creatures
        self.spawn_creatures()

        self.time += 1

    def draw(self):
        """Render to screen"""
        self.screen.fill((20, 20, 25))

        # Draw grid
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                px = x * CELL_SIZE
                py = y * CELL_SIZE
                pygame.draw.rect(self.screen, (40, 40, 45), (px, py, CELL_SIZE, CELL_SIZE), 1)

        # Draw food
        for fx, fy in self.food:
            px = fx * CELL_SIZE
            py = fy * CELL_SIZE
            pygame.draw.rect(self.screen, (100, 200, 100), (px + 2, py + 2, CELL_SIZE - 4, CELL_SIZE - 4))

        # Draw creatures
        for creature in self.creatures:
            creature.draw(self.screen)

        # Draw info
        font = pygame.font.Font(None, 18)
        avg_energy = np.mean([c.energy for c in self.creatures]) if self.creatures else 0
        info = f"Creatures: {len(self.creatures)} | Food: {len(self.food)} | "
        info += f"Gen: {self.time} | Avg Energy: {avg_energy:.1f} | "
        info += f"Speed: {self.updates_per_frame}x | P=Pause | S=Speed | Q=Quit"
        if self.paused:
            info += " [PAUSED]"

        text_surf = font.render(info, True, (200, 200, 200))
        self.screen.blit(text_surf, (10, 10))

        pygame.display.flip()

    def handle_events(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.updates_per_frame = 1 if self.updates_per_frame != 1 else 100

    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            # Run multiple updates per frame
            for _ in range(self.updates_per_frame):
                self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    sim = EvolvingCreaturesGrid()
    sim.run()
