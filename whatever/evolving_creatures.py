import pygame
import numpy as np
import random
import math
from collections import deque

pygame.init()

WIDTH, HEIGHT = 1200, 800
FPS = 60
MAX_CREATURES = 100
MAX_FOOD = 150

class DNA:
    """Genetic code for a creature"""
    def __init__(self, genes=None):
        if genes is None:
            # Create random DNA
            # 13 genes: 3 body traits + 10 neural weights (3 inputs -> 2 hidden -> 2 outputs)
            self.genes = np.random.uniform(-1, 1, 13).astype(np.float32)
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
        offspring_genes = np.zeros(13, dtype=np.float32)
        for i in range(13):
            offspring_genes[i] = self.genes[i] if random.random() < 0.5 else other.genes[i]
        offspring = DNA(offspring_genes)
        offspring.mutate()
        return offspring


class Creature:
    """A creature with body, brain, and energy"""
    def __init__(self, x, y, dna=None, initial_energy=40):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)

        self.dna = dna if dna else DNA()

        # Physical traits encoded in DNA
        self.size = 3 + abs(self.dna.genes[0]) * 3  # 3-6 pixels
        self.mass = self.size
        self.max_speed = 1.5 + abs(self.dna.genes[1])

        # Color based on DNA
        r = int(128 + 127 * np.tanh(self.dna.genes[2]))
        g = int(128 + 127 * np.tanh(self.dna.genes[3]))
        b = int(128 + 127 * np.tanh(self.dna.genes[4]))
        self.color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

        # Energy system
        self.energy = initial_energy
        self.max_energy = 150
        self.age = 0

        # Sensory inputs for simple neural network
        # We'll compute: closest food distance, closest creature distance, angle to food
        self.food_sense = 0
        self.creature_sense = 0
        self.energy_level = 1.0

    def brain_forward(self):
        """Simple neural network: 3 inputs -> 2 hidden -> 2 outputs (steering)"""
        # Inputs: energy level, food proximity, creature proximity
        inputs = np.array([self.energy_level, self.food_sense, self.creature_sense], dtype=np.float32)

        # First layer: 3 inputs -> 2 hidden (genes[3:9])
        h1 = np.tanh(inputs @ self.dna.genes[3:9].reshape(3, 2))

        # Second layer: 2 hidden -> 2 outputs (genes[9:13])
        output = np.tanh(h1 @ self.dna.genes[9:13].reshape(2, 2))

        # Output: [thrust, turn]
        return output

    def update_senses(self, creatures, food):
        """Update sensory information"""
        self.energy_level = self.energy / self.max_energy

        # Find closest food
        if food:
            min_dist = float('inf')
            for f in food:
                dx = f[0] - self.x
                dy = f[1] - self.y
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
            self.food_sense = max(0, 1.0 - min_dist / 200.0)  # 0 if far, 1 if close
        else:
            self.food_sense = 0

        # Find closest creature (potential threat or mate)
        if len(creatures) > 1:
            min_dist = float('inf')
            for other in creatures:
                if other is self:
                    continue
                dx = other.x - self.x
                dy = other.y - self.y
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
            self.creature_sense = max(0, 1.0 - min_dist / 200.0)
        else:
            self.creature_sense = 0

    def update(self, creatures, food):
        """Update creature physics and energy"""
        self.update_senses(creatures, food)

        # Get brain output
        outputs = self.brain_forward()
        thrust = outputs[0]  # -1 to 1
        turn = outputs[1]    # -1 to 1

        # Apply thrust in direction of current velocity
        if abs(thrust) > 0.1:
            angle = math.atan2(self.vy, self.vx)
            self.vx += thrust * 0.3 * math.cos(angle)
            self.vy += thrust * 0.3 * math.sin(angle)

        # Apply turn (rotate velocity vector)
        if abs(turn) > 0.1:
            angle = math.atan2(self.vy, self.vx)
            angle += turn * 0.05
            speed = math.sqrt(self.vx**2 + self.vy**2)
            speed = min(speed, self.max_speed)
            self.vx = speed * math.cos(angle)
            self.vy = speed * math.sin(angle)

        # Friction
        self.vx *= 0.95
        self.vy *= 0.95

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Boundary wrapping
        self.x = self.x % WIDTH
        self.y = self.y % HEIGHT

        # Energy consumption (moving costs energy)
        speed = math.sqrt(self.vx**2 + self.vy**2)
        self.energy -= 0.1 + speed * 0.05 + abs(thrust) * 0.05

        self.age += 1

    def eat(self, food_list):
        """Eat food at current position"""
        for i in range(len(food_list) - 1, -1, -1):
            fx, fy = food_list[i]
            dx = fx - self.x
            dy = fy - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < self.size:
                self.energy = min(self.energy + 25, self.max_energy)
                food_list.pop(i)

    def reproduce(self):
        """Create offspring"""
        if self.energy > 80:
            self.energy -= 50
            offspring_dna = self.dna.crossover(self.dna)
            offspring = Creature(
                self.x + random.uniform(-20, 20),
                self.y + random.uniform(-20, 20),
                offspring_dna,
                initial_energy=40
            )
            return offspring
        return None

    def is_alive(self):
        return self.energy > 0

    def draw(self, screen):
        # Brightness based on energy
        brightness = self.energy / self.max_energy
        draw_color = tuple(int(c * (0.5 + 0.5 * brightness)) for c in self.color)
        pygame.draw.circle(screen, draw_color, (int(self.x), int(self.y)), int(self.size))

        # Draw velocity vector
        end_x = self.x + self.vx * 5
        end_y = self.y + self.vy * 5
        pygame.draw.line(screen, draw_color, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)


class EvolvingCreatures:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Evolving Creatures: Neural Networks + Genetics")
        self.clock = pygame.time.Clock()
        self.running = True

        self.creatures = []
        self.food = []

        # Initialize with some creatures (starting energy: 40)
        for _ in range(20):
            x = random.uniform(50, WIDTH - 50)
            y = random.uniform(50, HEIGHT - 50)
            self.creatures.append(Creature(x, y, initial_energy=40))

        # Spawn initial food
        for _ in range(50):
            self.food.append((random.uniform(0, WIDTH), random.uniform(0, HEIGHT)))

        self.time = 0
        self.paused = False

    def spawn_food(self):
        """Spawn new food randomly"""
        if len(self.food) < MAX_FOOD and random.random() < 0.3:
            self.food.append((random.uniform(0, WIDTH), random.uniform(0, HEIGHT)))

    def spawn_creatures(self):
        """Continuously spawn random creatures (genetic lottery)"""
        # Spawn rate based on population: fewer creatures = higher spawn rate
        spawn_rate = 0.3 * (1.0 - len(self.creatures) / 200.0)  # Scale with population

        if random.random() < spawn_rate:
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, HEIGHT)
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

        # Remove dead creatures (natural selection!)
        self.creatures = [c for c in self.creatures if c.is_alive()]

        # Spawn food
        self.spawn_food()

        # Spawn new random creatures (genetic lottery)
        self.spawn_creatures()

        self.time += 1

    def draw(self):
        """Render to screen"""
        self.screen.fill((20, 20, 25))

        # Draw food
        for fx, fy in self.food:
            pygame.draw.circle(self.screen, (100, 200, 100), (int(fx), int(fy)), 2)

        # Draw creatures
        for creature in self.creatures:
            creature.draw(self.screen)

        # Draw info
        font = pygame.font.Font(None, 18)
        avg_energy = np.mean([c.energy for c in self.creatures]) if self.creatures else 0
        info = f"Creatures: {len(self.creatures)} | Food: {len(self.food)} | "
        info += f"Time: {self.time} | Avg Energy: {avg_energy:.1f} | "
        info += f"P=Pause | Q=Quit"
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

    def run(self):
        """Main loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    sim = EvolvingCreatures()
    sim.run()
