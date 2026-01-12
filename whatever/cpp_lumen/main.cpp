#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

const int WIDTH = 1400;
const int HEIGHT = 800;
const int NUM_PARTICLES = 3000;
const int NUM_TYPES = 6;
const float FRICTION = 0.85f;
const float R_MAX = 80.0f;
const float FORCE_STRENGTH = 0.5f;

struct Particle {
    float x, y;
    float vx, vy;
    int type;
};

struct Config {
    float matrix[NUM_TYPES][NUM_TYPES];
    SDL_Color colors[NUM_TYPES];
};

void init_config(Config& config) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_force(-1.0f, 1.0f);
    
    for (int i = 0; i < NUM_TYPES; ++i) {
        for (int j = 0; j < NUM_TYPES; ++j) {
            config.matrix[i][j] = dist_force(gen);
        }
    }

    // Photonic/Neon Palette
    config.colors[0] = {0, 255, 255, 255};   // Cyan
    config.colors[1] = {255, 0, 255, 255};   // Magenta
    config.colors[2] = {255, 255, 0, 255};   // Yellow
    config.colors[3] = {0, 255, 128, 255};   // Electric Green
    config.colors[4] = {255, 128, 0, 255};   // Orange
    config.colors[5] = {128, 0, 255, 255};   // Purple
}

void apply_force(Particle& p, const Particle& other, float g) {
    float dx = other.x - p.x;
    float dy = other.y - p.y;
    
    // Toroidal distance
    if (dx > WIDTH / 2) dx -= WIDTH;
    if (dx < -WIDTH / 2) dx += WIDTH;
    if (dy > HEIGHT / 2) dy -= HEIGHT;
    if (dy < -HEIGHT / 2) dy += HEIGHT;

    float dist_sq = dx * dx + dy * dy;
    if (dist_sq > 0 && dist_sq < R_MAX * R_MAX) {
        float r = std::sqrt(dist_sq);
        float force = 0;
        float normalized_r = r / R_MAX;

        if (normalized_r < 0.3f) {
            force = (normalized_r / 0.3f) - 1.0f; // Strong repulsion at close range
        } else {
            force = g * (1.0f - std::abs(2.0f * normalized_r - 1.0f - 0.3f) / (1.0f - 0.3f));
        }

        p.vx += (dx / r) * force * FORCE_STRENGTH;
        p.vy += (dy / r) * force * FORCE_STRENGTH;
    }
}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("LumenParticles - C++ M3 Optimized", 
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                                          WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    Config config;
    init_config(config);

    std::vector<Particle> particles(NUM_PARTICLES);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_x(0, WIDTH);
    std::uniform_real_distribution<float> dist_y(0, HEIGHT);
    std::uniform_int_distribution<int> dist_type(0, NUM_TYPES - 1);

    for (auto& p : particles) {
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.vx = 0;
        p.vy = 0;
        p.type = dist_type(gen);
    }

    bool running = true;
    SDL_Event event;

    // Simulation loop
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_r) init_config(config);
                if (event.key.keysym.sym == SDLK_q) running = false;
            }
        }

        // Physics Step
        // Optimized loop for M3: Inner loop is small and tight
        for (int i = 0; i < NUM_PARTICLES; ++i) {
            Particle& p = particles[i];
            for (int j = 0; j < NUM_PARTICLES; ++j) {
                if (i == j) continue;
                apply_force(p, particles[j], config.matrix[p.type][particles[j].type]);
            }
        }

        // Update positions
        for (auto& p : particles) {
            p.x += p.vx;
            p.y += p.vy;
            p.vx *= FRICTION;
            p.vy *= FRICTION;

            // Boundary wrapping
            if (p.x < 0) p.x += WIDTH;
            if (p.x >= WIDTH) p.x -= WIDTH;
            if (p.y < 0) p.y += HEIGHT;
            if (p.y >= HEIGHT) p.y -= HEIGHT;
        }

        // Render
        SDL_SetRenderDrawColor(renderer, 10, 10, 20, 255); // Dark cosmic background
        SDL_RenderClear(renderer);

        for (const auto& p : particles) {
            SDL_SetRenderDrawColor(renderer, config.colors[p.type].r, config.colors[p.type].g, config.colors[p.type].b, 255);
            SDL_Rect rect = { (int)p.x, (int)p.y, 3, 3 };
            SDL_RenderFillRect(renderer, &rect);
        }

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
