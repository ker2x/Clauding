
#pragma once

#include "game.h"
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace checkers {

// Node for MCTS Tree
struct Node {
  Game game; // State at this node
  Node *parent;
  int action_from_parent;

  std::vector<std::unique_ptr<Node>> children;
  std::vector<int> actions;  // Actions corresponding to children
  std::vector<float> priors; // Priors for children

  int visit_count;
  float value_sum;  // Sum of values from subtree
  float mean_value; // Q-value (value_sum / visit_count)

  bool is_expanded;

  // Virtual loss for batching
  int virtual_loss;

  Node(const Game &g, Node *p, int a)
      : game(g), parent(p), action_from_parent(a), visit_count(0),
        value_sum(0.0f), mean_value(0.0f), is_expanded(false), virtual_loss(0) {
  }

  ~Node() = default;

  // UCB / PUCT Score
  float get_ucb(float c_puct, float parent_visits_sqrt) const {
    // Q + U
    // Q = mean_value
    // U = c_puct * prior * sqrt(parent_visits) / (1 + visits)

    // Calculate PUCT score with Virtual Loss
    // Virtual loss increases the visit count temporarily to discourage other
    // threads/batches from selecting the same node while it's being evaluated.
    int effective_visits = visit_count + virtual_loss;
    float q = 0.0f;
    if (effective_visits > 0) {
      // Treat virtual loss as a loss for value calculation purposes
      float effective_value_sum = value_sum - virtual_loss;
      q = effective_value_sum / effective_visits;
    }

    // Getting Prior
    // We need to look up OUR prior from PARENT.
    // But Node struct doesn't store its own prior efficiently?
    // Parent stores `priors` vector. We need to find index.
    // This is slow. Store prior in node?
    return 0.0f; // Placeholder, logic in select_child
  }
};

class MCTS {
public:
  MCTS(float c_puct, int num_simulations, float dirichlet_alpha,
       float dirichlet_epsilon);

  // Perform MCTS search
  // Usage:
  // 1. calls start_search(root_game)
  // 2. Loop until is_finished():
  //    - find_leaves(batch_size) -> returns batch of leaf states for neural net
  //    - Neural net predicts policies and values
  //    - process_results(batch_id, policies, values) -> expands leaves and
  //    backups values
  // 3. get_policy(temp) -> returns interaction policy

  void start_search(const Game &root_game);

  // Batching methods
  // Find a batch of leaf nodes that need evaluation.
  // Returns a tuple of (batch_id, list_of_neural_inputs).

  // returns (batch_id, list_of_flat_inputs)
  std::pair<int, std::vector<std::vector<float>>> find_leaves(int batch_size);

  // Process results for a batch
  void process_results(int batch_id,
                       const std::vector<std::vector<float>> &policies,
                       const std::vector<float> &values);

  // Check if we need more simulations
  bool is_finished() const;

  // Get final policy
  std::vector<float> get_policy(float temperature);
  int get_root_visit_count() const;

private:
  float c_puct;
  int num_simulations;
  float dirichlet_alpha;
  float dirichlet_epsilon;

  std::unique_ptr<Node> root;
  int simulations_completed;

  // Pending leaves for current batch
  struct PendingBatch {
    int id;
    std::vector<Node *> nodes;
  };
  std::map<int, PendingBatch> pending_batches;
  int next_batch_id;

  // Random engine
  std::mt19937 rng;

  // Helper
  void add_dirichlet_noise(Node *node);
  Node *select_leaf(Node *sapling);
  void expand_node(Node *node, const std::vector<float> &policy, float value);
  void backup(Node *node, float value);
};

} // namespace checkers
