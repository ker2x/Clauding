
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

    // Virtual loss adjustment: temporarily decrease Q or increase visit count?
    // Standard way: Add virtual loss to visit count (to discourage selection)
    // and decrease value sum. Or simpler: just use (visit_count + virtual_loss)
    // in denominator.

    int effective_visits = visit_count + virtual_loss;
    float q = 0.0f;
    if (effective_visits > 0) {
      // Apply heavy penalty for virtual loss?
      // AlphaZero just treats virtual loss as "loss" update.
      // If virtual_loss > 0, we can subtract it from value_sum?
      // -1 is loss. So value_sum - virtual_loss.
      // Let's implement AlphaGo Zero virtual loss:
      // "In the search tree, we use virtual loss to enable parallel search.
      // When a thread selects a node ... we add a virtual loss ... "
      // Here we do batching, not multi-threading. But same concept.
      // While a leaf is being evaluated, we want to avoid picking same path.
      // So we treat it as if we visited it and lost.

      float effective_value_sum = value_sum - virtual_loss;
      q = effective_value_sum / effective_visits;
      // However, game result range is -1 to 1? Or 0 to 1?
      // Our game returns 1.0 (win), -1.0 (loss).
      // So virtual loss should be -1.0?
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

  // Python Interface
  // Returns: (policy_probs, value)
  // Runs search from root_game
  // Takes a callback for evaluation: evaluate_func(batch_states) ->
  // (batch_policies, batch_values) Using simple approach: We return a BATCH of
  // states to Python, Python evals, calls back. Wait, simpler: We implement
  // `search_batch` that handles the loop. But we need to call Python function.
  // We can't easily accept a python function in constructor without pybind11
  // overhead in headers. Better: `search` returns void, but we have helper
  // `get_leaves_to_evaluate` and `backup_values`. Python loop: while
  // mcts.has_simulations_left():
  //    batch = mcts.find_leaves(8)
  //    policies, values = network(batch)
  //    mcts.process_results(batch_indices, policies, values)

  void start_search(const Game &root_game);

  // Batching Interface
  // Returns list of leaf Node pointers (as opaque handles or indices?)
  // Actually, passing pointers to Python and back is unsafe/messy.
  // Let's store pending nodes internally and just return list of Game states
  // (neural inputs). And return a batch_id.

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
