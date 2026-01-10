#include "mcts.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace checkers {

MCTS::MCTS(float c_puct, int num_simulations, float dirichlet_alpha,
           float dirichlet_epsilon)
    : c_puct(c_puct), num_simulations(num_simulations),
      dirichlet_alpha(dirichlet_alpha), dirichlet_epsilon(dirichlet_epsilon),
      simulations_completed(0), next_batch_id(0), rng(std::random_device{}()) {}

void MCTS::start_search(const Game &root_game) {
  // 0 = parent action (dummy)
  root = std::make_unique<Node>(root_game, nullptr, 0);
  simulations_completed = 0;
  next_batch_id = 0;
  pending_batches.clear();

  // We need to evaluate root first?
  // Yes. `find_leaves` will pick root if not expanded.
}

bool MCTS::is_finished() const {
  return simulations_completed >= num_simulations;
}

std::pair<int, std::vector<std::vector<float>>>
MCTS::find_leaves(int batch_size) {
  std::vector<std::vector<float>> inputs;
  std::vector<Node *> nodes;

  for (int i = 0;
       i < batch_size && simulations_completed + nodes.size() < num_simulations;
       i++) {
    Node *leaf = select_leaf(root.get());

    bool term = leaf->game.is_terminal();
    if (term) {
      // If terminal, we don't need network eval.
      // We can backup immediately.
      backup(leaf, leaf->game.get_result());
      simulations_completed++;
      // Don't add to batch
      // Decrement loop counter because we "did" a simulation but didn't fill
      // batch slot? Actually `simulations_completed` increased, so we are good.
      // Continue to fill batch?
      // Yes, try to find another leaf.
      i--;
      continue;
    }

    // Add Apply Virtual Loss
    leaf->virtual_loss++;
    // Also update path virtual loss?
    // Usually virtual loss is applied to all nodes on path?
    // "Updates the visit count and value sum of nodes visited during the
    // selection"
    Node *curr = leaf;
    while (curr != nullptr) {
      curr->virtual_loss++;
      curr = curr->parent;
    }

    nodes.push_back(leaf);
    inputs.push_back(leaf->game.to_neural_input());
  }

  if (nodes.empty()) {
    return {-1, {}};
  }

  int bid = next_batch_id++;
  pending_batches[bid] = {bid, nodes};
  return {bid, inputs};
}

Node *MCTS::select_leaf(Node *node) {
  while (node->is_expanded) {
    // Compute PUCT for all children
    float best_score = -FLT_MAX;
    Node *best_child = nullptr;

    // We need total visits of parent.
    // Including virtual losses? Yes.
    float sqrt_n = std::sqrt((float)(node->visit_count + node->virtual_loss));

    for (size_t i = 0; i < node->children.size(); i++) {
      Node *child = node->children[i].get();
      float prior = node->priors[i];

      // Calculate Q
      float q = 0.0f;
      int visits = child->visit_count + child->virtual_loss;
      if (visits > 0) {
        // Invert value for opponent logic?
        // Our `game.get_result()` is from perspective of current player.
        // Q should be value for the player at `node` (who is choosing action).
        // Child node state is after `node` player moved.
        // So child state is from opponent perspective.
        // Value stored in child is from child's player perspective (opponent).
        // So we negate it?
        // `mean_value` in Node: represents value for the player to move at that
        // node? Or value of the state? Standard: Value from Net is "Value for
        // current player". So child->mean_value is good for Child's player
        // (Opponent). So for Node's player, it is -1 * child->mean_value.

        // Handling virtual loss (treated as loss = -1.0)
        // If we add virtual loss to counts, we drive Q down.
        // Q_child = (sum_value - virtual_loss) / (visits + virtual_loss)
        // Evaluation: -Q_child

        float child_sum = child->value_sum - child->virtual_loss;
        float child_q = child_sum / visits;
        q = -child_q;
      } else {
        // FPU (First Play Urgency)?
        // Zero is fine if values are [-1, 1].
        q = 0.0f;
      }

      float u = c_puct * prior * sqrt_n / (1.0f + visits);
      float score = q + u;

      if (score > best_score) {
        best_score = score;
        best_child = child;
      }
    }

    if (best_child == nullptr) {
      // Should not happen unless no children?
      // If terminal, is_expanded should be false?
      break;
    }
    node = best_child;
  }
  return node;
}

void MCTS::process_results(int batch_id,
                           const std::vector<std::vector<float>> &policies,
                           const std::vector<float> &values) {
  if (pending_batches.find(batch_id) == pending_batches.end())
    return;

  PendingBatch &batch = pending_batches[batch_id];

  for (size_t i = 0; i < batch.nodes.size(); i++) {
    Node *node = batch.nodes[i];

    // Remove Virtual Loss first!
    Node *curr = node;
    while (curr != nullptr) {
      curr->virtual_loss--;
      curr = curr->parent;
    }

    float value = values[i];
    const std::vector<float> &policy = policies[i];

    // Expand
    expand_node(node, policy, value);

    // Backup
    backup(node, value);

    simulations_completed++;
  }

  pending_batches.erase(batch_id);
}

void MCTS::expand_node(Node *node, const std::vector<float> &policy_probs,
                       float value) {
  if (node->is_expanded)
    return; // Already done?

  std::vector<int> legal_actions = node->game.get_legal_actions();

  // Helper to fetch prob
  // policy_probs is size 128.

  // Add Dirichlet noise if Root
  std::vector<float> noise;
  if (node->parent == nullptr && !legal_actions.empty()) {
    // Generate gamma samples
    std::gamma_distribution<float> distribution(dirichlet_alpha, 1.0);
    float sum = 0.0f;
    for (size_t k = 0; k < legal_actions.size(); k++) {
      float n = distribution(rng);
      noise.push_back(n);
      sum += n;
    }
    for (size_t k = 0; k < noise.size(); k++)
      noise[k] /= sum;
  }

  float policy_sum = 0.0f;
  for (size_t i = 0; i < legal_actions.size(); i++) {
    int action = legal_actions[i];
    float prob = policy_probs[action];

    if (node->parent == nullptr) {
      prob = (1 - dirichlet_epsilon) * prob + dirichlet_epsilon * noise[i];
    }

    // Create Child
    Game child_game = node->game;
    child_game.make_action(action);

    auto child = std::make_unique<Node>(child_game, node, action);

    node->children.push_back(std::move(child));
    node->actions.push_back(action);
    node->priors.push_back(prob);

    policy_sum += prob;
  }

  // Normalize priors just in case (network output should be softmaxed though)
  /*
  if (policy_sum > 0) {
      for (float& p : node->priors) p /= policy_sum;
  }
  */

  node->is_expanded = true;
}

void MCTS::backup(Node *node, float value) {
  // Value is from perspective of node->game player.
  // Parent needs value from its perspective (-value).

  Node *curr = node;
  // For leaf, we update it.
  // Leaf value is `value`.
  // It hasn't been visited yet in this sim?
  // No, expand adds children but we count visit on the node itself?
  // Actually we update `curr` then go to parent.

  // AlphaZero backup:
  // "Leaf node... value v... "
  // "Backward pass... update edge statistics... visit count N... Action-value
  // Q"

  // We update the path from leaf to root.
  // The value `value` is for `node` state.
  // `node` state is result of `parent`'s action.
  // `parent` wants to maximize its value.
  // If `node` state is good (value=1.0), it means `parent` made a good move.
  // BUT `value` is usually from `node` player perspective.
  // If `node` player is P2, and `value` is 1.0 (P2 wins).
  // `parent` player was P1. P1 lost.
  // So for `parent`, value is -1.0.

  float current_val = value;

  while (curr != nullptr) {
    curr->visit_count++;
    curr->value_sum += current_val;
    curr->mean_value = curr->value_sum / curr->visit_count;

    current_val = -current_val; // Flip for parent
    curr = curr->parent;
  }
}

std::vector<float> MCTS::get_policy(float temperature) {
  std::vector<float> policy(128, 0.0f);

  if (!root)
    return policy;

  if (temperature == 0.0f) {
    // Greedy - Find max visits
    int best_visits = -1;
    int best_action = -1;

    for (size_t i = 0; i < root->children.size(); i++) {
      int visits = root->children[i]->visit_count;
      if (visits > best_visits) {
        best_visits = visits;
        best_action = root->actions[i];
      }
    }
    if (best_action != -1)
      policy[best_action] = 1.0f;

  } else {
    // Softmax-ish with temp
    float sum = 0.0f;
    std::vector<float> visit_probs;

    for (size_t i = 0; i < root->children.size(); i++) {
      float count =
          std::pow((float)root->children[i]->visit_count, 1.0f / temperature);
      visit_probs.push_back(count);
      sum += count;
    }

    if (sum > 0) {
      for (size_t i = 0; i < root->children.size(); i++) {
        policy[root->actions[i]] = visit_probs[i] / sum;
      }
    }
  }

  return policy;
}

int MCTS::get_root_visit_count() const {
  if (root)
    return root->visit_count;
  return 0;
}

} // namespace checkers
