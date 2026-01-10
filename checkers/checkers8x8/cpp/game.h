
#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace checkers {

// Constants
constexpr int BOARD_SIZE = 8;
constexpr int NUM_ACTIONS = 128; // 32 squares * 4 directions
constexpr int MAX_MOVES = 400;

// Bitboard utilities (inline for performance)
inline uint32_t get_bit(uint32_t board, int square) {
  return (board >> square) & 1;
}

inline uint32_t set_bit(uint32_t board, int square) {
  return board | (1U << square);
}

inline uint32_t clear_bit(uint32_t board, int square) {
  return board & ~(1U << square);
}

inline int count_set_bits(uint32_t n) {
  int count = 0;
  while (n > 0) {
    n &= (n - 1);
    count++;
  }
  return count;
}

// Forward declarations for coordinate lookup
extern const int SQUARE_TO_ROW[32];
extern const int SQUARE_TO_COL[32];

inline uint32_t flip_bitboard(uint32_t board) {
  // Flip board using actual coordinates - matches Python implementation
  // A piece at (row, col) becomes (7-row, 7-col)
  // This is a 180-degree rotation

  uint32_t result = 0;
  for (int square = 0; square < 32; square++) {
    if (get_bit(board, square)) {
      int row = SQUARE_TO_ROW[square];
      int col = SQUARE_TO_COL[square];

      // Flip both row and column (180-degree rotation)
      int flipped_row = 7 - row;
      int flipped_col = 7 - col;

      // Convert back to square index
      // Dark squares have (row + col) odd
      if ((flipped_row + flipped_col) % 2 == 1) {
        int flipped_square = flipped_row * 4 + flipped_col / 2;
        if (flipped_square >= 0 && flipped_square < 32) {
          result = set_bit(result, flipped_square);
        }
      }
    }
  }
  return result;
}

struct Move {
  int from_square;
  int to_square;
  std::vector<int> captured_squares;
  bool promotes_to_king;
  bool is_jump;

  bool operator==(const Move &other) const {
    return from_square == other.from_square && to_square == other.to_square &&
           captured_squares == other.captured_squares;
  }
};

class Game {
public:
  // State
  uint32_t player_men;
  uint32_t player_kings;
  uint32_t opponent_men;
  uint32_t opponent_kings;

  int current_player; // 1 or 2
  int move_count;

  // Position history for draw detection: (p_men, p_kings, o_men, o_kings)
  // NOTE: Positions are already stored from current player's perspective due to
  // board flipping/swapping after each move, so current_player is NOT needed
  using PositionKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
  std::vector<PositionKey> position_history;

  // Constructors
  Game();
  Game(const Game &other); // Copy constructor

  // Core methods
  std::vector<Move> get_legal_moves() const;
  void make_move(const Move &move);

  // Action interface (for Neural Net)
  std::vector<int> get_legal_actions() const;
  bool make_action(int action); // Returns false if invalid

  // Game status
  bool is_terminal() const;
  float get_result() const; // 1.0 (win), -1.0 (loss), 0.0 (draw/continue)

  // Helper
  Game clone() const;
  void print_board() const;
  std::vector<float> to_neural_input() const; // Returns flat vector (8*8*8)

private:
  void get_simple_moves(int square, bool is_king,
                        std::vector<Move> &moves) const;
  void get_jumps(int square, uint32_t current_men, uint32_t current_kings,
                 uint32_t opp_men, uint32_t opp_kings, bool is_king,
                 std::vector<int> &current_path,
                 std::vector<int> &current_captured,
                 std::vector<Move> &moves) const;

  void update_position_hash();
};

} // namespace checkers
