
#include "game.h"
#include <fstream>

namespace checkers {

// Board Lookups
// 0-31 squares mapping to Row/Col (0-7, 0-7)
// Only 32 valid squares on the board (black squares)
const int SQUARE_TO_ROW[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                             4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7};

const int SQUARE_TO_COL[] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6,
                             1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6};

// Neighbor Lookups (4 directions: 0=NE, 1=NW, 2=SE, 3=SW)
// -1 means invalid (off board)
// Directions relative to Black (moving DOWN the board usually, but here
// 'Forward' is UP regarding row index processing logic often, but let's stick
// to board geometry) Actually in our bitboard: Row 0 is top (0-3). Row 7 is
// bottom (28-31). Player 1 starts at bottom (20-31), moves UP (negative index
// change). But wait, the python code says "Current player ALWAYS at bottom". So
// "Forward" is always "Up" the board (towards 0). Let's define neighbors based
// on board geometry.

// Hardcoded neighbors for 8x8 (32 squares)
// Directions: 0: Up-Right, 1: Up-Left, 2: Down-Right, 3: Down-Left
// Invalid = -1
const int NEIGHBORS[32][4] = {
    {-1, -1, 4, 5},   {-1, -1, 5, 6},
    {-1, -1, 6, 7},   {-1, -1, 7, -1}, // Row 0 (0-3)
    {0, -1, 8, 9},    {0, 1, 9, 10},
    {1, 2, 10, 11},   {2, 3, 11, -1}, // Row 1 (4-7)
    {-1, 4, 12, 13},  {4, 5, 13, 14},
    {5, 6, 14, 15},   {6, 7, 15, -1}, // Row 2 (8-11)
    {8, -1, 16, 17},  {8, 9, 17, 18},
    {9, 10, 18, 19},  {10, 11, 19, -1}, // Row 3 (12-15)
    {-1, 12, 20, 21}, {12, 13, 21, 22},
    {13, 14, 22, 23}, {14, 15, 23, -1}, // Row 4 (16-19)
    {16, -1, 24, 25}, {16, 17, 25, 26},
    {17, 18, 26, 27}, {18, 19, 27, -1}, // Row 5 (20-23)
    {-1, 20, 28, 29}, {20, 21, 29, 30},
    {21, 22, 30, 31}, {22, 23, 31, -1}, // Row 6 (24-27)
    {24, -1, -1, -1}, {24, 25, -1, -1},
    {25, 26, -1, -1}, {26, 27, -1, -1} // Row 7 (28-31)
};
// Note: The logic above:
// Even rows (0, 2, 4, 6) are shifted right (cols 1, 3, 5, 7)
// Odd rows (1, 3, 5, 7) are shifted left (cols 0, 2, 4, 6)
// Moving "Up" (smaller index) from Row 1 (square 4, col 0):
//   - Up-Right (NE): square 0
//   - Up-Left (NW): invalid (-1)
// Let's rely on standard logic.
// In "Current Player at Bottom" perspective:
// Forward is MINUS (smaller index).
// Forward-Left: usually -4 or -5
// Forward-Right: usually -3 or -4

// Actually, let's just use the `NEIGHBORS` table which I derived manually
// above. Row 0 is indices 0-3. Row 1 is 4-7. Square 4 (Row 1, Col 0). UR -> 0.
// UL -> -1. DR -> 8. DL -> 9. (Wait, 4+4=8, 4+5=9). Correct. Square 5 (Row 1,
// Col 2). UR -> 0 (wait. 5-5=0? No. 5 is connected to 0 and 1).
//   5 is Row 1 Col 2.
//   Row 0: 0(1), 1(3), 2(5), 3(7).
//   5 connects to 0 (UR? No, 0 is Left of 5) and 1 (Right of 5?).
//   Let's check `get_neighbor` implementation from python to be 100% sure or
//   derive carefully. Python: Even row (0,2..): UR(-3), UL(-4), DR(+5), DL(+4)
//   Odd row (1,3..): UR(-4), UL(-5), DR(+4), DL(+3)
// My table above:
// Square 4 (Row 1 - Odd): UR(0=-4), UL(-1), DR(8=+4), DL(9=+5 -- WAIT).
//   Odd Row: 4 -> UL is -1. UR is 0 (4-4). DL is 8 (4+4). DR is 9 (4+5).
//   Wait, visual check.
//   Row 1: . 4 . 5 . 6 . 7
//   Row 2: 8 . 9 . 10 . 11
//   4 is at Col 0. DL is N/A (would be -1 col). DR is 8. Correct?
//   No, 4 is connected to 8 (which is col 1). So 4 (col 0) -> 8 (col 1) is
//   Down-Right. So for Odd Row 4: DR (+4) -> 8. DL (N/A). UR (-4) -> 0. UL
//   (N/A). Table for 4: {-1(UL), 0(UR), 8(DL?), 9(DR?) }. Let's stick to:
//   0=Forward-Right, 1=Forward-Left, 2=Backward-Right, 3=Backward-Left.
//   "Forward" = UP = Negative.
//   Row 1 (Odd): 4. FR(-4)=0. FL(-5)=-1. BR(+4)=8. BL(+3)=? invalid?
//   Actually 4 is connected to 8 (Col 1). 8 is > 4. So it's Back.
//   4(0) -> 8(1). Right is increasing col. So Back-Right.

// RE-GENERATING NEIGHBORS TABLE CORRECTLY
// Directions: 0=FR, 1=FL, 2=BR, 3=BL.
// Forward = Upper Row (Index - 4ish). Backward = Lower Row (Index + 4ish).
// Right = Higher Col. Left = Lower Col.

// Even Row (0, 2...): Offset: -3(FR), -4(FL), +5(BR), +4(BL)
// Odd Row (1, 3...): Offset: -4(FR), -5(FL), +4(BR), +3(BL)

// Examples:
// Sq 4 (Odd): FR(0), FL(-1), BR(8), BL(7 - wait, 4 is col 0, can't go left. 7
// is row 1 col 6. wrapping!). Need bounds checking.

// Helper to generate at runtime or use corrected logic.
int get_neighbor(int square, int dir) {
  if (square < 0 || square >= 32)
    return -1;

  int row = square / 4;
  bool odd_row = (row % 2 != 0);

  int offset;
  switch (dir) {
  case 0: // FR (Forward-Right) NE
    offset = odd_row ? -4 : -3;
    break;
  case 1: // FL (Forward-Left) NW
    offset = odd_row ? -5 : -4;
    break;
  case 2: // BR (Backward-Right) SE
    offset = odd_row ? +4 : +5;
    break;
  case 3: // BL (Backward-Left) SW
    offset = odd_row ? +3 : +4;
    break;
  default:
    return -1;
  }

  int target = square + offset;
  if (target < 0 || target >= 32)
    return -1;

  // Check for column wrapping
  int target_row = target / 4;
  int expected_row_diff = (dir <= 1) ? -1 : 1;
  if (target_row != row + expected_row_diff)
    return -1;

  return target;
}

// Action Decoding
// Action is 0-127.
// from_square = action // 4
// direction = action % 4
// Direction mapping must match `get_neighbor`.
// In Python: 0=NE(FR), 1=NW(FL), 2=SE(BR), 3=SW(BL). Matches our
// `get_neighbor(dir)`.

Game::Game() {
  // Initial State
  // Player 1 (Bottom, 20-31)
  player_men = 0xFFF00000; // 1111... at top bits
  player_kings = 0;
  // Opponent (Top, 0-11)
  opponent_men = 0x00000FFF; // 1111... at bottom bits
  opponent_kings = 0;

  current_player = 1;
  move_count = 0;
  update_position_hash(); // Add initial position to history
}

Game::Game(const Game &other) {
  player_men = other.player_men;
  player_kings = other.player_kings;
  opponent_men = other.opponent_men;
  opponent_kings = other.opponent_kings;
  current_player = other.current_player;
  move_count = other.move_count;
  position_history = other.position_history;
}

Game Game::clone() const { return Game(*this); }

void Game::update_position_hash() {
  position_history.push_back(
      {player_men, player_kings, opponent_men, opponent_kings});
}

// ... intermediate code ...

std::vector<Move> Game::get_legal_moves() const {
  std::vector<Move> jumps;

  // 1. Check for Jumps (forced)
  uint32_t pieces = player_men | player_kings;
  for (int i = 0; i < 32; i++) {
    if ((pieces >> i) & 1) {
      bool is_king = (player_kings >> i) & 1;
      std::vector<int> path = {i};
      std::vector<int> captured;
      get_jumps(i, player_men, player_kings, opponent_men, opponent_kings,
                is_king, path, captured, jumps);
    }
  }

  if (!jumps.empty()) {
    // Filter for longest jumps if that's the rule?
    // Standard American checkers: Jump is forced, but not max length.
    // Just jumping is mandatory. So return all jumps.
    return jumps;
  }

  // 2. Simple Moves
  std::vector<Move> simple_moves;
  for (int i = 0; i < 32; i++) {
    if ((pieces >> i) & 1) {
      bool is_king = (player_kings >> i) & 1;
      get_simple_moves(i, is_king, simple_moves);
    }
  }

  return simple_moves;
}

void Game::get_simple_moves(int square, bool is_king,
                            std::vector<Move> &moves) const {
  // Directions to check
  // Men: Forward (0, 1)
  // Kings: All (0, 1, 2, 3)
  std::vector<int> dirs;
  if (is_king)
    dirs = {0, 1, 2, 3};
  else
    dirs = {0, 1};

  uint32_t occupied = player_men | player_kings | opponent_men | opponent_kings;

  for (int dir : dirs) {
    int target = get_neighbor(square, dir);
    if (target != -1 && !((occupied >> target) & 1)) {
      // Empty square
      Move m;
      m.from_square = square;
      m.to_square = target;
      m.captured_squares = {};
      m.is_jump = false;

      // Promotion?
      // Player is always moving UP (towards 0).
      // Promotion happens at Row 0 (squares 0-3).
      m.promotes_to_king = (!is_king && target < 4);
      moves.push_back(m);
    }
  }
}

void Game::get_jumps(int square, uint32_t current_men, uint32_t current_kings,
                     uint32_t opp_men, uint32_t opp_kings, bool is_king,
                     std::vector<int> &current_path,
                     std::vector<int> &current_captured,
                     std::vector<Move> &moves) const {

  std::vector<int> dirs;
  if (is_king)
    dirs = {0, 1, 2, 3};
  else
    dirs = {0, 1};

  bool found_jump = false;

  for (int dir : dirs) {
    int mid = get_neighbor(square, dir);
    if (mid == -1)
      continue;

    // Mid must have opponent piece
    if (!((opp_men | opp_kings) >> mid & 1))
      continue;

    // Already captured in this sequence?
    bool already_captured = false;
    for (int c : current_captured)
      if (c == mid)
        already_captured = true;
    if (already_captured)
      continue;

    int dest = get_neighbor(mid, dir);
    if (dest == -1)
      continue;

    // Dest must be empty (or start square if cycle? - checkers no cycles
    // usually) Check current board state (initial pieces - captured + moved
    // piece) To simplify: check if dest is occupied in ORIGINAL board, EXCEPT
    // start square. Actually, in recursive jumps, we occupy 'square' (current
    // pos). Is dest occupied by ANY piece? original pieces: (p_men | p_kings |
    // o_men | o_kings)
    // - captured pieces
    // - original start square (moved)
    // + current square

    bool dest_busy = false;

    // Check static occupancy first (fast)
    uint32_t all_static =
        (player_men | player_kings | opponent_men | opponent_kings);
    if ((all_static >> dest) & 1) {
      // Might be start square?
      if (dest == current_path[0]) {
        // Circular jump? Allowed? usually no.
        dest_busy = true;
      } else {
        dest_busy = true;
      }
    }

    // If 'captured' contains dest, then it's actually empty now!
    for (int c : current_captured)
      if (c == dest)
        dest_busy = false;

    if (dest_busy)
      continue;

    // Valid jump step!
    found_jump = true;

    current_path.push_back(dest);
    current_captured.push_back(mid);

    // Promote?
    bool promotes = (!is_king && dest < 4);
    bool now_king = is_king || promotes;

    // Recursion
    // Stop recursion if promoted (usually turn ends on promotion)
    // OR continue if not just promoted or rule says so.
    // Standard American rules: Capture sequence CONTINUES even if passing King
    // row, BUT if it STOPS there it becomes King? Actually: "If a man reaches
    // the kings row... the move terminates".
    if (promotes) {
      Move m;
      m.from_square = current_path[0];
      m.to_square = dest;
      m.captured_squares = current_captured;
      m.promotes_to_king = true;
      m.is_jump = true;
      moves.push_back(m);
    } else {
      // Continue jumping
      get_jumps(dest, current_men, current_kings, opp_men, opp_kings, now_king,
                current_path, current_captured, moves);
    }

    current_path.pop_back();
    current_captured.pop_back();
  }

  // Leaf node of jump sequence
  if (!found_jump && current_path.size() > 1) {
    Move m;
    m.from_square = current_path[0];
    m.to_square = current_path.back();
    m.captured_squares = current_captured;
    m.promotes_to_king =
        (!is_king &&
         current_path.back() < 4); // Should have been caught above if triggered
    m.is_jump = true;
    moves.push_back(m);
  }
}

void Game::make_move(const Move &move) {
  // 1. Check if moving piece is a king BEFORE removing it
  bool moving_king = get_bit(player_kings, move.from_square);

  // 2. Remove piece from start position
  if (moving_king) {
    player_kings = clear_bit(player_kings, move.from_square);
  } else {
    player_men = clear_bit(player_men, move.from_square);
  }

  // 3. Remove captured pieces
  for (int cap : move.captured_squares) {
    if (get_bit(opponent_men, cap)) {
      opponent_men = clear_bit(opponent_men, cap);
    } else {
      opponent_kings = clear_bit(opponent_kings, cap);
    }
  }

  // 4. Place piece at destination
  // Piece becomes/stays king if: it was already a king OR it promotes
  if (move.promotes_to_king || moving_king) {
    player_kings = set_bit(player_kings, move.to_square);
  } else {
    player_men = set_bit(player_men, move.to_square);
  }

  // 5. Flip and Swap for perspective switching
  player_men = flip_bitboard(player_men);
  player_kings = flip_bitboard(player_kings);
  opponent_men = flip_bitboard(opponent_men);
  opponent_kings = flip_bitboard(opponent_kings);

  std::swap(player_men, opponent_men);
  std::swap(player_kings, opponent_kings);

  current_player = 3 - current_player;
  move_count++;
  update_position_hash();
}

std::vector<int> Game::get_legal_actions() const {
  std::vector<Move> moves = get_legal_moves();
  std::set<int> actions;
  for (const auto &m : moves) {
    // Encode: from * 4 + dir
    // For simple move: use overall direction
    // For jump: use direction of FIRST STEP (critical for zig-zag jumps!)

    int r1 = SQUARE_TO_ROW[m.from_square];
    int c1 = SQUARE_TO_COL[m.from_square];

    int r2, c2;
    if (m.is_jump && !m.captured_squares.empty()) {
      // For jumps: determine first step direction by looking at first captured piece
      // The first captured piece is at position (r_cap, c_cap)
      // The first step goes over this piece
      int first_captured = m.captured_squares[0];
      int r_cap = SQUARE_TO_ROW[first_captured];
      int c_cap = SQUARE_TO_COL[first_captured];

      // First step direction: from (r1, c1) towards (r_cap, c_cap)
      // Since we jump OVER the captured piece, direction is same as to the captured piece
      r2 = r_cap;
      c2 = c_cap;
    } else {
      // For simple moves: use destination
      r2 = SQUARE_TO_ROW[m.to_square];
      c2 = SQUARE_TO_COL[m.to_square];
    }

    int dr = r2 - r1;
    int dc = c2 - c1;

    int dir_idx = -1;
    if (dr < 0 && dc > 0)
      dir_idx = 0; // NE (Up-Right)
    else if (dr < 0 && dc < 0)
      dir_idx = 1; // NW (Up-Left)
    else if (dr > 0 && dc > 0)
      dir_idx = 2; // SE (Down-Right)
    else if (dr > 0 && dc < 0)
      dir_idx = 3; // SW (Down-Left)

    if (dir_idx != -1) {
      actions.insert(m.from_square * 4 + dir_idx);
    }
  }
  return std::vector<int>(actions.begin(), actions.end());
}

bool Game::make_action(int action) {
  int from_sq = action / 4;
  int dir_idx = action % 4;

  std::vector<Move> moves = get_legal_moves();
  for (const auto &m : moves) {
    if (m.from_square != from_sq)
      continue;

    // Determine direction to match against
    // MUST match the encoding logic in get_legal_actions()!
    int r1 = SQUARE_TO_ROW[m.from_square];
    int c1 = SQUARE_TO_COL[m.from_square];

    int r2, c2;
    if (m.is_jump && !m.captured_squares.empty()) {
      // For jumps: use direction of first step (to first captured piece)
      int first_captured = m.captured_squares[0];
      r2 = SQUARE_TO_ROW[first_captured];
      c2 = SQUARE_TO_COL[first_captured];
    } else {
      // For simple moves: use destination
      r2 = SQUARE_TO_ROW[m.to_square];
      c2 = SQUARE_TO_COL[m.to_square];
    }

    int dr = r2 - r1;
    int dc = c2 - c1;

    // Validate direction matches
    bool match = false;
    if (dir_idx == 0)
      match = (dr < 0 && dc > 0);
    else if (dir_idx == 1)
      match = (dr < 0 && dc < 0);
    else if (dir_idx == 2)
      match = (dr > 0 && dc > 0);
    else if (dir_idx == 3)
      match = (dr > 0 && dc < 0);

    if (match) {
      make_move(m);
      return true;
    }
  }
  return false;
}

bool Game::is_terminal() const {
  if (get_legal_moves().empty())
    return true;

  // Check 3-fold repetition
  PositionKey current = {player_men, player_kings, opponent_men, opponent_kings};
  int count = 0;
  for (const auto &key : position_history) {
    if (key == current) {
      count++;
      if (count >= 3)
        return true;
    }
  }

  if (move_count >= MAX_MOVES)
    return true;

  return false;
}

float Game::get_result() const {
  if (!is_terminal())
    return 0.0;
  if (get_legal_moves().empty())
    return -1.0;
  return 0.0; // Draw
}

std::vector<float> Game::to_neural_input() const {
  std::vector<float> planes(8 * 8 * 8, 0.0f);

  auto set_plane = [&](int plane_idx, uint32_t board) {
    for (int i = 0; i < 32; i++) {
      if ((board >> i) & 1) {
        int r = SQUARE_TO_ROW[i];
        int c = SQUARE_TO_COL[i];
        planes[plane_idx * 64 + r * 8 + c] = 1.0f;
      }
    }
  };

  set_plane(0, player_men);
  set_plane(1, player_kings);
  set_plane(2, opponent_men);
  set_plane(3, opponent_kings);

  // Plane 4: Legal moves (destinations) -- simplified (just 'to' squares?)
  // Python code: "Legal move destinations"
  // Let's call get_legal_moves
  std::vector<Move> moves = get_legal_moves();
  for (const auto &m : moves) {
    int r = SQUARE_TO_ROW[m.to_square];
    int c = SQUARE_TO_COL[m.to_square];
    planes[4 * 64 + r * 8 + c] = 1.0f;
  }

  // Plane 5: Repetition
  PositionKey key = {player_men, player_kings, opponent_men, opponent_kings};
  int rep = 0;
  for (const auto &h : position_history) {
    if (h == key)
      rep++;
  }

  float rep_val = std::min(rep / 3.0f, 1.0f);
  std::fill(planes.begin() + 5 * 64, planes.begin() + 6 * 64, rep_val);

  // Plane 6: Move Count
  float mc_val = std::min(move_count / 100.0f, 1.0f);
  std::fill(planes.begin() + 6 * 64, planes.begin() + 7 * 64, mc_val);

  // Plane 7: Bias
  std::fill(planes.begin() + 7 * 64, planes.end(), 1.0f);

  return planes;
}

void Game::print_board() const {
  // Basic ASCII print
  // ...
}

} // namespace checkers
