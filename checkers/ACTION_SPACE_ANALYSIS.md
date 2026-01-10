# Current Action Space Analysis

## Current Implementation: **Dynamic/Variable Action Space**

### How It Works

1. **Network Output**: Policy head outputs 150 logits (POLICY_SIZE = 150)

2. **Move Generation**: Legal moves are computed dynamically for each position
   - Could be 5 moves, 20 moves, 40 moves, etc.
   - Varies based on board position

3. **Action Mapping**:
   ```python
   legal_moves = game.get_legal_moves()  # e.g., [Move(5→10), Move(7→12), Move(15→20)]

   # Action indices are just positional:
   action_0 = legal_moves[0]  # Move(5→10)
   action_1 = legal_moves[1]  # Move(7→12)
   action_2 = legal_moves[2]  # Move(15→20)
   actions_3_to_149 = masked with -inf
   ```

4. **Legal Move Masking**:
   ```python
   legal_mask[0:len(legal_moves)] = True   # First N actions valid
   legal_mask[len(legal_moves):150] = False  # Rest are masked
   masked_logits[~legal_mask] = -inf
   ```

### Key Characteristic

**The same action index means DIFFERENT moves in different positions!**

**Position A:**
- legal_moves = [Move(5→10), Move(7→12), Move(15→20)]
- Action 0 = Move from square 5 to square 10
- Action 1 = Move from square 7 to square 12

**Position B:**
- legal_moves = [Move(3→8), Move(9→14)]
- Action 0 = Move from square 3 to square 8  ← DIFFERENT move!
- Action 1 = Move from square 9 to square 14  ← DIFFERENT move!

### Advantages

✓ Simple to implement
✓ Works for any number of legal moves
✓ Handles captures, chains, promotions naturally
✓ No need to design complex action encoding

### Disadvantages

✗ Network can't learn move patterns across positions (action 5 isn't consistent)
✗ Policy head doesn't learn "moving to square 23 is good" - only learns relative preferences
✗ No transfer learning for specific move types
✗ Harder for network to generalize tactical patterns

---

## Alternative: Fixed Action Space

### Concept

Each action index represents a **specific (from_square, to_square) pair** consistently across ALL positions.

### Example Encoding

For 10x10 checkers (50 playable squares):

```
Action encoding: from_square * 50 + to_square

Action 0   = Square 0 → Square 0 (always illegal)
Action 1   = Square 0 → Square 1
Action 5   = Square 0 → Square 5
Action 50  = Square 1 → Square 0
Action 51  = Square 1 → Square 1
...
Action 255 = Square 5 → Square 5  (example: specific diagonal move)
...
Action 2499 = Square 49 → Square 49
```

Total actions needed: 50 × 50 = **2,500 actions**

But most are invalid (can't move to non-diagonal, can't move backwards as man, etc.)

### Smarter Fixed Encoding

Only encode **actually possible move types**:

**Men (regular pieces):**
- Forward-left diagonal (1 square)
- Forward-right diagonal (1 square)
- Captures in 4 directions (jump 2 squares)
- Capture chains (complex)

**Kings:**
- 4 diagonal directions × up to 9 squares distance = 36 moves per king
- Flying king captures (even more complex)

Estimated: **~800-1500 actions** for all possible move patterns

### Fixed Space Advantages

✓ Network learns consistent move patterns
✓ "Action 523 = move from square 12 to square 17" is always true
✓ Policy head can learn tactical move preferences (e.g., "moves to center are good")
✓ Better generalization across similar positions
✓ Transfer learning: same move in different contexts

### Fixed Space Disadvantages

✗ Complex to implement (need careful move encoding/decoding)
✗ Larger policy head (800-2500 actions vs 150)
✗ Most actions illegal in any given position (more masking needed)
✗ Harder to handle capture chains (multi-square captures)
✗ More memory usage and computation

---

## Comparison

| Aspect | Dynamic (Current) | Fixed |
|--------|------------------|-------|
| Policy size | 150 | 800-2500 |
| Action meaning | Changes per position | Always same |
| Implementation | Simple | Complex |
| Network learning | Relative preferences | Absolute move patterns |
| Generalization | Limited | Better |
| Memory usage | Low | Higher |
| Captures/chains | Easy | Difficult |

---

## Why You Might Have Been Suggested Fixed Action Space

1. **AlphaGo/AlphaZero Chess/Shogi** use fixed action spaces
   - Chess: ~1968 actions (all possible from→to queen moves)
   - Go: 19×19 = 361 actions (one per board position)

2. **Better pattern learning**: Network can learn "this specific move type is good in similar positions"

3. **Transfer across positions**: Move from square 10→15 has consistent meaning

4. **Professional RL wisdom**: Fixed spaces generally train better for board games

---

## Recommendation

Your current dynamic space works fine for learning, but **for production/competition-level play**, a fixed action space would likely improve:

- **Generalization**: Network learns move patterns, not just position-specific preferences
- **Sample efficiency**: Better transfer learning across similar positions
- **Strategic understanding**: Can learn "controlling center" vs "relative move quality"

**Trade-off**: Implementation complexity and larger network vs better learning.

For your 10x10 International Draughts, the complexity comes from:
- Flying kings (variable distance moves)
- Capture chains (multi-square sequences)
- Mandatory captures

A hybrid approach could work:
- Fixed encoding for simple moves (men forward, king moves)
- Separate encoding for capture chains (challenging but doable)
