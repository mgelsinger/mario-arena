# Mario Arena Scoring System

## Overview

The Mario Arena scoring system provides a standardized way to compare RL agents on Super Mario Bros. The scoring formula is designed to prioritize:

1. **Level completion** over partial progress
2. **Further progress** over less progress
3. **Efficiency** (fewer steps) over inefficiency

This ensures that agents are ranked in a way that reflects meaningful game performance.

## Scoring Formula

### Single Episode Score

For a single episode, the arena score is computed as:

```
arena_score = completion_bonus + progress_score - efficiency_penalty + minor_bonuses

Where:
- completion_bonus = 1,000,000 if completed, else 0
- progress_score = (world × 10,000) + (stage × 1,000) + max_x_pos
- efficiency_penalty = total_steps × 0.1
- minor_bonuses = (coins × 100) + (time_remaining × 10)
```

### Component Breakdown

#### 1. Completion Bonus (1,000,000 points)

The largest component of the score. An agent that completes a level will **always** score higher than one that doesn't, regardless of other factors.

**Example:**
- Agent A: Completes level in 500 steps → Gets 1,000,000 bonus
- Agent B: Gets 99% through level → Gets 0 bonus
- Agent A scores higher

#### 2. Progress Score

Rewards forward progress through the game:

- **World**: 10,000 points per world number
  - World 1 = 10,000 points
  - World 2 = 20,000 points
  - World 8 = 80,000 points

- **Stage**: 1,000 points per stage number
  - Stage 1 = 1,000 points
  - Stage 2 = 2,000 points
  - Stage 4 = 4,000 points

- **X-position**: 1 point per unit of max x-position reached
  - Typical level is ~3,000 units long
  - Getting to x=2,000 = 2,000 points

**Example:**
```
World 1, Stage 1, x=2,456
progress_score = (1 × 10,000) + (1 × 1,000) + 2,456 = 13,456
```

#### 3. Efficiency Penalty (0.1 points per step)

A small penalty for taking many steps. This encourages efficient play but is weighted much less than completion or progress.

**Example:**
- 300 steps → penalty of 30 points
- 1,000 steps → penalty of 100 points

This prevents two agents with identical progress from being scored the same if one took significantly longer.

#### 4. Minor Bonuses

Small bonuses that provide tiebreakers but don't significantly affect overall ranking:

- **Coins**: 100 points per coin collected
  - Encourages collecting coins
  - Max ~50 coins per level = ~5,000 points

- **Time remaining**: 10 points per second (only if level completed)
  - Typical level has 400 seconds
  - Finishing with 250 seconds left = 2,500 points

## Example Scores

### Example 1: Successful Completion

```
Level: 1-1
Completed: Yes
Max x-position: 3,266
Steps: 342
Coins: 15
Time remaining: 245 seconds

Calculation:
- Completion bonus: 1,000,000
- Progress: (1 × 10,000) + (1 × 1,000) + 3,266 = 14,266
- Efficiency penalty: 342 × 0.1 = 34
- Coin bonus: 15 × 100 = 1,500
- Time bonus: 245 × 10 = 2,450

Arena Score: 1,000,000 + 14,266 - 34 + 1,500 + 2,450 = 1,018,182
```

### Example 2: Partial Progress

```
Level: 1-1
Completed: No
Max x-position: 1,456
Steps: 892
Coins: 7
Time remaining: 0 (timed out)

Calculation:
- Completion bonus: 0
- Progress: (1 × 10,000) + (1 × 1,000) + 1,456 = 12,456
- Efficiency penalty: 892 × 0.1 = 89
- Coin bonus: 7 × 100 = 700
- Time bonus: 0

Arena Score: 0 + 12,456 - 89 + 700 + 0 = 13,067
```

### Example 3: Advanced Level

```
Level: 3-2
Completed: Yes
Max x-position: 2,888
Steps: 415
Coins: 22
Time remaining: 198 seconds

Calculation:
- Completion bonus: 1,000,000
- Progress: (3 × 10,000) + (2 × 1,000) + 2,888 = 34,888
- Efficiency penalty: 415 × 0.1 = 42
- Coin bonus: 22 × 100 = 2,200
- Time bonus: 198 × 10 = 1,980

Arena Score: 1,000,000 + 34,888 - 42 + 2,200 + 1,980 = 1,039,026
```

## Multi-Episode Aggregation

When evaluating over multiple episodes, we take the **best** episode score as the representative score, but also track:

- Average success rate across all episodes
- Average steps taken
- Average max x-position
- Standard deviation of scores

### Why Best-of-N?

Using the best score from N episodes:
- Rewards agents that can occasionally achieve great performance
- Reduces penalty for exploration/randomness
- Matches how humans would naturally compare agents ("can it beat the level?")

**Example:**
```
10 episodes:
- Episode 1: 1,018,182 (completed)
- Episode 2: 12,456 (failed at x=1,456)
- Episode 3: 1,016,890 (completed)
- ...
- Episode 10: 13,892 (failed at x=2,892)

Arena Score: 1,018,182 (best episode)
Success Rate: 3/10 = 30%
```

## Full-Game Scoring (Phase 5)

For full-game runs (1-1 through 8-4), the scoring extends:

```
arena_score = completion_bonus + progress_score - efficiency_penalty

Where:
- completion_bonus = 10,000,000 if full game completed
- progress_score = Σ(level_scores) for each level reached
- efficiency_penalty = total_steps × 0.1 (across all levels)
```

### Level-by-Level Breakdown

A full-game run tracks:
- Which levels were completed
- How far the agent got before game over
- Total steps across all levels
- Lives remaining

**Example:**
```
Completed: 1-1, 1-2, 1-3, 1-4 (world 1 complete)
Failed at: 2-1 (died at x=1,200)
Total steps: 2,456
Lives lost: 2

Completion bonus: 0 (didn't beat full game)
Progress score:
  - Reached world 2: 20,000
  - Reached stage 1: 1,000
  - Max x in 2-1: 1,200
  - Total: 22,200
Efficiency penalty: 2,456 × 0.1 = 246

Arena Score: 22,200 - 246 = 21,954
```

## Racing and Leaderboards

When comparing multiple models (Phase 5):

1. Each model is evaluated with the same protocol (same seed, same number of episodes)
2. Models are ranked by their arena scores
3. Tiebreakers:
   - Higher success rate
   - Higher average score
   - Lower average steps

### Leaderboard Display

```
===========================================
        MARIO ARENA LEADERBOARD
===========================================
Rank  Model Name        Score      Success  Avg Steps
---------------------------------------------------------
1     PPO-v3-1M        1,018,182   90%      328
2     PPO-v2-500k      1,016,450   85%      345
3     PPO-v1-100k        15,234    10%      892
===========================================
```

## Design Rationale

### Why These Weights?

The scoring weights were chosen to ensure:

1. **Completion dominates**: The 1M point bonus ensures completed levels always beat incomplete ones
2. **Progress matters**: Getting further is always better (10k/1k/1 point scaling)
3. **Efficiency is minor**: We don't want to penalize exploration too heavily (only 0.1 per step)
4. **Bonuses are tiebreakers**: Coins and time provide differentiation but don't dominate

### Alternative Scoring Systems

Future versions might include:

- **Speed-run scoring**: Heavily weight time remaining
- **Coin-collection scoring**: Heavily weight coins collected
- **Survival scoring**: Weight lives remaining more heavily
- **Style scoring**: Reward jumps, enemy defeats, etc.

These could be implemented as alternative scoring modes while keeping the standard arena score as the default.

## Implementation

The scoring system is implemented in [mario_arena/training/scoring.py](../mario_arena/training/scoring.py).

Key functions:
- `compute_arena_score()`: Compute score for a single episode
- `aggregate_scores()`: Aggregate scores from multiple episodes
- `format_score_display()`: Format scores for human-readable output

## References

- Inspired by speedrunning scoring systems
- Balances completion, progress, and efficiency
- Designed to be easily understood and debugged

---

**Note**: This scoring system will be implemented in Phase 4. The formulas and examples above represent the planned design.
