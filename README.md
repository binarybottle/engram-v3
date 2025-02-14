# engram-v3
Arno's Engram v3 ("Engram") is a method for optimizing keyboard layouts 
for touch typing in English based on a study of key-pair typing preference data.
===================================================================

https://github.com/binarybottle/engram-v3.git

Author: Arno Klein (binarybottle.com)

This script uses a branch and bound algorithm to find optimal positions for items 
and item pairs by jointly considering two scoring components: 
item/item_pair scores and position/position_pair scores.

The primary intended use-case is keyboard layout optimization for touch typing, where:
  - Items and item-pairs correspond to letters and bigrams.
  - Positions and position-pairs correspond to keys and key-pairs.  
  - Item scores and item-pair scores correspond to frequency of occurrence 
    of letters and bigrams in a given language.
  - Position scores and position-pair scores correspond to measures of speed or comfort  
    when typing single keys or pairs of keys (in a given language).
  


## Context
This software optimizes letter arrangements on a keyboard.
It scores layouts based on bigram and letter frequencies (in English) 
and corresponding key-pair and single-key typing comfort scores:

    bigram_component = (bigram1_frequency * keypair1_comfort) + 
                       (bigram2_frequency * keypair2_comfort) / 2

    letter_component = letter_frequency * key_comfort

    score = (bigram_weight * bigram_component) + 
            (letter_weight * letter_component)

The typing comfort scores were estimated by a Bayesian learning method
(https://github.com/binarybottle/bigram_typing_preferences_to_comfort_scores)
applied to key-pair typing preference data collected from hundreds of participants 
as part of the Bigram Typing Preference Study (https://github.com/binarybottle/bigram_typing_preference_study).

## Description
The main script optimizes keyboard layouts by jointly considering typing comfort 
and letter/bigram frequencies. It uses a branch and bound algorithm to find optimal 
letter placements while staying within memory constraints.

When configuring a set of letters to optimally arrange within a set of keys,
you can assign a subset of the letters to be constrained within a subset of the keys. 
It would then run in two phases: 
- Phase 1 finds all possible arrangements of the letters to be constrained, 
  and sorts these according to the scoring criteria above.
- Phase 2 optimally arranges the remaining letters within the remaining keys.
If no constraints are specified, the code effectively runs in Phase 2 mode.

Core Components:

1. Scoring System:
   - Combines comfort scores for individual keys and key pairs
   - Weights letter and bigram frequencies from language analysis
   - Considers hand alternation through left/right hand assignments
   - Normalizes all scores to 0-1 range for consistent weighting

2. Branch and Bound Optimization:
   - Calculates exact scores for placed letters
   - Uses provable upper bounds for unplaced letters
   - Prunes branches that cannot exceed best known solution
   - Maintains optimality while reducing search space

3. Memory Management:
   - Estimates memory requirements before execution
   - Uses numba-optimized array operations
   - Monitors memory usage during search
   - Provides early stopping with best found solutions

4. Layout Evaluation:
   - Calculates separate letter and bigram scores
   - Considers both directions for each bigram
   - Supports arbitrary letter subsets and key positions
   - Provides detailed scoring breakdowns

Input:

- Configuration file (config.yaml) specifying:
  - Letters to optimize
  - Available key positions
  - Scoring weights
  - Comfort score files
- Pre-computed comfort scores for keys and key pairs
- Language statistics (letter and bigram frequencies)

Output:

- Top N scoring layouts with:
  - Letter-to-key mappings
  - Total scores
  - Unweighted letter and bigram scores
- Visual keyboard layout representations including:
  - ASCII art visualization of the layout
  - Clear marking of constrained positions
- Detailed CSV results with:
  - Configuration parameters
  - Search space statistics
  - Complete scoring breakdown
- Progress updates during execution:
  - Nodes processed per second
  - Memory usage
  - Estimated time remaining
