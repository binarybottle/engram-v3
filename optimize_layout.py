from itertools import permutations
from multiprocessing import Pool, cpu_count
from math import factorial
from tqdm import tqdm
import numpy as np
import psutil
import itertools
from typing import List, Dict, Tuple, Optional
import time
from datetime import timedelta
import pandas as pd
import yaml
from pathlib import Path
import multiprocessing
import sys

from data.bigram_frequencies_english import (
    bigrams, bigram_frequencies,
    onegrams, onegram_frequencies
)

#--------------------------------------------------------------------
# Basic utility functions
#--------------------------------------------------------------------
def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_position_comfort_map(raw_comfort_scores: Dict[Tuple[str, str], float], config: dict) -> Dict[Tuple[str, str], float]:
    """Convert letter-based comfort scores to position-based comfort scores."""
    position_comfort = {}
    
    # Get all positions and create letter-to-position mapping
    right_keys = set()
    left_keys = set()
    default_letters = {}  # Map positions to their default letters
    
    for row in ['top', 'home', 'bottom']:
        for pos in config['layout']['positions']['right'][row]:
            right_keys.add(pos)
            default_letters[pos] = pos  # Use position as its default letter
            
        for pos in config['layout']['positions']['left'][row]:
            left_keys.add(pos)
            default_letters[pos] = pos  # Use position as its default letter
    
    # Create comfort scores for all position pairs
    for pos1 in right_keys | left_keys:
        for pos2 in right_keys | left_keys:
            if pos1 != pos2:
                # Check if positions are on the same side
                same_side = ((pos1 in right_keys and pos2 in right_keys) or 
                           (pos1 in left_keys and pos2 in left_keys))
                
                if same_side:
                    # Get comfort score from the corresponding letter pair
                    letter1 = default_letters[pos1]
                    letter2 = default_letters[pos2]
                    score = raw_comfort_scores.get((letter1, letter2),
                                                raw_comfort_scores.get((letter2, letter1), 0.0))
                    position_comfort[(pos1, pos2)] = score
                else:
                    position_comfort[(pos1, pos2)] = 0.0
    
    return position_comfort

def is_same_side(pos1: str, pos2: str) -> bool:
    """Check if two positions are on the same side of the keyboard."""
    on_right1 = pos1 in right_keys
    on_right2 = pos2 in right_keys
    on_left1 = pos1 in left_keys
    on_left2 = pos2 in left_keys
    
    same_side = (on_right1 and on_right2) or (on_left1 and on_left2)
    
    print(f"Debug - is_same_side: pos1={pos1} (right={on_right1}, left={on_left1}), "
          f"pos2={pos2} (right={on_right2}, left={on_left2}) -> {same_side}")
    
    return same_side

def get_comfort_scores(config: dict = None) -> Dict[Tuple[str, str], float]:
    """Load and process comfort scores for key positions."""
    if config is None:
        config = load_config()
    
    # Load raw bigram scores from CSV
    scores_file = Path(config['data']['bigram_scores_file'])
    df = pd.read_csv(scores_file)
    raw_bigram_scores = {}
    for _, row in df.iterrows():
        bigram = (row['first_char'], row['second_char'])
        raw_bigram_scores[bigram] = row['comfort_score']
    
    # Load raw key scores from CSV
    key_scores_file = Path(config['data']['key_scores_file'])
    key_df = pd.read_csv(key_scores_file)
    raw_key_scores = {}
    for _, row in key_df.iterrows():
        raw_key_scores[row['key']] = row['comfort_score']
    
    # Get keyboard sides and position mappings
    right_keys = set()
    left_keys = set()
    for row in ['top', 'home', 'bottom']:
        right_keys.update(config['layout']['positions']['right'][row])
        left_keys.update(config['layout']['positions']['left'][row])
    
    # Get right-to-left mappings
    right_to_left = config['layout']['position_mappings']['right_to_left']
    
    # Normalize scores to 0-1 range where 1 is best (least negative)
    min_score = min(min(raw_bigram_scores.values()), min(raw_key_scores.values()))
    max_score = max(max(raw_bigram_scores.values()), max(raw_key_scores.values()))
    
    def normalize_score(score):
        return 1 - ((score - max_score) / (min_score - max_score))
    
    # Create position-based comfort scores
    position_scores = {}
    
    # For each pair of positions
    for pos1 in (right_keys | left_keys):
        for pos2 in (right_keys | left_keys):
            if pos1 != pos2:
                # Check if positions are on the same side
                pos1_right = pos1 in right_keys
                pos2_right = pos2 in right_keys
                same_side = (pos1_right and pos2_right) or ((pos1 in left_keys) and (pos2 in left_keys))
                
                if same_side:
                    # If both positions are on the right, map to left equivalents
                    if pos1_right and pos2_right:
                        lookup_pos1 = right_to_left[pos1]
                        lookup_pos2 = right_to_left[pos2]
                    else:
                        lookup_pos1 = pos1
                        lookup_pos2 = pos2
                    
                    # Try both directions for the comfort score
                    raw_score = max(
                        raw_bigram_scores.get((lookup_pos1, lookup_pos2), float('-inf')),
                        raw_bigram_scores.get((lookup_pos2, lookup_pos1), float('-inf'))
                    )
                    if raw_score == float('-inf'):
                        print(f"Warning: No bigram score found for {lookup_pos1},{lookup_pos2}")
                        raw_score = 0
                    
                    position_scores[(pos1, pos2)] = normalize_score(raw_score)
                else:
                    # For alternating hands, use the average of individual key scores
                    pos1_lookup = right_to_left[pos1] if pos1_right else pos1
                    pos2_lookup = right_to_left[pos2] if pos2_right else pos2
                    
                    # Get and normalize individual key scores
                    score1 = normalize_score(raw_key_scores[pos1_lookup])
                    score2 = normalize_score(raw_key_scores[pos2_lookup])
                    
                    # Average the normalized scores
                    position_scores[(pos1, pos2)] = (score1 + score2) / 2
    
    # Print score summaries for verification
    print("\nScore summary:")
    print(f"Bigram scores range: {min(raw_bigram_scores.values()):.4f} to {max(raw_bigram_scores.values()):.4f}")
    print(f"Key scores range: {min(raw_key_scores.values()):.4f} to {max(raw_key_scores.values()):.4f}")
    print(f"Normalized scores range: {min(position_scores.values()):.4f} to {max(position_scores.values()):.4f}")
    
    return position_scores

def get_comfort_value(pos1: str, pos2: str, normalized_comfort: Dict[Tuple[str, str], float], 
                     right_keys: set, left_keys: set, config: dict) -> Tuple[float, bool]:
    """Get comfort score and same-side status for a position pair."""
    # Check if positions are on the same side
    pos1_right = pos1 in right_keys
    pos2_right = pos2 in right_keys
    same_side = (pos1_right and pos2_right) or ((pos1 in left_keys) and (pos2 in left_keys))
    
    if not same_side:
        # For alternating hands, calculate score based on individual key comfort
        right_to_left = config['layout']['position_mappings']['right_to_left']
        
        # Map each position to its left-side equivalent for scoring
        pos1_lookup = right_to_left[pos1] if pos1_right else pos1
        pos2_lookup = right_to_left[pos2] if pos2_right else pos2
        
        # Get raw scores from the preloaded normalized_comfort scores
        score1 = normalized_comfort.get((pos1_lookup, pos1_lookup), 0)  # Use self-bigram as key score
        score2 = normalized_comfort.get((pos2_lookup, pos2_lookup), 0)
        
        # Simply average the two position scores
        return (score1 + score2) / 2, False
        
    # For same side, use the bigram comfort mapping
    # If on right side, map to left side equivalents
    if pos1_right:
        right_to_left = config['layout']['position_mappings']['right_to_left']
        pos1 = right_to_left[pos1]
        pos2 = right_to_left[pos2]
    
    score = max(
        normalized_comfort.get((pos1, pos2), 0.0),
        normalized_comfort.get((pos2, pos1), 0.0)
    )
    
    return score, True

def get_all_positions(config: dict) -> List[str]:
    """Get all keyboard positions from config."""
    positions = []
    for hand in ['right', 'left']:
        for row in ['top', 'home', 'bottom']:
            positions.extend(config['layout']['positions'][hand][row])
    return positions

def print_keyboard_layout(positions: Dict[str, str], title: str, bigram_scores: Dict[str, Dict] = None, pos_perm: Tuple[str, ...] = None):
    """Print a visual representation of the keyboard layout."""
    layout_template = """
╭───────────────────────────────────────────────╮
│ Layout: {title:<34}    │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
│ {q:^3} │ {w:^3} │ {e:^3} │ {r:^3} ║ {u:^3} │ {i:^3} │ {o:^3} │ {p:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
│ {a:^3} │ {s:^3} │ {d:^3} │ {f:^3} ║ {j:^3} │ {k:^3} │ {l:^3} │ {sc:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  Z  │  X  │  C  │  V  ║  M  │  ,  │  .  │  /  │
│ {z:^3} │ {x:^3} │ {c:^3} │ {v:^3} ║ {m:^3} │ {cm:^3} │ {dt:^3} │ {sl:^3} │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯"""

    # Initialize layout characters
    layout_chars = {
        'q': ' ', 'w': ' ', 'e': ' ', 'r': ' ',
        'u': ' ', 'i': ' ', 'o': ' ', 'p': ' ',
        'a': ' ', 's': ' ', 'd': ' ', 'f': ' ',
        'j': ' ', 'k': ' ', 'l': ' ', 'sc': ' ',
        'z': ' ', 'x': ' ', 'c': ' ', 'v': ' ',
        'm': ' ', 'cm': ' ', 'dt': ' ', 'sl': ' '
    }
    
    # Convert special characters in positions to their keys in layout_chars
    position_conversion = {
        ',': 'cm',
        '.': 'dt',
        '/': 'sl',
        ';': 'sc'
    }
    
    # Update with provided positions
    for pos, letter in positions.items():
        layout_key = position_conversion.get(pos, pos)
        layout_chars[layout_key] = letter.upper()
    
    # Print the layout
    print(layout_template.format(title=title, **layout_chars))

    # Print detailed bigram scores if available
    if bigram_scores is not None:
        print("\nTop contributing bigrams:")
        for bigram, score_data in sorted(bigram_scores.items(), key=lambda x: x[1]['total'], reverse=True)[:10]:
            components = score_data['components']
            print(f"{bigram}: {score_data['total']:.6f} ("
                  f"bf: {components['bigram_freq'][1]:.3f}×{components['bigram_freq'][0]}, "
                  f"lf: {components['letter_freq'][1]:.3f}×{components['letter_freq'][0]}, "
                  f"c: {components['comfort'][1]:.3f}×{components['comfort'][0]}, "
                  f"pos: {components['positions']}, "
                  f"same_side: {components['same_side']})")
                        
def print_results(best_placements: List[Tuple[float, List[str], Dict[str, Dict]]], letters: str):
    """Print the results of the optimization."""
    print("\nTop placements:")
    for i, (score, positions, bigram_scores) in enumerate(best_placements, 1):
        print(f"\nPlacement {i}:")
        layout = dict(zip(positions, letters))  # Create the layout dictionary
        # Fix the parameter order here:
        print_keyboard_layout(layout, f"Score: {score:.4f}", bigram_scores, positions)

def validate_config(config: dict):
    """Validate configuration settings.
    
    Args:
        config: Configuration dictionary loaded from YAML
        
    Raises:
        ValueError: If any configuration values are invalid
    """
    # Check required top-level sections
    required_sections = ['data', 'layout', 'optimization', 'visualization', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate layout section
    layout = config.get('layout', {})
    if not layout:
        raise ValueError("Empty layout section")
        
    # Check position definitions
    for hand in ['right', 'left']:
        if hand not in layout.get('positions', {}):
            raise ValueError(f"Missing {hand} hand positions")
        for row in ['top', 'home', 'bottom']:
            if row not in layout['positions'][hand]:
                raise ValueError(f"Missing {row} row for {hand} hand")
            if not layout['positions'][hand][row]:
                raise ValueError(f"Empty {row} row for {hand} hand")
                
    # Validate position mappings
    mappings = layout.get('position_mappings', {})
    if 'right_to_left' not in mappings:
        raise ValueError("Missing right_to_left position mappings")
        
    # Ensure all right-hand positions have mappings
    right_positions = []
    for row in ['top', 'home', 'bottom']:
        right_positions.extend(layout['positions']['right'][row])
    
    for pos in right_positions:
        if pos not in mappings['right_to_left']:
            raise ValueError(f"Missing mapping for right-hand position: {pos}")
            
    # Validate optimization section
    opt = config.get('optimization', {})

    # Validate letters and keys
    if 'letters' not in opt or 'keys' not in opt:
        raise ValueError("Missing letters or keys in optimization section")
        
    letters = opt['letters']
    keys = opt['keys']
    
    # Check lengths match
    if len(letters) != len(keys):
        raise ValueError(f"Number of letters ({len(letters)}) must match number of keys ({len(keys)})")
        
    # Validate keys exist in layout
    all_positions = get_all_positions(config)
    invalid_keys = [k for k in keys if k not in all_positions]
    if invalid_keys:
        raise ValueError(f"Invalid key positions specified: {invalid_keys}")
    
    # Validate scoring weights
    scoring = opt.get('scoring', {})
    if 'bigram_frequency_weight' not in scoring or 'letter_frequency_weight' not in scoring or 'comfort_weight' not in scoring:
        raise ValueError("Missing scoring weights")
        
    bigram_freq_weight = scoring['bigram_frequency_weight']
    letter_freq_weight = scoring['letter_frequency_weight']
    comfort_weight = scoring['comfort_weight']
    
    if not isinstance(bigram_freq_weight, (int, float)) or not isinstance(letter_freq_weight, (int, float)) or not isinstance(comfort_weight, (int, float)):
        raise ValueError("Scoring weights must be numbers")
        
    if bigram_freq_weight < 0 or bigram_freq_weight > 1 or letter_freq_weight < 0 or letter_freq_weight > 1 or comfort_weight < 0 or comfort_weight > 1:
        raise ValueError("Scoring weights must be between 0 and 1")
        
    if abs(bigram_freq_weight + letter_freq_weight + comfort_weight - 1.0) > 1e-6:  # Allow for floating point imprecision
        raise ValueError("Scoring weights must sum to 1.0")
        
    # Validate data paths
    if 'bigram_scores_file' not in config['data']:
        raise ValueError("Missing bigram_scores_file in data section")
    if not Path(config['data']['bigram_scores_file']).exists():
        raise ValueError(f"Bigram scores file not found: {config['data']['bigram_scores_file']}")
        
    if 'output_dir' not in config['data']:
        raise ValueError("Missing output_dir in data section")
    
    # Ensure output directory exists or can be created
    Path(config['data']['output_dir']).mkdir(parents=True, exist_ok=True)

#--------------------------------------------------------------------
# Memory management and parallelization functions
#--------------------------------------------------------------------
def get_available_memory():
    """Get available system memory in bytes."""
    return psutil.virtual_memory().available

def estimate_memory_per_perm(letters: str, positions: str) -> int:
    """Estimate memory needed per permutation in bytes."""
    perm_size = len(positions) * 2  # tuple of position strings
    bigram_size = len(letters) * len(letters) * 20  # bigram scores dict
    return perm_size + bigram_size + 100  # add overhead

#--------------------------------------------------------------------
# Core functions
#--------------------------------------------------------------------
def normalize_scores(comfort_scores: Dict[Tuple[str, str], float], 
                     bigram_frequencies: Dict[str, float], 
                     letter_frequencies: Dict[str, float]) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], Dict[str, float]]:
    """Normalize comfort and frequency scores to 0-1 range."""
    import numpy as np
    
    # Print some raw comfort scores
    print("\nDebug - Raw comfort scores sample:")
    sample_keys = list(comfort_scores.keys())[:5]
    for k in sample_keys:
        print(f"{k}: {comfort_scores[k]}")
    
    # Normalize comfort scores - higher raw scores (less negative) are better
    comfort_min = min(comfort_scores.values())
    comfort_max = max(comfort_scores.values())
    print(f"\nDebug - Comfort score range: min={comfort_min}, max={comfort_max}")
    
    normalized_comfort = {
        k: 1 - ((v - comfort_max) / (comfort_min - comfort_max))  # Flip the normalization
        for k, v in comfort_scores.items()
    }
    
    # Print normalized versions of the same scores
    print("\nDebug - Normalized comfort scores for same keys:")
    for k in sample_keys:
        print(f"{k}: {normalized_comfort[k]}")
    
    # Normalize bigram frequencies logarithmically
    bigram_freqs = list(bigram_frequencies.values())
    bigram_freqs = np.array([f if f > 0 else min(bigram_freqs) for f in bigram_freqs])
    log_bigram_freqs = np.log(bigram_freqs)
    log_min = np.min(log_bigram_freqs)
    log_max = np.max(log_bigram_freqs)    
    normalized_bigram_frequencies = {
        k: (np.log(v if v > 0 else min(bigram_freqs)) - log_min) / (log_max - log_min)
        for k, v in bigram_frequencies.items()
    }
    
    # Normalize letter frequencies linearly
    letter_freqs = list(letter_frequencies.values())
    letter_freqs = np.array([f if f > 0 else min(letter_freqs) for f in letter_freqs])
    letter_freqs_min = np.min(letter_freqs)
    letter_freqs_max = np.max(letter_freqs)
    normalized_letter_frequencies = {
        k: ((v - letter_freqs_min) / (letter_freqs_max - letter_freqs_min))
        for k, v in letter_frequencies.items()
    }

    return normalized_comfort, normalized_bigram_frequencies, normalized_letter_frequencies

def evaluate_layout_permutations(args: Tuple[List[Tuple[str, ...]], str, Dict[Tuple[str, str], float], Dict[str, float], Dict[str, float], dict]):

    # Unpack all arguments
    perms_chunk, letters, normalized_comfort, normalized_bigram_frequencies, normalized_letter_frequencies, config = args

    results = []
    
    # Get keyboard side information
    right_keys = set()
    left_keys = set()
    for row in ['top', 'home', 'bottom']:
        right_keys.update(config['layout']['positions']['right'][row])
        left_keys.update(config['layout']['positions']['left'][row])
    
    # Get weights from config once
    bigram_freq_weight = float(config['optimization']['scoring']['bigram_frequency_weight'])
    letter_freq_weight = float(config['optimization']['scoring']['letter_frequency_weight'])
    comfort_weight = float(config['optimization']['scoring']['comfort_weight'])
    
    if isinstance(perms_chunk, np.ndarray):
        perms_chunk = perms_chunk.tolist()
    
    for pos_perm in perms_chunk:
        pos_perm = tuple(str(p) for p in pos_perm)
        position_map = dict(zip(letters, pos_perm))
        total_score = 0.0
        bigram_scores = {}
        
        for l1 in letters:
            for l2 in letters:
                if l1 != l2:
                    pos1 = position_map[l1]
                    pos2 = position_map[l2]
                    
                    # Get comfort score and same-side status
                    comfort_score, same_side = get_comfort_value(
                        pos1, pos2, normalized_comfort, right_keys, left_keys, config
                    )
                    
                    # Get bigram frequency
                    bigram_forward = f"{l1}{l2}"
                    bigram_freq_forward = normalized_bigram_frequencies.get(bigram_forward, 0)
                    
                    # Get letter frequencies
                    letter1_freq = normalized_letter_frequencies.get(l1, 0)
                    letter2_freq = normalized_letter_frequencies.get(l2, 0)
                    letter_freq_score = (letter1_freq + letter2_freq) / 2
                    
                    # Apply hand alternation bonus for different hands
                    final_comfort_score = comfort_score
                    if not same_side:
                        # The comfort score is already 1.0 for different hands
                        pass
                    
                    # Calculate weighted score
                    weighted_score = (
                        bigram_freq_weight * bigram_freq_forward +
                        letter_freq_weight * letter_freq_score +
                        comfort_weight * final_comfort_score
                    )
                    
                    # Store detailed scoring information
                    bigram_scores[bigram_forward] = {
                        'total': weighted_score,
                        'components': {
                            'bigram_freq': (bigram_freq_weight, bigram_freq_forward),
                            'letter_freq': (letter_freq_weight, letter_freq_score),
                            'comfort': (comfort_weight, final_comfort_score),
                            'positions': (pos1, pos2),
                            'same_side': same_side
                        }
                    }
                    
                    total_score += weighted_score
        
        results.append((total_score, pos_perm, bigram_scores))
    
    n_positions = len(perms_chunk[0]) if perms_chunk else 0
    return sorted(results, key=lambda x: x[0], reverse=True)[:n_positions]

def optimize_letter_placement(
    letters: str,
    config: dict,
    comfort_scores: Optional[Dict[Tuple[str, str], float]] = None,
    n_processes: int = None,
    top_n: int = 3,
    batch_size: int = 1000000
) -> List[Tuple[float, List[str], Dict[str, float]]]:
    """Core function to optimize letter placement."""
    if n_processes is None:
        n_processes = cpu_count() - 1

    letters = config['optimization']['letters']
    target_keys = config['optimization']['keys']

    # Load comfort scores if not provided
    if comfort_scores is None:
        comfort_scores = get_comfort_scores(config)

    # Normalize scores
    normalized_comfort, normalized_bigram_frequencies, normalized_letter_frequencies = normalize_scores(
        comfort_scores, bigram_frequencies, onegram_frequencies  # Pass just the comfort_scores
    )

    # Calculate total permutations for progress bar
    total_perms = factorial(len(target_keys))
    
    # Process permutations in parallel, batch by batch
    best_placements = []
    perms_iterator = permutations(target_keys)
    
    with tqdm(total=total_perms, desc="Processing permutations") as pbar:
        while True:
            batch = list(itertools.islice(perms_iterator, batch_size))
            if not batch:
                break
                
            chunks = np.array_split(batch, n_processes)
            # chunk_args must match evaluate_layout_permutations parameters
            chunk_args = [(chunk, letters, normalized_comfort, normalized_bigram_frequencies, 
                            normalized_letter_frequencies, config) 
                            for chunk in chunks]
            
            with Pool(n_processes) as pool:
                for chunk_results in pool.imap_unordered(evaluate_layout_permutations, chunk_args):
                    best_placements.extend(chunk_results)
                    best_placements.sort(reverse=True)
                    best_placements = best_placements[:len(target_keys)]
                    pbar.update(len(chunk_results) * n_processes)
    
    return best_placements

if __name__ == "__main__":
    config = load_config()
    right_keys = set()
    left_keys = set()
    for row in ['top', 'home', 'bottom']:
        right_keys.update(config['layout']['positions']['right'][row])
        left_keys.update(config['layout']['positions']['left'][row])

    print("Debug - Keyboard sides:")
    print(f"Right keys: {sorted(right_keys)}")
    print(f"Left keys: {sorted(left_keys)}")

    try:
        config = load_config()
        validate_config(config)

        letters = config['optimization']['letters']
        positions = config['optimization']['keys']
        positions_string = ''.join(positions)
        total_perms = factorial(len(positions)) // factorial(len(positions) - len(letters))
        n_processes = cpu_count() - 1

        # Calculate batch size based on memory
        available_memory = get_available_memory()
        memory_to_use = int(available_memory * 0.7)
        mem_per_perm = estimate_memory_per_perm(letters, positions)
        batch_size = min(memory_to_use // mem_per_perm, 1000000)
        
        # Print optimization parameters
        print(f"\nOptimization Parameters:")
        print(f"------------------------")
        print(f"Letters to optimize:     {letters.upper()}")
        print(f"Available positions:     {positions_string.upper()}")
        print(f"Available positions to fill (marked with '*'):")
        layout = {pos: '*' for pos in positions}
        print_keyboard_layout(layout, "Available Positions")

        print(f"\nPermutation Statistics:")
        print(f"----------------------")
        print(f"Total permutations:     {total_perms:,}")
        print(f"Number of processes:    {n_processes}")
        print(f"Batch size:             {batch_size:,}")
        print(f"Memory per permutation: {mem_per_perm / 1024:.1f} KB")
        print(f"Available memory:       {available_memory / (1024**3):.1f} GB")
        print(f"Memory utilization:     {memory_to_use / (1024**3):.1f} GB")
        
        # Estimate time
        estimated_time_per_perm = 0.000001
        total_batches = (total_perms + batch_size - 1) // batch_size
        estimated_total_seconds = total_perms * estimated_time_per_perm / n_processes
        print(f"\nTime Estimate:")
        print(f"-------------")
        print(f"Total batches:          {total_batches:,}")
        print(f"Estimated time:         {timedelta(seconds=int(estimated_total_seconds))}")
        
        # Run optimization
        print(f"\nStarting optimization...")
        start_time = time.time()
        comfort_scores = get_comfort_scores(config)
        
        results = optimize_letter_placement(
            letters=letters,
            config=config,
            comfort_scores=comfort_scores,
            n_processes=n_processes,
            top_n=len(config['optimization']['keys']),
            batch_size=batch_size
        )
        
        elapsed = time.time() - start_time
        print(f"\nOptimization complete!")
        print(f"Total runtime: {timedelta(seconds=int(elapsed))}")
        
        # Print results
        print_results(results, letters)

    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

#--------------------------------------------------------------------
# Pipeline
#--------------------------------------------------------------------
if __name__ == "__main__":

    config = load_config()
    right_keys = set()
    left_keys = set()
    for row in ['top', 'home', 'bottom']:
        right_keys.update(config['layout']['positions']['right'][row])
        left_keys.update(config['layout']['positions']['left'][row])

    print("Debug - Keyboard sides:")
    print(f"Right keys: {sorted(right_keys)}")
    print(f"Left keys: {sorted(left_keys)}")

    try:
        config = load_config()
        validate_config(config)

        letters = config['optimization']['letters']
        positions = config['optimization']['keys']
        positions_string = ''.join(positions)
        total_perms = factorial(len(positions)) // factorial(len(positions) - len(letters))
        n_processes = cpu_count() - 1

        # Calculate batch size based on memory
        available_memory = get_available_memory()
        memory_to_use = int(available_memory * 0.7)
        mem_per_perm = estimate_memory_per_perm(letters, positions)
        batch_size = min(memory_to_use // mem_per_perm, 1000000)  # Cap at 1M permutations per batch
        
        # Print optimization parameters
        print(f"\nOptimization Parameters:")
        print(f"------------------------")
        print(f"Letters to optimize:     {letters.upper()}")
        print(f"Available positions:     {positions_string.upper()}")
        print(f"Available positions to fill (marked with '*'):")
        # Create blank layout with '*' in available positions
        layout = {pos: '*' for pos in positions}
        print_keyboard_layout(layout, "Available Positions")

        print(f"\nPermutation Statistics:")
        print(f"----------------------")
        print(f"Total permutations:     {total_perms:,}")
        print(f"Number of processes:    {n_processes}")
        print(f"Batch size:             {batch_size:,}")
        print(f"Memory per permutation: {mem_per_perm / 1024:.1f} KB")
        print(f"Available memory:       {available_memory / (1024**3):.1f} GB")
        print(f"Memory utilization:     {memory_to_use / (1024**3):.1f} GB")
        
        # Estimate time
        estimated_time_per_perm = 0.000001
        total_batches = (total_perms + batch_size - 1) // batch_size
        estimated_total_seconds = total_perms * estimated_time_per_perm / n_processes
        print(f"\nTime Estimate:")
        print(f"-------------")
        print(f"Total batches:          {total_batches:,}")
        print(f"Estimated time:         {timedelta(seconds=int(estimated_total_seconds))}")
        
        # Run optimization
        print(f"\nStarting optimization...")
        start_time = time.time()
        comfort_scores = get_comfort_scores(config)
        
        results = optimize_letter_placement(
            letters=letters,
            config=config,
            comfort_scores=comfort_scores,
            n_processes=n_processes,
            top_n=len(config['optimization']['keys']),
            batch_size=batch_size
        )
        
        elapsed = time.time() - start_time
        print(f"\nOptimization complete!")
        print(f"Total runtime: {timedelta(seconds=int(elapsed))}")
        
        # Print results
        print_results(results, letters)

    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)