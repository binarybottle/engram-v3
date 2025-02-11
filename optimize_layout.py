# # engram-v3/optimize_layout.py
"""
Memory-efficient keyboard layout optimization using branch and bound search.

This script optimizes keyboard layouts by jointly considering typing comfort 
and letter/bigram frequencies. It uses a branch and bound algorithm to find optimal 
letter placements while staying within memory constraints.

See README for more details.

Usage: python optimize_layout.py
"""
import os
import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Set, Tuple
from itertools import permutations, islice
from multiprocessing import Pool, cpu_count
from math import factorial, comb
from tqdm import tqdm
import psutil
import time
from datetime import timedelta
import numba
from numba import jit, float64, boolean
import heapq

from input.bigram_frequencies_english import (
    bigrams, bigram_frequencies_array,
    onegrams, onegram_frequencies_array
)

#-----------------------------------------------------------------------------
# Loading and saving functions
#-----------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_normalize_comfort_scores(config: dict) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Load and normalize comfort scores for both key pairs and individual keys.
    Returns normalized scores for both left and (left-mirrored) right hand positions.
    """
    # Load raw data
    twokey_df = pd.read_csv(config['paths']['input']['two_key_comfort_scores_file'])
    onekey_df = pd.read_csv(config['paths']['input']['one_key_comfort_scores_file'])
    
    # Get position mappings for right-to-left
    right_to_left = config['layout']['position_mappings']['right_to_left']
    left_to_right = {v: k for k, v in right_to_left.items()}
    
    # Normalize comfort scores (lower raw scores are better)
    twokey_min = twokey_df['comfort_score'].min()
    twokey_max = twokey_df['comfort_score'].max()
    onekey_min = onekey_df['comfort_score'].min()
    onekey_max = onekey_df['comfort_score'].max()
    
    #print("\nComfort score normalization:")
    #print(f"2-key comfort range: min={twokey_min:.4f}, max={twokey_max:.4f}")
    #print(f"1-key comfort range: min={onekey_min:.4f}, max={onekey_max:.4f}")

    # Create normalized bigram comfort scores dictionary
    norm_twokey_scores = {}
    for _, row in twokey_df.iterrows():
        # Normalize score
        norm_score = (row['comfort_score'] - twokey_min) / (twokey_max - twokey_min)
        
        # Add left hand bigrams
        key1, key2 = row['first_char'], row['second_char']
        norm_twokey_scores[(key1, key2)] = norm_score
        
        # Mirror to right hand if both keys have right-hand equivalents
        if key1 in left_to_right and key2 in left_to_right:
            right_key1 = left_to_right[key1]
            right_key2 = left_to_right[key2]
            norm_twokey_scores[(right_key1, right_key2)] = norm_score
    
        #print(f"Key pair {key1}, {key2}: raw={row['comfort_score']:.4f}, normalized={norm_score:.4f}")

    # Create normalized single-key comfort scores dictionary
    norm_onekey_scores = {}
    for _, row in onekey_df.iterrows():
        # Normalize score
        norm_score = (row['comfort_score'] - onekey_min) / (onekey_max - onekey_min)

        # Add left hand key
        key = row['key']
        norm_onekey_scores[key] = norm_score
        
        #print(f"Key {key}: raw={row['comfort_score']:.4f}, normalized={norm_score:.4f}")
        
        # Mirror to right hand if key has right-hand equivalent
        if key in left_to_right:
            right_key = left_to_right[key]
            norm_onekey_scores[right_key] = norm_score
    
    return norm_twokey_scores, norm_onekey_scores

def load_and_normalize_frequencies(onegrams: str, 
                                   onegram_frequencies_array: np.ndarray,
                                   bigrams: list, 
                                   bigram_frequencies_array: np.ndarray) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Load and normalize letter and bigram frequencies using log base 10.
    Returns normalized frequencies scaled to 0-1 range.
    """
    # Create letter frequency dictionary
    letter_freqs = dict(zip(onegrams, onegram_frequencies_array))
    
    # Log transform frequencies (use log10)
    # Handle zeros by replacing with minimum non-zero frequency
    min_letter_freq = min(f for f in letter_freqs.values() if f > 0)
    log_letter_freqs = {
        k: np.log10(v if v > 0 else min_letter_freq) 
        for k, v in letter_freqs.items()
    }

    #print("\nLetter frequency normalization:")
    #print(f"Raw freq range: min={min_letter_freq:.4f}, max={max(letter_freqs.values()):.4f}")

    # Normalize letter frequencies to 0-1
    letter_min = min(log_letter_freqs.values())
    letter_max = max(log_letter_freqs.values())
    norm_letter_freqs = {
        k: (v - letter_min) / (letter_max - letter_min)
        for k, v in log_letter_freqs.items()
    }
    
    #print(f"Log freq range: min={letter_min:.4f}, max={letter_max:.4f}")
    #print(f"Letter e: raw={letter_freqs['e']:.4f}, log={log_letter_freqs['e']:.4f}, norm={norm_letter_freqs['e']:.4f}")
    #print(f"Letter z: raw={letter_freqs['z']:.4f}, log={log_letter_freqs['z']:.4f}, norm={norm_letter_freqs['z']:.4f}")

    # Create and normalize bigram frequencies
    bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
    
    # Log transform bigram frequencies
    min_bigram_freq = min(f for f in bigram_freqs.values() if f > 0)
    log_bigram_freqs = {
        k: np.log10(v if v > 0 else min_bigram_freq)
        for k, v in bigram_freqs.items()
    }
    
    # Normalize bigram frequencies to 0-1
    bigram_min = min(log_bigram_freqs.values())
    bigram_max = max(log_bigram_freqs.values())
    norm_bigram_freqs = {
        tuple(k): (v - bigram_min) / (bigram_max - bigram_min)
        for k, v in log_bigram_freqs.items()
    }
    
    return norm_letter_freqs, norm_bigram_freqs

def prepare_arrays(
    letters: str,
    keys: str,
    right_keys: Set[str],
    norm_2key_scores: Dict[Tuple[str, str], float],
    norm_1key_scores: Dict[str, float],
    norm_bigram_freqs: Dict[Tuple[str, str], float],
    norm_letter_freqs: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare input arrays for the numba-optimized scoring function.
    """
    n = len(letters)
    
    # Create key position array (1 for right side, 0 for left side)
    key_LR = np.array([1 if k in right_keys else 0 for k in keys], dtype=np.int8)
    
    # Create comfort score matrix
    comfort_matrix = np.zeros((n, n), dtype=np.float64)
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i == j:
                comfort_matrix[i, j] = norm_1key_scores.get(k1, 0.0)
            else:
                comfort_matrix[i, j] = norm_2key_scores.get((k1, k2), 0.0)
    
    # Create letter frequency array
    letter_freqs = np.array([norm_letter_freqs.get(l, 0.0) for l in letters], dtype=np.float64)
    
    # Create bigram frequency matrix
    bigram_freqs = np.zeros((n, n), dtype=np.float64)
    for i, l1 in enumerate(letters):
        for j, l2 in enumerate(letters):
            bigram_freqs[i, j] = norm_bigram_freqs.get((l1, l2), 0.0)
    
    return key_LR, comfort_matrix, letter_freqs, bigram_freqs

def save_results_to_csv(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]], 
                        config: dict,
                        output_path: str = "layout_results.csv") -> None:
    """
    Save layout results to a CSV file.
    """
    import csv
    from datetime import datetime
    
    # Generate timestamp and set output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config['paths']['output']['layout_results_folder'], 
                              f"layout_results_{timestamp}.csv")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with configuration info
        writer.writerow(['Letters', config['optimization']['letters']])
        writer.writerow(['Keys', config['optimization']['keys']])
        writer.writerow(['Bigram weight', config['optimization']['scoring']['bigram_weight']])
        writer.writerow(['Letter weight', config['optimization']['scoring']['letter_weight']])
        writer.writerow([])  # Empty row for separation
        
        # Write results header
        writer.writerow([
            'Arrangement',
            'Rank',
            'Total score',
            'Bigram score',
            'Letter score'
            #'Avg 2-key comfort',
            #'Avg 2-gram frequency',
            #'Avg 1-key comfort',
            #'Avg 1-gram frequency'
        ])
        
        # Write results
        for rank, (score, mapping, detailed_scores) in enumerate(results, 1):
            arrangement = "".join(mapping.keys())
            #positions = "".join(mapping.values())
           
            # Get first bigram score and letter score (they're the same for all entries)
            first_entry = next(iter(detailed_scores.values()))
            bigram_score = first_entry['bigram_score']
            letter_score = first_entry['letter_score']

            # Component scores
            """
            comfort_scores1 = [d['components']['comfort_seq1'] for d in detailed_scores.values()]
            comfort_scores2 = [d['components']['comfort_seq2'] for d in detailed_scores.values()]
            avg_2key_comfort_score1 = np.mean(comfort_scores1)
            avg_2key_comfort_score2 = np.mean(comfort_scores2)
            avg_2key_comfort_score = (avg_2key_comfort_score1 + avg_2key_comfort_score2) / 2

            freq_scores1 = [d['components']['freq_seq1'] for d in detailed_scores.values()]
            freq_scores2 = [d['components']['freq_seq2'] for d in detailed_scores.values()]
            avg_2gram_freq_score1 = np.mean(freq_scores1)
            avg_2gram_freq_score2 = np.mean(freq_scores2)
            avg_2gram_freq_score = (avg_2gram_freq_score1 + avg_2gram_freq_score2) / 2

            comfort_scores = [d['components']['avg_comfort'] for d in detailed_scores.values()]
            freq_scores = [d['components']['avg_freq'] for d in detailed_scores.values()]
            avg_1key_comfort_score = np.mean(comfort_scores)
            avg_1gram_freq_score = np.mean(freq_scores)
            """

            writer.writerow([
                arrangement,
                rank,
                f"{score:.4f}",
                f"{bigram_score:.4f}",
                f"{letter_score:.4f}"
                #f"{avg_2key_comfort_score:.4f}",
                #f"{avg_2gram_freq_score:.4f}",
                #f"{avg_1key_comfort_score:.4f}",
                #f"{avg_1gram_freq_score:.4f}"
            ])
    
    print(f"\nResults saved to: {output_path}")

#-----------------------------------------------------------------------------
# Visualizing functions
#-----------------------------------------------------------------------------
def visualize_keyboard_layout(mapping: Dict[str, str] = None, title: str = "Layout", config: dict = None) -> None:
    """
    Print a visual representation of the keyboard layout.
    If mapping is None, shows empty positions for keys to be filled.
    
    Args:
        mapping: Dictionary mapping letters to key positions (or None)
        title: Title to display on the layout
        config: Configuration dictionary
    """
    # Key mapping for special characters
    key_mapping = {
        ';': 'sc',  # semicolon
        ',': 'cm',  # comma
        '.': 'dt',  # dot/period
        '/': 'sl'   # forward slash
    }

    # Ensure we have a config
    if config is None:
        raise ValueError("Configuration must be provided")
        
    # Create layout characters dictionary
    layout_chars = {
        'title': title,
        'q': ' ', 'w': ' ', 'e': ' ', 'r': ' ',
        'u': ' ', 'i': ' ', 'o': ' ', 'p': ' ',
        'a': ' ', 's': ' ', 'd': ' ', 'f': ' ',
        'j': ' ', 'k': ' ', 'l': ' ', 'sc': ' ',
        'z': ' ', 'x': ' ', 'c': ' ', 'v': ' ',
        'm': ' ', 'cm': ' ', 'dt': ' ', 'sl': ' '
    }
    
    if mapping:
        # Fill in the mapped letters
        for letter, key in mapping.items():
            layout_chars[key] = letter.upper()
    else:
        # Mark positions to be filled
        for key in config['optimization']['keys']:
            # Convert special characters to their internal representation
            converted_key = key_mapping.get(key, key)
            layout_chars[converted_key] = '_'

    # Get template from config and print
    template = config['visualization']['keyboard_template']
    print(template.format(**layout_chars))

def print_top_results(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]], 
                     config: dict,
                     n: int = None) -> None:
    """
    Print the top N results with their scores and mappings.
    
    Args:
        results: List of (score, mapping, detailed_scores) tuples
        config: Configuration dictionary
        n: Number of layouts to display (defaults to config['optimization']['nlayouts'])
    """
    if n is None:
        n = config['optimization'].get('nlayouts', 5)
    
    print(f"\nTop {n} scoring layouts:")
    for i, (score, mapping, detailed_scores) in enumerate(results[:n], 1):
        print(f"\n#{i}: Score: {score:.4f}")
        visualize_keyboard_layout(mapping, f"Layout #{i}", config)

#-----------------------------------------------------------------------------
# Optimizing functions (with numba)
#-----------------------------------------------------------------------------
def estimate_memory_requirements(n_letters: int, n_positions: int, max_candidates: int) -> dict:
    """
    Estimate memory requirements for the branch and bound algorithm.
    """
    # Size of core data structures (in bytes)
    bytes_per_mapping = n_letters * 4  # int32 array
    bytes_per_assigned = n_positions  # boolean array
    bytes_per_heap_entry = (
        8 +  # score (float64)
        4 +  # depth (int32)
        bytes_per_mapping +
        bytes_per_assigned
    )
    
    # Estimate maximum number of partial solutions
    max_partial_solutions = 0
    for depth in range(n_letters):
        remaining_positions = n_positions - depth
        max_partial_solutions += min(max_candidates, comb(n_positions, depth)) * remaining_positions
    
    # Total memory estimate
    total_bytes = max_partial_solutions * bytes_per_heap_entry
    
    # Get available system memory
    available_memory = psutil.virtual_memory().available
    
    return {
        'estimated_bytes': total_bytes,
        'estimated_mb': total_bytes / (1024 * 1024),
        'estimated_gb': total_bytes / (1024 * 1024 * 1024),
        'available_memory_gb': available_memory / (1024 * 1024 * 1024),
        'memory_sufficient': total_bytes < available_memory * 0.8,  # Use 80% threshold
        'max_partial_solutions': max_partial_solutions
    }

@jit(nopython=True)
def calculate_upper_bound(
    partial_mapping: np.ndarray,
    depth: int,
    key_LR: np.ndarray,
    comfort_matrix: np.ndarray,
    letter_freqs: np.ndarray,
    bigram_freqs: np.ndarray,
    bigram_weight: float,
    letter_weight: float
) -> float:
    """
    Calculate upper bound for a partial solution.
    """
    n_letters = len(partial_mapping)
    
    # Get score for assigned positions (exact)
    total_score, bigram_score, letter_score = calculate_layout_score_numba(
        partial_mapping,
        key_LR,
        comfort_matrix,
        bigram_freqs,
        letter_freqs,
        bigram_weight,
        letter_weight
    )
    
    if depth < n_letters:
        # For unassigned positions, use best possible scores
        remaining_letters = n_letters - depth
        
        # Best possible letter scores
        max_comfort = np.max(comfort_matrix)
        remaining_freq_sum = np.sum(letter_freqs[depth:])
        max_letter_score = letter_weight * max_comfort * remaining_freq_sum
        
        # Best possible bigram scores
        max_bigram_freq = np.max(bigram_freqs[depth:, depth:])
        n_remaining_bigrams = (remaining_letters * (remaining_letters - 1)) // 2
        max_bigram_score = bigram_weight * max_comfort * max_bigram_freq * n_remaining_bigrams
        
        total_score += max_letter_score + max_bigram_score
    
    return total_score

@jit(nopython=True)
def calculate_layout_score_numba(
    letter_indices: np.ndarray,  
    key_LR: np.ndarray,   
    comfort_matrix: np.ndarray,  
    bigram_freqs: np.ndarray,    
    letter_freqs: np.ndarray,    
    bigram_weight: float,
    letter_weight: float
) -> tuple:
    """
    Calculate layout score and return both total and component scores.

    Scoring approach:

        bigram_component = (bigram1_frequency * keypair1_comfort) + 
                           (bigram2_frequency * keypair2_comfort) / 2

        letter_component = letter_frequency * key_comfort

        score = (bigram_weight * bigram_component) + (letter_weight * letter_component)

    Args:
        letter_indices: Array mapping letters to position indices
        key_LR: Binary array (1 for right side, 0 for left side)
        comfort_matrix: Matrix of comfort scores for key pairs
        bigram_freqs: Matrix of bigram frequencies
        letter_freqs: Array of letter frequencies
        bigram_weight: Weight for bigram scores
        letter_weight: Weight for letter scores
    
    Returns:
        float: Total weighted score for the layout
    """
    n = len(letter_indices)
    positions = np.zeros(n, dtype=np.int64)
    for i in range(n):
        positions[i] = letter_indices[i]
    
    # Calculate letter scores (once per letter)
    letter_component = 0.0
    for i in range(n):
        pos = positions[i]
        letter_component += comfort_matrix[pos, pos] * letter_freqs[i]
    
    # Calculate bigram scores (for both sequences of the two keys/letters)
    bigram_component = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pos1 = positions[i]
            pos2 = positions[j]
            
            if key_LR[pos1] == key_LR[pos2]:
                comfort_seq1 = comfort_matrix[pos1, pos2]
                freq_seq1 = bigram_freqs[i, j]
                comfort_seq2 = comfort_matrix[pos2, pos1]
                freq_seq2 = bigram_freqs[j, i]
                bigram_component += (comfort_seq1 * freq_seq1 + comfort_seq2 * freq_seq2)
    
    # Apply weights
    weighted_bigram = bigram_component * bigram_weight / 2
    weighted_letter = letter_component * letter_weight
    
    # Calculate total score
    total_score = weighted_bigram + weighted_letter
    
    return total_score, weighted_bigram, weighted_letter

def branch_and_bound_optimal(
    letters: str,
    keys: str,
    arrays: tuple,
    weights: tuple,
    max_candidates: int = 1000,
    memory_threshold_gb: float = 0.9
) -> List[Tuple[float, Dict[str, str], Dict]]:
    """
    Optimal branch and bound implementation.
    """
    n_letters = len(letters)
    n_positions = len(keys)
    key_LR, comfort_matrix, letter_freqs, bigram_freqs = arrays
    bigram_weight, letter_weight = weights
    
    # Check memory requirements
    memory_estimate = estimate_memory_requirements(n_letters, n_positions, max_candidates)
    if not memory_estimate['memory_sufficient']:
        raise MemoryError(
            f"Estimated memory requirement ({memory_estimate['estimated_gb']:.2f} GB) "
            f"exceeds available memory ({memory_estimate['available_memory_gb']:.2f} GB)"
        )
    
    candidates = []
    best_complete_score = float('-inf')
    best_solutions = []
    
    # Initialize with empty layout
    initial_mapping = np.full(n_letters, -1, dtype=np.int32)
    
    # Calculate initial upper bound
    initial_bound = calculate_upper_bound(
        initial_mapping, 0, key_LR, comfort_matrix,
        letter_freqs, bigram_freqs, bigram_weight, letter_weight
    )
    
    heapq.heappush(candidates, (-initial_bound, 0, initial_mapping))
    
    while candidates:
        # Check memory usage
        if psutil.virtual_memory().percent > memory_threshold_gb * 100:
            print("\nWarning: Memory usage high, saving current best solutions")
            break
        
        neg_bound, depth, partial_mapping = heapq.heappop(candidates)
        upper_bound = -neg_bound
        
        # Skip if this branch can't beat best solution
        if upper_bound <= best_complete_score:
            continue
        
        # Complete solution found
        if depth == n_letters:
            total_score, bigram_score, letter_score = calculate_layout_score_numba(
                partial_mapping, key_LR, comfort_matrix,
                bigram_freqs, letter_freqs, bigram_weight, letter_weight
            )
            
            if total_score > best_complete_score:
                best_complete_score = total_score
                best_solutions = [(total_score, bigram_score, letter_score, partial_mapping.copy())]
            elif total_score == best_complete_score:
                best_solutions.append((total_score, bigram_score, letter_score, partial_mapping.copy()))
            continue
        
        # Try each available position for next letter
        used_positions = set(partial_mapping[partial_mapping >= 0])
        for pos in range(n_positions):
            if pos not in used_positions:
                new_mapping = partial_mapping.copy()
                new_mapping[depth] = pos
                
                new_bound = calculate_upper_bound(
                    new_mapping, depth + 1, key_LR, comfort_matrix,
                    letter_freqs, bigram_freqs, bigram_weight, letter_weight
                )
                
                if new_bound > best_complete_score:
                    heapq.heappush(candidates, (-new_bound, depth + 1, new_mapping))
    
    # Convert results to required format
    return [(
        total_score,
        dict(zip(letters, [keys[i] for i in mapping])),
        {'total': {
            'total_score': total_score,
            'bigram_score': bigram_score,
            'letter_score': letter_score
        }}
    ) for total_score, bigram_score, letter_score, mapping in sorted(best_solutions, reverse=True)]

def optimize_layout(config: dict) -> None:
    """
    Main optimization function using branch and upper bound.
    """
    # Load and normalize scores
    norm_2key_scores, norm_1key_scores = load_and_normalize_comfort_scores(config)
    norm_letter_freqs, norm_bigram_freqs = load_and_normalize_frequencies(
        onegrams, onegram_frequencies_array, bigrams, bigram_frequencies_array
    )
    
    # Get configuration parameters
    letters = config['optimization']['letters']
    keys = config['optimization']['keys']
    bigram_weight = config['optimization']['scoring']['bigram_weight']
    letter_weight = config['optimization']['scoring']['letter_weight']
    
    # Get right-hand keys
    right_keys = set()
    for row in ['top', 'home', 'bottom']:
        right_keys.update(config['layout']['positions']['right'][row])
    
    # Prepare arrays
    arrays = prepare_arrays(
        letters, keys, right_keys,
        norm_2key_scores, norm_1key_scores,
        norm_bigram_freqs, norm_letter_freqs
    )
    
    weights = (bigram_weight, letter_weight)
    
    # Check memory requirements first
    memory_estimate = estimate_memory_requirements(len(letters), len(keys), 1000)
    print("\nMemory Requirement Estimation:")
    print(f"Estimated memory usage: {memory_estimate['estimated_gb']:.2f} GB")
    print(f"Available system memory: {memory_estimate['available_memory_gb']:.2f} GB")
    print(f"Memory sufficient: {memory_estimate['memory_sufficient']}")
    
    if not memory_estimate['memory_sufficient']:
        raise MemoryError("Insufficient memory available for optimization")
    
    # Run branch and bound optimization
    results = branch_and_bound_optimal(
        letters=letters,
        keys=keys,
        arrays=arrays,
        weights=weights
    )
    
    # Process results as before
    print_top_results(results[:config['optimization'].get('nlayouts', 5)], config)
    save_results_to_csv(results[:config['optimization'].get('nlayouts', 5)], config)

#--------------------------------------------------------------------
# Pipeline
#--------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config('config.yaml')
    
        # Show initial keyboard with positions to fill
        visualize_keyboard_layout(None, "Keys to optimize", config)

        # Print optimization parameters
        letters = config['optimization']['letters']
        positions = config['optimization']['keys']
        total_perms = factorial(len(positions)) // factorial(len(positions) - len(letters))
        n_processes = cpu_count() - 1
        print(f"Letters to arrange:      {letters.upper()}")
        print(f"Available positions:     {positions.upper()}")
        print(f"Total permutations:     {total_perms:,}")
        print(f"Number of processes:    {n_processes}")
        
        #--------------------------
        # Optimize the layout
        #--------------------------
        start_time = time.time()

        optimize_layout(config)

        elapsed = time.time() - start_time
        print(f"Done! Total runtime: {timedelta(seconds=int(elapsed))}")

    except Exception as e:
        print(f"Error: {e}")


    

