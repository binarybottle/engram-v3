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
from math import factorial, comb, perm
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
    Estimate memory requirements.
    """
    print_debug = False
    if print_debug:
        # Print input parameters
        print(f"\nDebug - Input parameters:")
        print(f"n_letters: {n_letters}")
        print(f"n_positions: {n_positions}")
        print(f"max_candidates: {max_candidates}")
    
    # Fixed sizes for data types in bytes
    INT32_SIZE = 4
    INT8_SIZE = 1
    FLOAT64_SIZE = 8
    BOOL_SIZE = 1
    
    # Core data structure sizes
    mapping_size = n_letters * INT32_SIZE
    position_size = n_positions * BOOL_SIZE
    comfort_matrix_size = n_positions * n_positions * FLOAT64_SIZE
    freq_matrix_size = n_letters * n_letters * FLOAT64_SIZE
    letter_freq_size = n_letters * FLOAT64_SIZE
    key_lr_size = n_positions * INT8_SIZE
    
    if print_debug:
        print("\nDebug - Data structure sizes (bytes):")
        print(f"mapping_size: {mapping_size}")
        print(f"comfort_matrix_size: {comfort_matrix_size}")
        print(f"freq_matrix_size: {freq_matrix_size}")
        print(f"letter_freq_size: {letter_freq_size}")
    
    # Size of each candidate in the priority queue
    bytes_per_candidate = (
        mapping_size +    # partial_mapping array
        FLOAT64_SIZE +   # score
        INT32_SIZE +     # depth
        position_size    # positions boolean array
    )
    
    # Calculate maximum queue size
    max_queue_size = 0
    if print_debug:
        print("\nDebug - Queue size calculation:")
    for depth in range(n_letters):
        branches = min(max_candidates, comb(n_positions, depth))
        branch_memory = branches * (n_positions - depth)
        max_queue_size += branch_memory
        if print_debug:
            print(f"depth {depth}: {branches} branches * {n_positions - depth} positions = {branch_memory}")
    
    # Calculate total memory
    queue_memory = max_queue_size * bytes_per_candidate
    core_memory = (comfort_matrix_size + freq_matrix_size + 
                  letter_freq_size + key_lr_size)
    total_bytes = queue_memory + core_memory
    
    if print_debug:
        print("\nDebug - Final calculations:")
        print(f"bytes_per_candidate: {bytes_per_candidate}")
        print(f"max_queue_size: {max_queue_size}")
        print(f"queue_memory: {queue_memory}")
        print(f"core_memory: {core_memory}")
        print(f"total_bytes: {total_bytes}")
    
    # Get available system memory
    available_memory = psutil.virtual_memory().available
    
    return {
        'estimated_bytes': total_bytes,
        'estimated_mb': total_bytes / (1024 * 1024),
        'estimated_gb': total_bytes / (1024 * 1024 * 1024),
        'available_memory_gb': available_memory / (1024 * 1024 * 1024),
        'memory_sufficient': total_bytes < (available_memory * 0.8),
        'details': {
            'mapping_size_mb': mapping_size / (1024 * 1024),
            'comfort_matrix_mb': comfort_matrix_size / (1024 * 1024),
            'freq_matrix_mb': freq_matrix_size / (1024 * 1024),
            'queue_memory_mb': queue_memory / (1024 * 1024),
            'core_memory_mb': core_memory / (1024 * 1024)
        }
    }

@jit(nopython=True)
def is_position_used(partial_mapping: np.ndarray, position: int) -> bool:
    """Helper function to check if a position is used in partial mapping."""
    return np.any(partial_mapping >= 0) and position in partial_mapping[partial_mapping >= 0]

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
    n_solutions: int = 5,  # Number of solutions to keep
    max_candidates: int = 1000,
    memory_threshold_gb: float = 0.9
) -> List[Tuple[float, Dict[str, str], Dict]]:
    """
    Branch and bound implementation that maintains top N solutions.
    Uses a min-heap to track the N best solutions found so far.
    """
    n_letters = len(letters)
    n_positions = len(keys)
    key_LR, comfort_matrix, letter_freqs, bigram_freqs = arrays
    bigram_weight, letter_weight = weights
    
    # Search candidates priority queue
    candidates = []
    
    # Min-heap for top N solutions (using negatives for max-heap behavior)
    top_n_solutions = []
    worst_top_n_score = float('-inf')
    
    # Initial empty mapping
    initial_mapping = np.full(n_letters, -1, dtype=np.int32)
    initial_used = np.zeros(n_positions, dtype=bool)
    
    # Get initial score
    score_tuple = calculate_layout_score_numba(
        initial_mapping, key_LR, comfort_matrix,
        bigram_freqs, letter_freqs, bigram_weight, letter_weight
    )
    initial_score = score_tuple[0]
    
    heapq.heappush(candidates, (-initial_score, 0, initial_mapping, initial_used))
    
    # Calculate total nodes for progress tracking
    total_nodes = 0
    for depth in range(n_letters):
        total_nodes += perm(n_positions, depth + 1)
    
    processed_nodes = 0
    start_time = time.time()
    
    with tqdm(total=total_nodes, desc="Optimizing layout") as pbar:
        while candidates:
            if psutil.virtual_memory().percent > memory_threshold_gb * 100:
                print("\nWarning: Memory usage high, saving current best solutions")
                break
            
            neg_score, depth, mapping, used = heapq.heappop(candidates)
            score = -neg_score
            
            # Only prune if we have N solutions and this branch can't beat our worst
            if len(top_n_solutions) >= n_solutions and score <= worst_top_n_score:
                # Count skipped nodes
                remaining_levels = n_letters - depth
                skipped_nodes = sum(perm(n_positions - depth, d + 1) 
                                  for d in range(remaining_levels))
                pbar.update(skipped_nodes)
                processed_nodes += skipped_nodes
                continue
            
            if depth == n_letters:
                score_tuple = calculate_layout_score_numba(
                    mapping, key_LR, comfort_matrix,
                    bigram_freqs, letter_freqs, bigram_weight, letter_weight
                )
                total_score, bigram_score, letter_score = score_tuple
                
                # Add to top N if either:
                # 1. We haven't found N solutions yet, or
                # 2. This solution is better than our worst solution
                if (len(top_n_solutions) < n_solutions or 
                    total_score > worst_top_n_score):
                    # Create solution record
                    solution = (
                        total_score,
                        bigram_score,
                        letter_score,
                        mapping.copy()
                    )
                    
                    # Add to min-heap (using negative score for max-heap behavior)
                    heapq.heappush(top_n_solutions, (-total_score, solution))
                    
                    # If we have too many solutions, remove the worst one
                    if len(top_n_solutions) > n_solutions:
                        heapq.heappop(top_n_solutions)
                    
                    # Update worst score (it's the root of our min-heap)
                    if top_n_solutions:
                        worst_top_n_score = -top_n_solutions[0][0]
                
                pbar.update(1)
                processed_nodes += 1
                
                # Update ETA
                elapsed = time.time() - start_time
                if elapsed > 0:
                    nodes_per_second = processed_nodes / elapsed
                    remaining_nodes = total_nodes - processed_nodes
                    eta_seconds = remaining_nodes / nodes_per_second if nodes_per_second > 0 else 0
                    pbar.set_postfix({'ETA': str(timedelta(seconds=int(eta_seconds)))})
                continue
            
            # Try each available position
            nodes_at_depth = 0
            for pos in range(n_positions):
                if not used[pos]:
                    new_mapping = mapping.copy()
                    new_mapping[depth] = pos
                    new_used = used.copy()
                    new_used[pos] = True
                    
                    score_tuple = calculate_layout_score_numba(
                        new_mapping, key_LR, comfort_matrix,
                        bigram_freqs, letter_freqs, bigram_weight, letter_weight
                    )
                    new_score = score_tuple[0]
                    
                    # Only add if this branch could produce a top-N solution
                    if (len(top_n_solutions) < n_solutions or 
                        new_score > worst_top_n_score):
                        heapq.heappush(candidates, (-new_score, depth + 1, new_mapping, new_used))
                        nodes_at_depth += 1
            
            pbar.update(nodes_at_depth)
            processed_nodes += nodes_at_depth
            
            # Update ETA
            elapsed = time.time() - start_time
            if elapsed > 0:
                nodes_per_second = processed_nodes / elapsed
                remaining_nodes = total_nodes - processed_nodes
                eta_seconds = remaining_nodes / nodes_per_second if nodes_per_second > 0 else 0
                pbar.set_postfix({'ETA': str(timedelta(seconds=int(eta_seconds)))})
    
    # Convert heap to sorted list of solutions
    solutions = []
    while top_n_solutions:
        _, solution = heapq.heappop(top_n_solutions)
        total_score, bigram_score, letter_score, mapping = solution
        solutions.append((
            total_score,
            dict(zip(letters, [keys[i] for i in mapping])),
            {'total': {
                'total_score': total_score,
                'bigram_score': bigram_score,
                'letter_score': letter_score
            }}
        ))
    
    # Return solutions in descending score order
    return list(reversed(solutions))

def optimize_layout(config: dict) -> None:
    """
    Main optimization function with improved memory reporting
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
    
    # Check memory requirements first
    memory_estimate = estimate_memory_requirements(len(letters), len(keys), 1000)
    total_kb = memory_estimate['estimated_bytes'] / 1024
    available_gb = memory_estimate['available_memory_gb']
    
    print("\nMemory Requirement Estimation:")
    if total_kb < 1024:  # Less than 1 MB
        print(f"Estimated memory usage: {total_kb:.2f} KB")
    else:
        print(f"Estimated memory usage: {total_kb/1024:.2f} MB")
    print(f"Available system memory: {available_gb:.2f} GB")
    print(f"Memory sufficient: {memory_estimate['memory_sufficient']}")
    
    print("\nDetailed Memory Breakdown:")
    for key, value in memory_estimate['details'].items():
        kb_value = value * 1024  # Convert MB to KB
        if kb_value < 1:
            print(f"{key}: {kb_value*1024:.2f} bytes")
        else:
            print(f"{key}: {kb_value:.2f} KB")
    
    if not memory_estimate['memory_sufficient']:
        raise MemoryError("Insufficient memory available for optimization")

    # Prepare arrays for optimization
    arrays = prepare_arrays(
        letters, keys, right_keys,
        norm_2key_scores, norm_1key_scores,
        norm_bigram_freqs, norm_letter_freqs
    )
    weights = (bigram_weight, letter_weight)
    
    # Get number of layouts from config
    n_layouts = config['optimization'].get('nlayouts', 5)
    
    # Run optimization with n_solutions parameter
    results = branch_and_bound_optimal(
        letters=letters,
        keys=keys,
        arrays=arrays,
        weights=weights,
        n_solutions=n_layouts
    )
    
    # Sort results by total score (descending)
    sorted_results = sorted(
        results,
        key=lambda x: (
            x[0],  # total_score
            x[2]['total']['bigram_score'],  # use bigram score as secondary sort
            x[2]['total']['letter_score']   # use letter score as tertiary sort
        ),
        reverse=True
    )
 
    # Process results (no need to slice anymore)
    print_top_results(sorted_results, config)
    save_results_to_csv(sorted_results, config)

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


    

