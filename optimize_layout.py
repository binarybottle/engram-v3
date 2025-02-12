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
from math import factorial, comb, perm
from tqdm import tqdm
import psutil
import time
from datetime import datetime, timedelta
from numba import prange, jit, float64, boolean
import heapq
import csv

from input.bigram_frequencies_english import (
    bigrams, bigram_frequencies_array,
    onegrams, onegram_frequencies_array
)

#-----------------------------------------------------------------------------
# Loading and saving functions
#-----------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file and normalize letter case and numeric types."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert weights to float32
    config['optimization']['scoring']['bigram_weight'] = np.float32(
        config['optimization']['scoring']['bigram_weight']
    )
    config['optimization']['scoring']['letter_weight'] = np.float32(
        config['optimization']['scoring']['letter_weight']
    )
    
    # Normalize position mappings
    for side in ['right', 'left']:
        for row in ['top', 'home', 'bottom']:
            if side in config['layout']['positions']:
                config['layout']['positions'][side][row] = [
                    k.lower() for k in config['layout']['positions'][side][row]
                ]
    
    # Normalize strings to lowercase
    optimization = config['optimization']
    for key in ['letters', 'keys', 
                'letters_to_display', 'keys_to_display',
                'letters_to_constrain', 'keys_to_constrain']:
        if key in optimization:
            optimization[key] = optimization[key].lower()
    
    # Normalize right_to_left mappings
    right_to_left = config['layout']['position_mappings']['right_to_left']
    normalized_mappings = {}
    for key, value in right_to_left.items():
        normalized_mappings[key.lower()] = value.lower()
    config['layout']['position_mappings']['right_to_left'] = normalized_mappings
    
    # Validate constraints
    validate_optimization_inputs(config)
    
    return config

def load_and_normalize_comfort_scores(config: dict) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Load and normalize comfort scores using float32 precision.
    """
    # Load raw data
    twokey_df = pd.read_csv(config['paths']['input']['two_key_comfort_scores_file'])
    onekey_df = pd.read_csv(config['paths']['input']['one_key_comfort_scores_file'])
    
    # Get position mappings for right-to-left
    right_to_left = config['layout']['position_mappings']['right_to_left']
    left_to_right = {v: k for k, v in right_to_left.items()}
    
    # Normalize comfort scores (lower raw scores are better)
    twokey_min = np.float32(twokey_df['comfort_score'].min())
    twokey_max = np.float32(twokey_df['comfort_score'].max())
    onekey_min = np.float32(onekey_df['comfort_score'].min())
    onekey_max = np.float32(onekey_df['comfort_score'].max())
    
    # Create normalized bigram comfort scores dictionary
    norm_twokey_scores = {}
    for _, row in twokey_df.iterrows():
        # Normalize score with float32 precision
        norm_score = np.float32((row['comfort_score'] - twokey_min) / (twokey_max - twokey_min))
        
        # Add left hand bigrams
        key1, key2 = row['first_char'], row['second_char']
        norm_twokey_scores[(key1, key2)] = norm_score
        
        # Mirror to right hand if both keys have right-hand equivalents
        if key1 in left_to_right and key2 in left_to_right:
            right_key1 = left_to_right[key1]
            right_key2 = left_to_right[key2]
            norm_twokey_scores[(right_key1, right_key2)] = norm_score

    # Create normalized single-key comfort scores dictionary
    norm_onekey_scores = {}
    for _, row in onekey_df.iterrows():
        # Normalize score with float32 precision
        norm_score = np.float32((row['comfort_score'] - onekey_min) / (onekey_max - onekey_min))

        # Add left hand key
        key = row['key']
        norm_onekey_scores[key] = norm_score
        
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
    Load and normalize frequencies using float32 precision.
    """
    # Create letter frequency dictionary
    letter_freqs = dict(zip(onegrams, onegram_frequencies_array))
    
    # Log transform frequencies with float32 precision
    min_letter_freq = np.float32(min(f for f in letter_freqs.values() if f > 0))
    log_letter_freqs = {
        k: np.float32(np.log10(v if v > 0 else min_letter_freq))
        for k, v in letter_freqs.items()
    }

    # Normalize letter frequencies to 0-1
    letter_min = np.float32(min(log_letter_freqs.values()))
    letter_max = np.float32(max(log_letter_freqs.values()))
    norm_letter_freqs = {
        k: np.float32((v - letter_min) / (letter_max - letter_min))
        for k, v in log_letter_freqs.items()
    }

    # Create and normalize bigram frequencies
    bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
    
    # Log transform bigram frequencies
    min_bigram_freq = np.float32(min(f for f in bigram_freqs.values() if f > 0))
    log_bigram_freqs = {
        k: np.float32(np.log10(v if v > 0 else min_bigram_freq))
        for k, v in bigram_freqs.items()
    }
    
    # Normalize bigram frequencies to 0-1
    bigram_min = np.float32(min(log_bigram_freqs.values()))
    bigram_max = np.float32(max(log_bigram_freqs.values()))
    norm_bigram_freqs = {
        tuple(k): np.float32((v - bigram_min) / (bigram_max - bigram_min))
        for k, v in log_bigram_freqs.items()
    }
    
    return norm_letter_freqs, norm_bigram_freqs

def validate_optimization_inputs(config):

    # Get letters/keys from config
    letters_to_assign = set(config['optimization']['letters_to_assign'].lower())
    keys_to_assign = set(config['optimization']['keys_to_assign'].lower())
    letters_to_constrain = set(config['optimization'].get('letters_to_constrain', '').lower())
    keys_to_constrain = set(config['optimization'].get('keys_to_constrain', '').lower())
    letters_assigned = set(config['optimization'].get('letters_assigned', '').lower())
    keys_assigned = set(config['optimization'].get('keys_assigned', '').lower())

    # Check for duplicates in letters
    if len(set(letters_to_assign)) != len(letters_to_assign):
        raise ValueError(f"Duplicate letters: {letters_to_assign}")
    
    # Check for duplicates in keys
    if len(set(keys_to_assign)) != len(keys_to_assign):
        raise ValueError(f"Duplicate keys: {keys_to_assign}")
    
    # Check that we have enough positions
    if len(letters_to_assign) > len(keys_to_assign):
        raise ValueError(f"More letters ({len(letters_to_assign)}) than available positions ({len(keys_to_assign)})")

    # Validate no overlap between assigned and to_assign/to_constrain
    if not letters_assigned.isdisjoint(letters_to_assign):
        raise ValueError("letters_to_assign/constrain contains letters that are already assigned")
    if not keys_assigned.isdisjoint(keys_to_assign):
        raise ValueError("keys_to_assign/constrain contains keys that are already assigned")
        
    # Check if letters_to_constrain is subset of letters
    if not letters_to_constrain.issubset(letters_to_assign):
        invalid_letters = letters_to_constrain - letters_to_assign
        raise ValueError(f"letters_to_constrain contains letters not in main letters set: {invalid_letters}")

    # Check if keys_to_constrain is subset of keys
    if not keys_to_constrain.issubset(keys_to_assign):
        invalid_keys = keys_to_constrain - keys_to_assign
        raise ValueError(f"keys_to_constrain contains keys not in main keys set: {invalid_keys}")
    
    # Check if we have enough constraint keys for constraint letters
    if len(letters_to_constrain) > len(keys_to_constrain):
        raise ValueError(
            f"Not enough constraint keys ({len(keys_to_constrain)}) "
            f"for constraint letters ({len(letters_to_constrain)})"
        )

    # Check for duplicates in assigned letters/keys
    if len(set(letters_assigned)) != len(letters_assigned):
        raise ValueError(f"Duplicate assigned letters: {letters_assigned}")
    if len(set(keys_assigned)) != len(keys_assigned):
        raise ValueError(f"Duplicate assigned keys: {keys_assigned}")
    
    # Check that assigned letters and keys have matching lengths
    if len(letters_assigned) != len(keys_assigned):
        raise ValueError(
            f"Mismatched number of assigned letters ({len(letters_assigned)}) "
            f"and assigned keys ({len(keys_assigned)})"
        )

    # Check no overlap between assigned and constrained
    if not letters_assigned.isdisjoint(letters_to_constrain):
        overlap = letters_assigned.intersection(letters_to_constrain)
        raise ValueError(f"letters_to_constrain contains assigned letters: {overlap}")
    if not keys_assigned.isdisjoint(keys_to_constrain):
        overlap = keys_assigned.intersection(keys_to_constrain)
        raise ValueError(f"keys_to_constrain contains assigned keys: {overlap}")

    # Check for duplicates in constrained letters/keys
    if len(set(letters_to_constrain)) != len(letters_to_constrain):
        raise ValueError(f"Duplicate constrained letters: {letters_to_constrain}")
    if len(set(keys_to_constrain)) != len(keys_to_constrain):
        raise ValueError(f"Duplicate constrained keys: {keys_to_constrain}")
 
def prepare_arrays(
    letters_to_assign: str,
    keys_to_assign: str,
    right_keys: Set[str],
    norm_2key_scores: Dict[Tuple[str, str], float],
    norm_1key_scores: Dict[str, float],
    norm_bigram_freqs: Dict[Tuple[str, str], float],
    norm_letter_freqs: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare input arrays for the scoring function.
    """
    n_letters_to_assign = len(letters_to_assign)
    n_keys_to_assign = len(keys_to_assign)
    
    # Create key position array (1 for right side, 0 for left side)
    key_LR = np.array([1 if k in right_keys else 0 for k in keys_to_assign], dtype=np.int8)
    
    # Create comfort score matrix for all possible positions
    comfort_matrix = np.zeros((n_keys_to_assign, n_keys_to_assign), dtype=np.float32)
    for i, k1 in enumerate(keys_to_assign):
        for j, k2 in enumerate(keys_to_assign):
            if i == j:  # Single key comfort
                comfort_matrix[i, j] = norm_1key_scores.get(k1.lower(), 0.0)
            else:  # Two key comfort
                comfort_matrix[i, j] = norm_2key_scores.get((k1.lower(), k2.lower()), 0.0)
    
    print("Comfort matrix:")
    print(comfort_matrix)
    
    # Create letter frequency array
    letter_freqs = np.array([
        norm_letter_freqs.get(l.lower(), 0.0) for l in letters_to_assign
    ], dtype=np.float32)
    
    # Create bigram frequency matrix
    bigram_freqs = np.zeros((n_letters_to_assign, n_letters_to_assign), dtype=np.float32)
    for i, l1 in enumerate(letters_to_assign):
        for j, l2 in enumerate(letters_to_assign):
            bigram_freqs[i, j] = norm_bigram_freqs.get((l1.lower(), l2.lower()), 0.0)
    
    return key_LR, comfort_matrix, letter_freqs, bigram_freqs

def prepare_assigned_indices(letters: str, letters_assigned: str, keys: str, keys_assigned: str) -> np.ndarray:
    """
    Create array of assigned positions (-1 for unassigned letters)
    
    Args:
        letters: String of all letters being optimized
        letters_assigned: String of pre-assigned letters
        keys: String of all available keys
        keys_assigned: String of keys where letters are pre-assigned
        
    Returns:
        numpy array of indices where assigned letters are marked with their position index
        and unassigned letters are marked with -1
        
    Example:
        letters = "abcdef"
        letters_assigned = "ac"
        keys = "FDSVRA"
        keys_assigned = "FR"
        Result: [-1, 0, -1, 1, -1, -1] # 'a' assigned to position 0 (F), 'c' to position 1 (R)
    """
    n_letters = len(letters)
    assigned_indices = np.full(n_letters, -1, dtype=np.int32)
    
    if letters_assigned and keys_assigned:
        # Create letter to index mapping for input letters
        letter_to_idx = {letter: idx for idx, letter in enumerate(letters)}
        # Create key to position mapping for input keys
        key_to_pos = {key: pos for pos, key in enumerate(keys)}
        
        # For each assigned letter-key pair
        for letter, key in zip(letters_assigned, keys_assigned):
            # Only process if both letter and key are in our mappings
            if letter in letter_to_idx and key in key_to_pos:
                idx = letter_to_idx[letter]
                pos = key_to_pos[key]
                assigned_indices[idx] = pos
                
    return assigned_indices

def save_results_to_csv(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]], 
                        config: dict,
                        output_path: str = "layout_results.csv") -> None:
    """
    Save layout results to a CSV file.
    """
    # Generate timestamp and set output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config['paths']['output']['layout_results_folder'], 
                              f"layout_results_{timestamp}.csv")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with configuration info
        writer.writerow(['Letters to assign', config['optimization']['letters_to_assign']])
        writer.writerow(['Available keys', config['optimization']['keys_to_assign']])
        writer.writerow(['Letters to constrain', config['optimization']['letters_to_constrain']])
        writer.writerow(['Constraint keys', config['optimization']['keys_to_constrain']])
        writer.writerow(['Assigned letters', config['optimization']['letters_assigned']])
        writer.writerow(['Assigned keys', config['optimization']['keys_assigned']])
        writer.writerow(['Bigram weight', config['optimization']['scoring']['bigram_weight']])
        writer.writerow(['Letter weight', config['optimization']['scoring']['letter_weight']])
        writer.writerow([])  # Empty row for separation
        
        # Write results header
        writer.writerow([
            'Arrangement',
            'Positions',
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
            positions = "".join(mapping.values())
           
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
                positions,    
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
def visualize_keyboard_layout(mapping: Dict[str, str] = None, title: str = "Layout", config: dict = None, letters_to_display: str = None, keys_to_display: str = None) -> None:
    """
    Print a visual representation of the keyboard layout showing both mapped and assigned letters.
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
        
    # Create layout characters dictionary with empty spaces
    layout_chars = {
        'title': title,
        'q': ' ', 'w': ' ', 'e': ' ', 'r': ' ',
        'u': ' ', 'i': ' ', 'o': ' ', 'p': ' ',
        'a': ' ', 's': ' ', 'd': ' ', 'f': ' ',
        'j': ' ', 'k': ' ', 'l': ' ', 'sc': ' ',
        'z': ' ', 'x': ' ', 'c': ' ', 'v': ' ',
        'm': ' ', 'cm': ' ', 'dt': ' ', 'sl': ' '
    }
    
    # First apply assigned letters from config
    letters_assigned = config['optimization'].get('letters_assigned', '').lower()
    keys_assigned = config['optimization'].get('keys_assigned', '').lower()
    
    if letters_assigned and keys_assigned:
        for letter, key in zip(letters_assigned, keys_assigned):
            converted_key = key_mapping.get(key.lower(), key.lower())
            layout_chars[converted_key] = letter.lower()
    
    # Then mark positions to be filled from keys_to_assign
    if not mapping:
        keys_to_mark = config['optimization'].get('keys_to_assign', '').lower()
        for key in keys_to_mark:
            if key not in keys_assigned:  # Skip if already assigned
                converted_key = key_mapping.get(key, key)
                layout_chars[converted_key] = '-'
    
    # Fill in letters from letters_to_display and keys_to_display
    if letters_to_display and keys_to_display:
        if len(letters_to_display) != len(keys_to_display):
            raise ValueError("letters_to_display and keys_to_display must have the same length")
            
        for letter, key in zip(letters_to_display, keys_to_display):
            converted_key = key_mapping.get(key.lower(), key.lower())
            layout_chars[converted_key] = letter.lower()
    
    # Fill in letters from mapping
    if mapping:
        for letter, key in mapping.items():
            converted_key = key_mapping.get(key.lower(), key.lower())
            current_char = layout_chars[converted_key]
            
            # If there's already an assigned letter, combine them
            if current_char.strip() and current_char != '_':
                # Put mapped letter (uppercase) first, then assigned letter (lowercase)
                layout_chars[converted_key] = f"{letter.upper()}{current_char}"
            else:
                # Just show the mapped letter in uppercase
                layout_chars[converted_key] = letter.upper()

    # Get template from config and print
    template = config['visualization']['keyboard_template']
    print(template.format(**layout_chars))

def print_top_results(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]], 
                      config: dict,
                      n: int = None,
                      letters_to_display: str = None,
                      keys_to_display: str = None) -> None:
    """
    Print the top N results with their scores and mappings.
    
    Args:
        results: List of (score, mapping, detailed_scores) tuples
        config: Configuration dictionary
        n: Number of layouts to display (defaults to config['optimization']['nlayouts'])
        letters_to_display: Letters that have already been assigned
        keys_to_display: Keys that have already been assigned
    """
    if n is None:
        n = config['optimization'].get('nlayouts', 5)
    
    print(f"\nTop {n} scoring layouts:")
    for i, (score, mapping, detailed_scores) in enumerate(results[:n], 1):
        print(f"\n#{i}: Score: {score:.4f}")
        visualize_keyboard_layout(
            mapping=mapping,
            title=f"Layout #{i}", 
            letters_to_display=letters_to_display,
            keys_to_display=keys_to_display,
            config=config
        )
        
#-----------------------------------------------------------------------------
# Optimizing functions (with numba)
#-----------------------------------------------------------------------------
def calculate_memory_upper_bound(n_letters: int, n_positions: int) -> dict:
    """
    Calculate worst-case memory requirements.
    """
    # Fixed sizes in bytes
    INT32_SIZE = 4
    INT8_SIZE = 1
    FLOAT64_SIZE = 8
    BOOL_SIZE = 1
    
    # Core arrays (these are fixed)
    mapping_size = n_letters * INT32_SIZE
    comfort_matrix_size = n_positions * n_positions * FLOAT64_SIZE
    freq_matrix_size = n_letters * n_letters * FLOAT64_SIZE
    letter_freq_size = n_letters * FLOAT64_SIZE
    key_lr_size = n_positions * INT8_SIZE
    
    # Size of each candidate in queue
    bytes_per_candidate = (
        mapping_size +    # partial_mapping array
        FLOAT64_SIZE +   # score
        INT32_SIZE +     # depth
        n_positions * BOOL_SIZE  # used positions array
    )
    
    # Worst case: store candidates for every possible partial assignment
    # At each depth d, we could have P(n_positions, d) candidates
    max_queue_size = 0
    candidates_by_depth = []
    for depth in range(n_letters):
        candidates_at_depth = perm(n_positions, depth + 1)
        max_queue_size += candidates_at_depth
        candidates_by_depth.append(candidates_at_depth)
    
    # Total memory
    queue_memory = max_queue_size * bytes_per_candidate
    core_memory = comfort_matrix_size + freq_matrix_size + letter_freq_size + key_lr_size
    total_memory = queue_memory + core_memory
    
    return {
        'total_bytes': total_memory,
        'total_gb': total_memory / (1024**3),
        'queue_size': max_queue_size,
        'candidates_by_depth': candidates_by_depth,
        'breakdown': {
            'core_arrays_mb': core_memory / (1024**2),
            'queue_mb': queue_memory / (1024**2),
            'bytes_per_candidate': bytes_per_candidate
        }
    }

def estimate_memory_requirements(n_letters: int, n_positions: int, max_candidates: int) -> dict:
    """
    Estimate memory requirements more accurately for branch and bound search.
    """
    # Fixed sizes for data types in bytes
    INT32_SIZE = 4
    INT8_SIZE = 1
    FLOAT32_SIZE = 4
    BOOL_SIZE = 1
    
    # Core data structure sizes (these are fixed overhead)
    mapping_size = n_letters * INT32_SIZE
    position_size = n_positions * BOOL_SIZE
    comfort_matrix_size = n_positions * n_positions * FLOAT32_SIZE
    freq_matrix_size = n_letters * n_letters * FLOAT32_SIZE
    letter_freq_size = n_letters * FLOAT32_SIZE
    key_lr_size = n_positions * INT8_SIZE
    
    # Size of each candidate in the priority queue
    bytes_per_candidate = (
        mapping_size +   # partial_mapping array
        FLOAT32_SIZE +   # score
        INT32_SIZE +     # depth
        position_size    # positions boolean array
    )
    
    # For large problems, we need to estimate the working set size
    # Assume we'll keep max_candidates at each depth
    max_queue_size = max_candidates * n_letters
    
    # Calculate branching factor at each depth
    # For depth d, we have (n_positions - d) choices
    avg_branching_factor = sum(n_positions - d for d in range(n_letters)) / n_letters
    
    # Estimate working memory needed for search
    working_memory = max_queue_size * bytes_per_candidate * avg_branching_factor
    
    # Calculate total memory including fixed overhead
    core_memory = (comfort_matrix_size + freq_matrix_size + 
                  letter_freq_size + key_lr_size)
    total_bytes = working_memory + core_memory
    
    # Add safety margin (50% extra)
    total_bytes = total_bytes * 1.5
    
    # Get available system memory
    available_memory = psutil.virtual_memory().available
    
    # Convert to appropriate units
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)
    available_gb = available_memory / (1024 * 1024 * 1024)
    
    # Calculate statistics about the search space
    total_arrangements = perm(n_positions, n_letters)
    estimated_nodes_per_sec = 10000  # Based on typical performance
    estimated_seconds = total_arrangements / estimated_nodes_per_sec
    estimated_runtime = str(timedelta(seconds=int(estimated_seconds)))
    
    return {
        'estimated_bytes': total_bytes,
        'estimated_mb': total_mb,
        'estimated_gb': total_gb,
        'available_memory_gb': available_gb,
        'memory_sufficient': total_bytes < (available_memory * 0.8),
        'estimated_runtime': estimated_runtime,
        'search_space_size': total_arrangements,
        'details': {
            'comfort_matrix_mb': comfort_matrix_size / (1024 * 1024),
            'freq_matrix_mb': freq_matrix_size / (1024 * 1024),
            'working_memory_mb': working_memory / (1024 * 1024),
            'core_memory_mb': core_memory / (1024 * 1024),
            'bytes_per_candidate': bytes_per_candidate,
            'max_queue_size': max_queue_size,
            'avg_branching_factor': avg_branching_factor
        }
    }

@jit(nopython=True, fastmath=True)
def is_position_used(partial_mapping: np.ndarray, position: int) -> bool:
    """Helper function to check if a position is used in partial mapping."""
    # Check if any element equals position
    for i in range(len(partial_mapping)):
        if partial_mapping[i] == position and partial_mapping[i] >= 0:
            return True
    return False

@jit(nopython=True, fastmath=True)
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
    Calculate upper bound with safe array comparisons.
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
        remaining_letters = n_letters - depth
        
        # Best possible letter scores (using scalar operations)
        max_comfort = np.float32(0.0)
        for i in range(len(comfort_matrix)):
            for j in range(len(comfort_matrix[i])):
                if comfort_matrix[i,j] > max_comfort:
                    max_comfort = comfort_matrix[i,j]
        
        remaining_freq_sum = np.float32(0.0)
        for i in range(depth, len(letter_freqs)):
            remaining_freq_sum += letter_freqs[i]
            
        max_letter_score = letter_weight * max_comfort * remaining_freq_sum
        
        # Best possible bigram scores (using scalar operations)
        max_bigram_freq = np.float32(0.0)
        for i in range(depth, len(bigram_freqs)):
            for j in range(depth, len(bigram_freqs[i])):
                if bigram_freqs[i,j] > max_bigram_freq:
                    max_bigram_freq = bigram_freqs[i,j]
                    
        n_remaining_bigrams = (remaining_letters * (remaining_letters - 1)) // 2
        max_bigram_score = bigram_weight * max_comfort * max_bigram_freq * n_remaining_bigrams
        
        total_score += max_letter_score + max_bigram_score
    
    return total_score

@jit(nopython=True, fastmath=True)
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
    Calculate layout score with safe array comparisons.
    """
    # Check if we have any valid positions
    has_valid = False
    for i in range(len(letter_indices)):
        if letter_indices[i] >= 0:
            has_valid = True
            break
    
    if not has_valid:
        return 0.0, 0.0, 0.0
        
    # Calculate single-letter component
    letter_component = np.float32(0.0)
    for i in range(len(letter_indices)):
        pos = letter_indices[i]
        if pos >= 0:
            letter_component += comfort_matrix[pos, pos] * letter_freqs[i]
    
    # Calculate bigram component
    bigram_component = np.float32(0.0)
    for i in range(len(letter_indices)):
        pos1 = letter_indices[i]
        if pos1 >= 0:
            for j in range(i + 1, len(letter_indices)):
                pos2 = letter_indices[j]
                if pos2 >= 0 and key_LR[pos1] == key_LR[pos2]:
                    comfort_seq1 = comfort_matrix[pos1, pos2]
                    freq_seq1 = bigram_freqs[i, j]
                    comfort_seq2 = comfort_matrix[pos2, pos1]
                    freq_seq2 = bigram_freqs[j, i]
                    bigram_component += (comfort_seq1 * freq_seq1 + comfort_seq2 * freq_seq2)
    
    weighted_bigram = bigram_component * bigram_weight / 2.0
    weighted_letter = letter_component * letter_weight
    total_score = weighted_bigram + weighted_letter
    
    return float(total_score), float(weighted_bigram), float(weighted_letter)

def calculate_layout_score_debug(
    letter_indices: np.ndarray,
    key_LR: np.ndarray,   
    comfort_matrix: np.ndarray,  
    bigram_freqs: np.ndarray,    
    letter_freqs: np.ndarray,    
    bigram_weight: float,
    letter_weight: float
) -> tuple:
    """Debug version of calculate_layout_score_numba without JIT compilation."""
    print(f"\nDebug info:")
    print(f"letter_indices: {letter_indices}")
    
    # Only calculate scores for valid positions
    valid_positions = letter_indices >= 0
    if not np.any(valid_positions):
        print("No valid positions found, returning zeros")
        return 0.0, 0.0, 0.0
        
    # Calculate single-letter component
    letter_component = np.float32(0.0)
    for i in range(len(letter_indices)):
        pos = letter_indices[i]
        if pos >= 0:
            curr_comfort = comfort_matrix[pos, pos]
            curr_freq = letter_freqs[i]
            curr_score = curr_comfort * curr_freq
            letter_component += curr_score
            print(f"Letter {i}: pos={pos}, comfort={curr_comfort}, freq={curr_freq}, score={curr_score}")
    
    # Calculate bigram component
    bigram_component = np.float32(0.0)
    for i in range(len(letter_indices)):
        pos1 = letter_indices[i]
        if pos1 >= 0:
            for j in range(i + 1, len(letter_indices)):
                pos2 = letter_indices[j]
                if pos2 >= 0 and key_LR[pos1] == key_LR[pos2]:
                    comfort_seq1 = comfort_matrix[pos1, pos2]
                    freq_seq1 = bigram_freqs[i, j]
                    comfort_seq2 = comfort_matrix[pos2, pos1]
                    freq_seq2 = bigram_freqs[j, i]
                    curr_score = (comfort_seq1 * freq_seq1 + comfort_seq2 * freq_seq2)
                    bigram_component += curr_score
                    print(f"Bigram {i}-{j}: pos1={pos1}, pos2={pos2}, comfort1={comfort_seq1}, "
                          f"freq1={freq_seq1}, comfort2={comfort_seq2}, freq2={freq_seq2}, score={curr_score}")
    
    weighted_bigram = bigram_component * bigram_weight / 2.0
    weighted_letter = letter_component * letter_weight
    total_score = weighted_bigram + weighted_letter
    
    print(f"Final scores: total={total_score}, bigram={weighted_bigram}, letter={weighted_letter}")
    return float(total_score), float(weighted_bigram), float(weighted_letter)

class HeapItem:
    """Helper class to make heap operations work with numpy arrays."""
    def __init__(self, score: float, depth: int, mapping: np.ndarray, used: np.ndarray):
        self.score = score
        self.depth = depth
        self.mapping = mapping
        self.used = used
    
    def __lt__(self, other):
        return self.score < other.score

def branch_and_bound_optimal(
    arrays: tuple,
    weights: tuple,
    config: dict,
    n_solutions: int = 5,
    memory_threshold_gb: float = 0.9
) -> List[Tuple[float, Dict[str, str], Dict]]:
    """
    Branch and bound implementation with support for assigned letters.
    """
    # Get letters and keys from config
    letters_to_assign = config['optimization']['letters_to_assign']
    keys_to_assign = config['optimization']['keys_to_assign']
    letters_to_constrain = config['optimization'].get('letters_to_constrain', '')
    keys_to_constrain = config['optimization'].get('keys_to_constrain', '')
    letters_assigned = config['optimization'].get('letters_assigned', '')
    keys_assigned = config['optimization'].get('keys_assigned', '')

    # Convert to lowercase for consistency
    letters_to_assign = letters_to_assign.lower()
    keys_to_assign = keys_to_assign.upper()  # Convention: keys are uppercase
    letters_to_constrain = letters_to_constrain.lower()
    keys_to_constrain = keys_to_constrain.upper()
    letters_assigned = letters_assigned.lower()
    keys_assigned = keys_assigned.upper()

    n_letters_to_assign = len(letters_to_assign)
    n_keys_to_assign = len(keys_to_assign)
    is_full_layout = n_letters_to_assign == n_keys_to_assign
    
    key_LR, comfort_matrix, letter_freqs, bigram_freqs = arrays
    bigram_weight, letter_weight = weights
    
    # Initialize candidates list and top solutions
    candidates = []
    top_n_solutions = []
    worst_top_n_score = float('-inf')
    
    # Initialize mapping with all positions as -1
    initial_mapping = np.full(n_letters_to_assign, -1, dtype=np.int32)
    
    # Create initial used positions array
    initial_used = np.zeros(n_keys_to_assign, dtype=bool)
    
    # Handle assigned letters if any
    if letters_assigned:
        letter_to_idx = {letter: idx for idx, letter in enumerate(letters_to_assign)}
        key_to_pos = {key: pos for pos, key in enumerate(keys_to_assign)}
        
        for letter, key in zip(letters_assigned, keys_assigned):
            if letter in letter_to_idx and key in key_to_pos:
                idx = letter_to_idx[letter]
                pos = key_to_pos[key]
                initial_mapping[idx] = pos
                initial_used[pos] = True
    
    # Find first unassigned position for start depth
    start_depth = 0
    while start_depth < n_letters_to_assign and initial_mapping[start_depth] >= 0:
        start_depth += 1
    
    # Calculate total nodes for progress tracking
    total_nodes = 0
    remaining_letters = n_letters_to_assign - len(letters_assigned)
    for depth in range(remaining_letters):
        if is_full_layout:
            total_nodes += perm(n_keys_to_assign - len(letters_assigned) - depth, 1)
        else:
            total_nodes += comb(n_keys_to_assign - len(letters_assigned) - depth, 1)
    
    # Calculate initial score
    #score_tuple = calculate_layout_score_numba(
    score_tuple = calculate_layout_score_numba(
        initial_mapping,
        key_LR,
        comfort_matrix,
        bigram_freqs,
        letter_freqs,
        bigram_weight,
        letter_weight
    )
    initial_score = score_tuple[0]

    # Create HeapItem for initial state
    heapq.heappush(candidates, HeapItem(-initial_score, start_depth, initial_mapping, initial_used))
    
    processed_nodes = 0
    start_time = time.time()
    
    with tqdm(total=total_nodes, desc="Optimizing layout") as pbar:
        while candidates:
            if psutil.virtual_memory().percent > memory_threshold_gb * 100:
                print("\nWarning: Memory usage high, saving current best solutions")
                break
            
            # Get next candidate
            candidate = heapq.heappop(candidates)
            score = -candidate.score
            depth = candidate.depth
            mapping = candidate.mapping
            used = candidate.used
            
            # Process complete solutions
            if depth == n_letters_to_assign:
                score_tuple = calculate_layout_score_numba(
                    mapping,
                    key_LR,
                    comfort_matrix,
                    bigram_freqs,
                    letter_freqs,
                    bigram_weight,
                    letter_weight
                )
                total_score, bigram_score, letter_score = score_tuple
                
                # Convert numpy float32 to Python float
                total_score_float = float(total_score)
                bigram_score_float = float(bigram_score)
                letter_score_float = float(letter_score)
                
                if (len(top_n_solutions) < n_solutions or 
                    total_score_float > worst_top_n_score):
                    # Create solution tuple with Python types, not numpy arrays
                    solution = (
                        total_score_float,
                        bigram_score_float,
                        letter_score_float,
                        mapping.tolist()  # Convert numpy array to list
                    )
                    
                    # Create heap item with Python types
                    heap_item = (-total_score_float, len(top_n_solutions), solution)  # Add index as tiebreaker
                    heapq.heappush(top_n_solutions, heap_item)
                    
                    if len(top_n_solutions) > n_solutions:
                        heapq.heappop(top_n_solutions)
                    
                    if top_n_solutions:
                        worst_top_n_score = -top_n_solutions[0][0]
    
                pbar.update(1)
                processed_nodes += 1
                continue
            
            # Prune incomplete solutions using upper bound
            if len(top_n_solutions) >= n_solutions:
                upper_bound = calculate_upper_bound(
                    mapping, depth, key_LR, comfort_matrix,
                    letter_freqs, bigram_freqs, bigram_weight, letter_weight
                )
                if upper_bound <= worst_top_n_score:
                    remaining_levels = n_letters_to_assign - depth
                    skipped_nodes = sum(
                        perm(n_keys_to_assign - d, 1) if is_full_layout else comb(n_keys_to_assign - d, 1)
                        for d in range(depth, depth + remaining_levels)
                    )
                    pbar.update(skipped_nodes)
                    processed_nodes += skipped_nodes
                    continue
            
            # Try each available position, respecting constraints
            nodes_at_depth = 0
            current_letter = letters_to_assign[depth]
            
            # Skip if letter is already assigned (check initial_mapping)
            if initial_mapping[depth] >= 0:
                # Create new state with same mapping but increment depth
                heapq.heappush(
                    candidates,
                    HeapItem(-score, depth + 1, mapping, used)
                )
                continue
                
            for pos in range(n_keys_to_assign):
                if not used[pos]:
                    # Skip if position is already assigned
                    if pos in initial_mapping[initial_mapping >= 0]:
                        continue
                        
                    # Check constraints
                    if current_letter in letters_to_constrain:
                        if keys_to_assign[pos] not in keys_to_constrain:
                            continue
                    elif keys_to_assign[pos] in keys_to_constrain:
                        if current_letter not in letters_to_constrain:
                            continue
                                      
                    # Create new state
                    new_mapping = mapping.copy()
                    new_mapping[depth] = pos  
                    new_used = used.copy()
                    new_used[pos] = True
                    
                    # Calculate score for new state
                    score_tuple = calculate_layout_score_numba(  
                        new_mapping,
                        key_LR,
                        comfort_matrix,
                        bigram_freqs,
                        letter_freqs,
                        bigram_weight,
                        letter_weight
                    )
                    new_score = score_tuple[0]
                    
                    # Add to candidates if promising
                    if len(top_n_solutions) < n_solutions or new_score > worst_top_n_score:
                        heapq.heappush(
                            candidates,
                            HeapItem(-new_score, depth + 1, new_mapping, new_used)
                        )
                    
                    nodes_at_depth += 1
            
            pbar.update(nodes_at_depth)
            processed_nodes += nodes_at_depth
            
            # Update progress and ETA
            if processed_nodes % 100 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    nodes_per_second = processed_nodes / elapsed
                    remaining_nodes = total_nodes - processed_nodes
                    eta_seconds = remaining_nodes / nodes_per_second if nodes_per_second > 0 else 0
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    pbar.set_postfix({
                        'ETA': eta_str,
                        'Memory': f"{psutil.Process().memory_info().rss / 1024**3:.1f}GB"
                    })
    
    # Convert solutions
    solutions = []
    while top_n_solutions:
        _, _, solution = heapq.heappop(top_n_solutions)
        total_score, bigram_score, letter_score, mapping_list = solution
        # Convert back to numpy array if needed
        mapping = np.array(mapping_list, dtype=np.int32)
        letter_mapping = dict(zip(letters_to_assign, [keys_to_assign[i] for i in mapping]))
        solutions.append((
            total_score,
            letter_mapping,
            {'total': {
                'total_score': total_score,
                'bigram_score': bigram_score,
                'letter_score': letter_score
            }}
        ))
    
    return list(reversed(solutions))

def optimize_layout(config: dict) -> None:
    """
    Main optimization function with improved memory estimation and runtime prediction.
    """
    # Validate configuration
    letters_to_assign = config['optimization']['letters_to_assign']
    keys_to_assign = config['optimization']['keys_to_assign']
    letters_to_constrain = config['optimization']['letters_to_constrain']
    keys_to_constrain = config['optimization']['keys_to_constrain']
    letters_assigned = config['optimization'].get('letters_assigned', '')
    keys_assigned = config['optimization'].get('keys_assigned', '')
    
    validate_optimization_inputs(config)

    is_full_layout = len(letters_to_assign) == len(keys_to_assign)
    
    # Calculate arrangements
    if is_full_layout:
        total_arrangements = perm(len(keys_to_assign), len(letters_to_assign))
    else:
        positions_choices = comb(len(keys_to_assign), len(letters_to_assign))
        letter_arrangements = factorial(len(letters_to_assign))
        total_arrangements = positions_choices * letter_arrangements
    
    # Print initial information
    print(f"Letters to arrange:      {letters_to_assign.upper()}")
    print(f"Available positions:     {keys_to_assign.upper()}")
    print(f"Letters to constrain:    {letters_to_constrain.upper()}")
    print(f"Positions to constrain:  {keys_to_constrain.upper()}")
    print(f"Layout type:             {'Full (N=M)' if is_full_layout else 'Partial (N<M)'}")
    print(f"Total arrangements:      {total_arrangements:,}")
    if not is_full_layout:
        print(f"  - Position choices:    {positions_choices:,}")
        print(f"  - Letter arrangements: {letter_arrangements:,}")
    
    # Calculate memory requirements with more accurate estimation
    memory_estimate = estimate_memory_requirements(len(letters_to_assign), len(keys_to_assign), max_candidates=1000)
    
    print("\nMemory and Runtime Estimates:")
    print(f"Estimated memory usage:  {memory_estimate['estimated_gb']:.2f} GB")
    print(f"Available memory:        {memory_estimate['available_memory_gb']:.2f} GB")
    print(f"Memory sufficient:       {memory_estimate['memory_sufficient']}")
    print(f"Estimated runtime:       {memory_estimate['estimated_runtime']}")
    print("\nSearch space details:")
    print(f"Max queue size:          {memory_estimate['details']['max_queue_size']:,}")
    print(f"Avg branching factor:    {memory_estimate['details']['avg_branching_factor']:.1f}")
    print(f"Memory per candidate:    {memory_estimate['details']['bytes_per_candidate']:,} bytes")
    
    #user_continue = input("\nContinue with optimization? (y/n): ")
    #if user_continue.lower() != 'y':
    #    print("Optimization cancelled.")
    #    return
    
    if not memory_estimate['memory_sufficient']:
        raise MemoryError("Insufficient memory available for optimization")
        
    # Show initial keyboard
    visualize_keyboard_layout(
        mapping=None,
        title="Keys to optimize",
        letters_to_display=letters_assigned,
        keys_to_display=keys_assigned,
        config=config
    )
        
    # Load and normalize scores
    norm_2key_scores, norm_1key_scores = load_and_normalize_comfort_scores(config)
    norm_letter_freqs, norm_bigram_freqs = load_and_normalize_frequencies(
        onegrams, onegram_frequencies_array, bigrams, bigram_frequencies_array
    )
    
    # Get scoring weights
    bigram_weight = config['optimization']['scoring']['bigram_weight']
    letter_weight = config['optimization']['scoring']['letter_weight']
    
    # Get right-hand keys
    right_keys = set()
    for row in ['top', 'home', 'bottom']:
        right_keys.update(config['layout']['positions']['right'][row])
    
    # Prepare arrays and run optimization
    arrays = prepare_arrays(
        letters_to_assign, keys_to_assign, right_keys,
        norm_2key_scores, norm_1key_scores,
        norm_bigram_freqs, norm_letter_freqs
    )
    weights = (bigram_weight, letter_weight)
    
    # Get number of layouts from config
    n_layouts = config['optimization'].get('nlayouts', 5)
    
    # Run optimization
    results = branch_and_bound_optimal(
        arrays=arrays,
        weights=weights,
        config=config,
        n_solutions=n_layouts
    )

    # Sort and save results
    sorted_results = sorted(
        results,
        key=lambda x: (
            x[0],  # total_score
            x[2]['total']['bigram_score'],  # use bigram score as secondary sort
            x[2]['total']['letter_score']   # use letter score as tertiary sort
        ),
        reverse=True
    )
    
    print_top_results(
        results=sorted_results,
        config=config,
        n=None,
        letters_to_display=letters_assigned,
        keys_to_display=keys_assigned
    )
    save_results_to_csv(sorted_results, config)

#--------------------------------------------------------------------
# Pipeline
#--------------------------------------------------------------------
if __name__ == "__main__":
    try:
        start_time = time.time()

        # Load configuration
        config = load_config('config.yaml')
    
        # Optimize the layout
        optimize_layout(config)

        elapsed = time.time() - start_time
        print(f"Done! Total runtime: {timedelta(seconds=int(elapsed))}")

    except Exception as e:
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()


    

