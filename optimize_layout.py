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
import pickle

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
    """
    Validate optimization inputs from config, safely handling None values.
    """
    # Safely get and convert values
    letters_to_assign = config['optimization'].get('letters_to_assign', '')
    letters_to_assign = set(letters_to_assign.lower() if letters_to_assign else '')

    keys_to_assign = config['optimization'].get('keys_to_assign', '')
    keys_to_assign = set(keys_to_assign.upper() if keys_to_assign else '')

    letters_to_constrain = config['optimization'].get('letters_to_constrain', '')
    letters_to_constrain = set(letters_to_constrain.lower() if letters_to_constrain else '')

    keys_to_constrain = config['optimization'].get('keys_to_constrain', '')
    keys_to_constrain = set(keys_to_constrain.upper() if keys_to_constrain else '')

    letters_assigned = config['optimization'].get('letters_assigned', '')
    letters_assigned = set(letters_assigned.lower() if letters_assigned else '')

    keys_assigned = config['optimization'].get('keys_assigned', '')
    keys_assigned = set(keys_assigned.upper() if keys_assigned else '')

    # Check for duplicates
    if len(letters_to_assign) != len(config['optimization']['letters_to_assign']):
        raise ValueError(f"Duplicate letters in letters_to_assign: {config['optimization']['letters_to_assign']}")
    if len(keys_to_assign) != len(config['optimization']['keys_to_assign']):
        raise ValueError(f"Duplicate keys in keys_to_assign: {config['optimization']['keys_to_assign']}")
    if len(letters_assigned) != len(config['optimization']['letters_assigned']):
        raise ValueError(f"Duplicate letters in letters_assigned: {config['optimization']['letters_assigned']}")
    if len(keys_assigned) != len(config['optimization']['keys_assigned']):
        raise ValueError(f"Duplicate keys in keys_assigned: {config['optimization']['keys_assigned']}")
    
    # Check that assigned letters and keys have matching lengths
    if len(letters_assigned) != len(keys_assigned):
        raise ValueError(
            f"Mismatched number of assigned letters ({len(letters_assigned)}) "
            f"and assigned keys ({len(keys_assigned)})"
        )

    # Check no overlap between assigned and to_assign
    overlap = letters_assigned.intersection(letters_to_assign)
    if overlap:
        raise ValueError(f"letters_to_assign contains assigned letters: {overlap}")
    overlap = keys_assigned.intersection(keys_to_assign)
    if overlap:
        raise ValueError(f"keys_to_assign contains assigned keys: {overlap}")

    # Check that we have enough positions
    if len(letters_to_assign) > len(keys_to_assign):
        raise ValueError(
            f"More letters to assign ({len(letters_to_assign)}) "
            f"than available positions ({len(keys_to_assign)})"
        )

    # Check constraints are subsets
    if not letters_to_constrain.issubset(letters_to_assign):
        invalid = letters_to_constrain - letters_to_assign
        raise ValueError(f"letters_to_constrain contains letters not in letters_to_assign: {invalid}")
    if not keys_to_constrain.issubset(keys_to_assign):
        invalid = keys_to_constrain - keys_to_assign
        raise ValueError(f"keys_to_constrain contains keys not in keys_to_assign: {invalid}")

    # Check if we have enough constraint keys for constraint letters
    if len(letters_to_constrain) > len(keys_to_constrain):
        raise ValueError(
            f"Not enough constraint keys ({len(keys_to_constrain)}) "
            f"for constraint letters ({len(letters_to_constrain)})"
        )
    
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
    #print("\nPreparing arrays:")
    #print(f"Letters to assign: {letters_to_assign}")
    #print(f"Keys to assign: {keys_to_assign}")
    #print(f"Right keys: {right_keys}")
    
    n_letters_to_assign = len(letters_to_assign)
    n_keys_to_assign = len(keys_to_assign)
    
    key_LR = np.array([1 if k in right_keys else 0 for k in keys_to_assign], dtype=np.int8)
    #print(f"key_LR: {key_LR}")
    
    # Create comfort score matrix
    comfort_matrix = np.zeros((n_keys_to_assign, n_keys_to_assign), dtype=np.float32)
    for i, k1 in enumerate(keys_to_assign):
        for j, k2 in enumerate(keys_to_assign):
            if i == j:
                comfort_matrix[i, j] = norm_1key_scores.get(k1.lower(), 0.0)
            else:
                comfort_matrix[i, j] = norm_2key_scores.get((k1.lower(), k2.lower()), 0.0)
    #print(f"Comfort matrix shape: {comfort_matrix.shape}")
    
    # Create letter frequency array
    letter_freqs = np.array([
        norm_letter_freqs.get(l.lower(), 0.0) for l in letters_to_assign
    ], dtype=np.float32)
    #print(f"Letter frequencies: {letter_freqs}")
    
    # Create bigram frequency matrix
    bigram_freqs = np.zeros((n_letters_to_assign, n_letters_to_assign), dtype=np.float32)
    for i, l1 in enumerate(letters_to_assign):
        for j, l2 in enumerate(letters_to_assign):
            bigram_freqs[i, j] = norm_bigram_freqs.get((l1.lower(), l2.lower()), 0.0)
    #print(f"Bigram frequencies shape: {bigram_freqs.shape}")
    
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

def setup_checkpointing(config: dict, search_space: dict) -> dict:
    """
    Configure checkpointing based on search space size and estimated runtime.
    
    Returns:
        dict with checkpoint settings:
        - enabled: bool, whether checkpointing is enabled
        - interval: int, nodes between checkpoints
        - path: str, checkpoint file path
        - metadata: dict, search configuration data
    """
    # Estimate if checkpointing is needed based on search space size
    LONG_SEARCH_THRESHOLD = 1e9  # 1 billion nodes
    VERY_LONG_SEARCH_THRESHOLD = 1e12  # 1 trillion nodes
    
    total_nodes = search_space['total_nodes']
    estimated_nodes_per_second = 100000  # Conservative estimate
    estimated_runtime = total_nodes / estimated_nodes_per_second
    
    # Determine if we need checkpointing
    checkpointing_enabled = total_nodes > LONG_SEARCH_THRESHOLD
    
    if not checkpointing_enabled:
        return {'enabled': False}
    
    # Calculate checkpoint interval based on search size
    if total_nodes > VERY_LONG_SEARCH_THRESHOLD:
        checkpoint_interval = 1000000  # Every million nodes for very long searches
    else:
        checkpoint_interval = 100000   # Every 100k nodes for long searches
    
    # Create checkpoint directory if needed
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint filename with search parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_params = (
        f"L{len(config['optimization']['letters_to_assign'])}"
        f"_K{len(config['optimization']['keys_to_assign'])}"
        f"_C{len(config['optimization']['letters_to_constrain'])}"
    )
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"search_{search_params}_{timestamp}.checkpoint"
    )
    
    # Prepare metadata for checkpoint
    metadata = {
        'config': config,
        'search_space': search_space,
        'timestamp_start': timestamp,
        'estimated_runtime': str(timedelta(seconds=int(estimated_runtime))),
        'total_nodes': total_nodes
    }
    
    return {
        'enabled': True,
        'interval': checkpoint_interval,
        'path': checkpoint_path,
        'metadata': metadata,
        'last_save': time.time()
    }

def validate_mapping(mapping: np.ndarray, constrained_letter_indices: set, constrained_positions: set) -> bool:
    """Validate that mapping follows all constraints."""
    for idx in constrained_letter_indices:
        if mapping[idx] >= 0 and mapping[idx] not in constrained_positions:
            return False
    return True

def save_checkpoint(
    checkpoint_path: str,
    current_state: dict
) -> None:
    """Save current search state to checkpoint file."""
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, 'wb') as f:
        pickle.dump(current_state, f)
    os.replace(temp_path, checkpoint_path)

def load_checkpoint(checkpoint_path: str) -> dict:
    """Load search state from checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None

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
    
    # Calculate theoretical maximum arrangements
    total_arrangements = perm(n_positions, n_letters)
    
    # Estimate actual search space (considering constraints and pruning)
    # Assume we'll only explore about 1% of the theoretical space due to pruning
    estimated_search_space = total_arrangements // 100
    
    # Use more realistic nodes per second based on actual performance
    estimated_nodes_per_sec = 100000  # Based on observed performance
    estimated_seconds = estimated_search_space / estimated_nodes_per_sec
    estimated_runtime = str(timedelta(seconds=int(estimated_seconds)))
    
    return {
        'estimated_bytes': total_bytes,
        'estimated_mb': total_mb,
        'estimated_gb': total_gb,
        'available_memory_gb': available_gb,
        'memory_sufficient': total_bytes < (available_memory * 0.8),
        'estimated_runtime': estimated_runtime,
        'search_space_details': {
            'theoretical_arrangements': total_arrangements,
            'estimated_search_space': estimated_search_space,
            'nodes_per_second': estimated_nodes_per_sec
        }
    }

@jit(nopython=True, fastmath=True)
def calculate_layout_score(
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

def exact_remaining_arrangements(
    constrained_letters: set,
    constrained_positions: set,
    unconstrained_letters: int,
    unconstrained_positions: int
) -> int:
    """
    Calculate exact number of possible arrangements for remaining letters.
    Ensures all inputs to perm() are non-negative.
    """
    # Validate inputs
    if unconstrained_letters < 0 or unconstrained_positions < 0:
        print(f"Warning: Invalid inputs to exact_remaining_arrangements:")
        print(f"constrained_letters: {len(constrained_letters)}")
        print(f"constrained_positions: {len(constrained_positions)}")
        print(f"unconstrained_letters: {unconstrained_letters}")
        print(f"unconstrained_positions: {unconstrained_positions}")
        return 0

    if not constrained_letters:
        return perm(unconstrained_positions, unconstrained_letters) if unconstrained_positions >= unconstrained_letters else 0
        
    n_constrained = len(constrained_letters)
    n_constrained_pos = len(constrained_positions)
    
    # Ensure we have enough positions for constrained letters
    if n_constrained_pos < n_constrained:
        return 0
        
    constrained_arrangements = perm(n_constrained_pos, n_constrained)
    
    # Calculate arrangements for unconstrained letters
    if unconstrained_letters > 0:
        if unconstrained_positions < unconstrained_letters:
            return 0
        unconstrained_arrangements = perm(unconstrained_positions, unconstrained_letters)
    else:
        unconstrained_arrangements = 1
    
    return (constrained_arrangements * factorial(n_constrained) * 
            unconstrained_arrangements * factorial(unconstrained_letters))

def calculate_total_nodes(
    n_letters: int,
    n_positions: int,
    letters_to_constrain: set,
    keys_to_constrain: set,
    letters_assigned: set,
    keys_assigned: set
) -> dict:
    """
    Calculate exact number of nodes in search tree.
    """
    # First level: account for pre-assigned letters
    available_positions = n_positions - len(keys_assigned)
    remaining_letters = n_letters - len(letters_assigned)
    
    # Calculate constrained letter arrangements
    n_constrained = len(letters_to_constrain)
    n_constrained_positions = len(keys_to_constrain)
    
    # For each depth, calculate number of possibilities
    depth_distributions = {}
    total_nodes = 0
    
    # First handle constrained letters
    for depth in range(n_constrained):
        positions_at_depth = n_constrained_positions - depth
        arrangements_at_depth = positions_at_depth * factorial(remaining_letters - depth)
        depth_distributions[depth] = arrangements_at_depth
        total_nodes += arrangements_at_depth
    
    # Then handle unconstrained letters
    remaining_positions = available_positions - n_constrained
    for depth in range(n_constrained, remaining_letters):
        positions_at_depth = remaining_positions - (depth - n_constrained)
        if positions_at_depth > 0:
            arrangements_at_depth = positions_at_depth * factorial(remaining_letters - depth)
            depth_distributions[depth] = arrangements_at_depth
            total_nodes += arrangements_at_depth
    
    return {
        'total_nodes': total_nodes,
        'constrained_arrangements': factorial(n_constrained) * perm(n_constrained_positions, n_constrained),
        'unconstrained_arrangements': factorial(remaining_letters - n_constrained) * 
                                      perm(remaining_positions, remaining_letters - n_constrained),
        'depth_distributions': depth_distributions,
        'details': {
            'available_positions': available_positions,
            'remaining_letters': remaining_letters,
            'constrained_letters': n_constrained,
            'constrained_positions': n_constrained_positions
        }
    }

@jit(nopython=True, fastmath=True)
def calculate_upper_bound(
    mapping: np.ndarray,
    depth: int,
    used: np.ndarray,
    key_LR: np.ndarray,
    comfort_matrix: np.ndarray,
    letter_freqs: np.ndarray,
    bigram_freqs: np.ndarray,
    bigram_weight: float,
    letter_weight: float
) -> float:
    """Calculate accurate upper bound on best possible score from this node."""
    # Get current score from assigned letters
    current_score = calculate_layout_score(
        mapping, key_LR, comfort_matrix,
        bigram_freqs, letter_freqs, bigram_weight, letter_weight
    )[0]
    
    # All possible directed pairs:
    #   - Pairs between already assigned letters (already accounted for in current_score)
    #   - Pairs between assigned and unassigned letters
    #   - Pairs between unassigned letters
    n_total = len(mapping)  # Total number of letters
    n_pairs = n_total * (n_total - 1)  

    # For remaining letters, use best possible scores
    n_remaining = n_total - depth
    if n_remaining == 0:
        return current_score
        
    # Get unassigned letters and available positions
    unassigned = np.where(mapping < 0)[0]
    available_positions = np.where(~used)[0]
    
    # Get comfort scores only for available positions
    remaining_comfort_scores = comfort_matrix[available_positions][:,available_positions].flatten()
    comfort_scores = np.sort(remaining_comfort_scores)[-n_remaining:][::-1]  # highest to lowest
    
    # Get frequencies for unassigned letters
    remaining_freqs = letter_freqs[unassigned]
    remaining_freqs = np.sort(remaining_freqs)[::-1]  # highest to lowest
    
    # Maximum possible letter score contribution
    max_letter_score = np.sum(comfort_scores * remaining_freqs) * letter_weight
    
    # Maximum possible bigram score contribution
    # In calculate_layout_score:
    #   bigram_component += (comfort_seq1 * freq_seq1 + comfort_seq2 * freq_seq2)
    #   weighted_bigram = bigram_component * bigram_weight / 2.0
    max_comfort = np.max(comfort_matrix)
    max_bigram_freq = np.max(bigram_freqs)
    max_bigram_score = n_pairs * max_comfort * max_bigram_freq * bigram_weight / 2.0
    
    return float(current_score + max_letter_score + max_bigram_score)

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
    Branch and bound implementation with phased search for constrained letters,
    improved progress monitoring, and checkpoint saving.
    """
    # Get letters and keys from config
    letters_to_assign = config['optimization']['letters_to_assign']
    keys_to_assign = config['optimization']['keys_to_assign']
    letters_to_constrain = config['optimization'].get('letters_to_constrain', '')
    keys_to_constrain = config['optimization'].get('keys_to_constrain', '')
    letters_assigned = config['optimization'].get('letters_assigned', '')
    keys_assigned = config['optimization'].get('keys_assigned', '')

    # Convert to lowercase/uppercase for consistency
    letters_to_assign = letters_to_assign.lower()
    keys_to_assign = keys_to_assign.upper()
    letters_to_constrain = letters_to_constrain.lower()
    keys_to_constrain = keys_to_constrain.upper()
    letters_assigned = letters_assigned.lower()
    keys_assigned = keys_assigned.upper()

    # Initialize dimensions and arrays
    n_letters_to_assign = len(letters_to_assign)
    n_keys_to_assign = len(keys_to_assign)
    
    key_LR, comfort_matrix, letter_freqs, bigram_freqs = arrays
    bigram_weight, letter_weight = weights
    
    # Set up constraint tracking
    constrained_letters = set(letters_to_constrain.lower())
    constrained_positions = set(i for i, key in enumerate(keys_to_assign) 
                              if key.upper() in keys_to_constrain.upper())
    constrained_letter_indices = set(i for i, letter in enumerate(letters_to_assign) 
                                   if letter in constrained_letters)
    
    print("\nConstraint tracking:")
    print(f"Constrained letters at indices: {sorted(constrained_letter_indices)}")
    print(f"Constrained positions: {sorted(constrained_positions)}")
    
    # Set up checkpointing
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"search_L{n_letters_to_assign}_K{n_keys_to_assign}_C{len(constrained_letters)}_{timestamp}.checkpoint"
    )
    checkpoint_interval = 1000000  # Save every million nodes

    # Initialize search structures
    candidates = []
    top_n_solutions = []
    worst_top_n_score = float('-inf')
    
    # Initialize mapping and used positions
    initial_mapping = np.full(n_letters_to_assign, -1, dtype=np.int32)
    initial_used = np.zeros(n_keys_to_assign, dtype=bool)
    
    # Handle pre-assigned letters
    if letters_assigned:
        letter_to_idx = {letter: idx for idx, letter in enumerate(letters_to_assign)}
        key_to_pos = {key: pos for pos, key in enumerate(keys_to_assign)}
        
        for letter, key in zip(letters_assigned, keys_assigned):
            if letter in letter_to_idx and key in key_to_pos:
                idx = letter_to_idx[letter]
                pos = key_to_pos[key]
                initial_mapping[idx] = pos
                initial_used[pos] = True
                print(f"Pre-assigned: {letter} -> {key} (position {pos})")
    
    # Calculate initial score
    score_tuple = calculate_layout_score(
        initial_mapping,
        key_LR,
        comfort_matrix,
        bigram_freqs,
        letter_freqs,
        bigram_weight,
        letter_weight
    )
    initial_score = score_tuple[0]
    
    # Initialize tracking variables
    processed_nodes = 0
    start_time = time.time()
    last_update_time = start_time
    last_checkpoint_time = start_time
    update_interval = 5  # seconds
    
    # Calculate phase nodes
    total_nodes_first_phase = perm(len(constrained_positions), len(constrained_letters))
    remaining_positions = n_keys_to_assign - len(constrained_letters)
    remaining_letters = n_letters_to_assign - len(constrained_letters) 

    # For second phase, need to account for positions still available
    available_unconstrained_positions = remaining_positions - (len(constrained_positions) - len(constrained_letters))
    total_nodes_second_phase = perm(available_unconstrained_positions, remaining_letters)

    print(f"\nPhase 1 (Constrained letters): {total_nodes_first_phase:,} nodes")
    print(f"Phase 2 (Remaining letters): {total_nodes_second_phase:,} nodes")
    
    # Start search
    heapq.heappush(candidates, HeapItem(-initial_score, 0, initial_mapping, initial_used))
    
    with tqdm(total=total_nodes_first_phase + total_nodes_second_phase, 
              desc="Optimizing layout",
              unit='nodes') as pbar:
        current_phase = 1
        phase_start_time = time.time()

        while candidates:
            # Memory check
            if psutil.virtual_memory().percent > memory_threshold_gb * 100:
                print("\nWarning: Memory usage high, saving checkpoint and stopping")
                save_checkpoint(checkpoint_path, {
                    'candidates': candidates,
                    'top_n_solutions': top_n_solutions,
                    'processed_nodes': processed_nodes,
                    'current_phase': current_phase,
                    'phase_start_time': phase_start_time,
                    'start_time': start_time
                })
                break
            
            # Get next candidate
            candidate = heapq.heappop(candidates)
            score = -candidate.score
            depth = candidate.depth
            mapping = candidate.mapping
            used = candidate.used
            
            # Process complete solutions
            if depth == n_letters_to_assign:
                if not validate_mapping(mapping, constrained_letter_indices, constrained_positions):
                    continue  # Skip invalid solutions
                    
                score_tuple = calculate_layout_score(
                    mapping,
                    key_LR,
                    comfort_matrix,
                    bigram_freqs,
                    letter_freqs,
                    bigram_weight,
                    letter_weight
                )
                total_score, bigram_score, letter_score = score_tuple

                if (len(top_n_solutions) < n_solutions or 
                    total_score > worst_top_n_score):
                    solution = (
                        float(total_score),
                        float(bigram_score),
                        float(letter_score),
                        mapping.tolist()
                    )
                    heap_item = (-float(total_score), len(top_n_solutions), solution)
                    heapq.heappush(top_n_solutions, heap_item)
                    
                    if len(top_n_solutions) > n_solutions:
                        heapq.heappop(top_n_solutions)
                    
                    if top_n_solutions:
                        worst_top_n_score = -top_n_solutions[0][0]
                        print(f"\nNew solution found (score: {total_score:.4f})")
                        print(f"Current best score: {-top_n_solutions[0][0]:.4f}")
                
                processed_nodes += 1
                pbar.update(1)
                continue
            
            # Only prune if we have enough solutions and can prove this branch won't yield better ones
            if len(top_n_solutions) >= n_solutions:
                upper_bound = calculate_upper_bound(
                    mapping, depth, used, key_LR, comfort_matrix,
                    letter_freqs, bigram_freqs, bigram_weight, letter_weight
                )
                
                margin = 1e-8 * abs(worst_top_n_score)  # Relative margin for precision error

                if upper_bound < worst_top_n_score - margin:
                    processed_nodes += 1
                    pbar.update(1)
                    continue
            
            # Try each available position
            nodes_at_depth = 0
            current_letter = letters_to_assign[depth]
            is_constrained_letter = current_letter in constrained_letters

            # Get valid positions for current letter
            valid_positions = []
            if is_constrained_letter:
                # Constrained letter can only go in constrained positions
                valid_positions = [pos for pos in constrained_positions if not used[pos]]
            else:
                # Count remaining constrained letters and positions
                remaining_constrained_letters = sum(1 for i in constrained_letter_indices if mapping[i] < 0)
                remaining_constrained_positions = [pos for pos in constrained_positions if not used[pos]]
                
                if remaining_constrained_letters > len(remaining_constrained_positions):
                    # Not enough constrained positions left, this branch is invalid
                    continue
                
                # Can use any position except those needed for remaining constrained letters
                positions_needed_for_constrained = set(remaining_constrained_positions[:remaining_constrained_letters])
                valid_positions = [pos for pos in range(n_keys_to_assign) 
                                   if not used[pos] and pos not in positions_needed_for_constrained]
            
            for pos in valid_positions:
                # Create new state
                new_mapping = mapping.copy()
                new_mapping[depth] = pos
                new_used = used.copy()
                new_used[pos] = True
                
                # Validate partial solution
                if not validate_mapping(new_mapping, constrained_letter_indices, constrained_positions):
                    continue
                    
                # Calculate score
                score_tuple = calculate_layout_score(
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
            
            processed_nodes += nodes_at_depth
            pbar.update(nodes_at_depth)
            
            # Update progress and save checkpoint if needed
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                elapsed = current_time - phase_start_time
                if elapsed > 0:
                    nodes_per_second = processed_nodes / elapsed
                    
                    # Detect phase change
                    if current_phase == 1 and all(mapping[i] >= 0 for i in constrained_letter_indices):
                        print("\nPhase 1 complete!")
                        print(f"Time taken: {str(timedelta(seconds=int(elapsed)))}")
                        current_phase = 2
                        phase_start_time = current_time
                        processed_nodes = 0
                    
                    # Calculate remaining time for current phase
                    if current_phase == 1:
                        remaining_nodes = total_nodes_first_phase - processed_nodes
                    else:
                        remaining_nodes = total_nodes_second_phase - processed_nodes
                    
                    eta_seconds = remaining_nodes / nodes_per_second if nodes_per_second > 0 else float('inf')
                    
                    pbar.set_postfix({
                        'Phase': f"{current_phase}/2",
                        'Nodes/sec': f"{nodes_per_second:.0f}",
                        'ETA': str(timedelta(seconds=int(eta_seconds))),
                        'Memory': f"{psutil.Process().memory_info().rss / 1024**3:.1f}GB"
                    })
                
                last_update_time = current_time
            
            # Save checkpoint if needed
            if processed_nodes - last_checkpoint_time >= checkpoint_interval:
                save_checkpoint(checkpoint_path, {
                    'candidates': candidates,
                    'top_n_solutions': top_n_solutions,
                    'processed_nodes': processed_nodes,
                    'current_phase': current_phase,
                    'phase_start_time': phase_start_time,
                    'start_time': start_time
                })
                last_checkpoint_time = processed_nodes
    
    # Convert solutions
    solutions = []
    print(f"\nConverting {len(top_n_solutions)} solutions...")
    while top_n_solutions:
        _, _, solution = heapq.heappop(top_n_solutions)
        total_score, bigram_score, letter_score, mapping_list = solution
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
    Main optimization function.
    """
    # Get parameters from config
    letters_to_assign = config['optimization']['letters_to_assign']
    keys_to_assign = config['optimization']['keys_to_assign']
    letters_to_constrain = config['optimization'].get('letters_to_constrain', '')
    keys_to_constrain = config['optimization'].get('keys_to_constrain', '')
    letters_assigned = config['optimization'].get('letters_assigned', '')
    keys_assigned = config['optimization'].get('keys_assigned', '')
    
    print("\nConfiguration:")
    print(f"Letters to assign: {letters_to_assign}")
    print(f"Keys to assign: {keys_to_assign}")
    print(f"Letters to constrain: {letters_to_constrain}")
    print(f"Keys to constrain: {keys_to_constrain}")
    print(f"Letters assigned: {letters_assigned}")
    print(f"Keys assigned: {keys_assigned}")
    
    # Validate configuration
    validate_optimization_inputs(config)
    
    # Calculate exact search space size
    search_space = calculate_total_nodes(
        n_letters=len(letters_to_assign),
        n_positions=len(keys_to_assign),
        letters_to_constrain=set(letters_to_constrain),
        keys_to_constrain=set(keys_to_constrain),
        letters_assigned=set(letters_assigned),
        keys_assigned=set(keys_assigned)
    )
    
    print("\nSearch space analysis:")
    print(f"Total arrangements: {search_space['total_nodes']:,}")
    print(f"  - Constrained phase arrangements: {search_space['constrained_arrangements']:,}")
    print(f"  - Unconstrained phase arrangements: {search_space['unconstrained_arrangements']:,}")
    print("\nSearch tree by depth:")
    for depth, nodes in search_space['depth_distributions'].items():
        print(f"Depth {depth}: {nodes:,} nodes")
        
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
    
    print("\nStarting optimization with:")
    print(f"- {len(letters_to_constrain)} constrained letters: {letters_to_constrain}")
    print(f"- {len(keys_to_constrain)} constrained positions: {keys_to_constrain}")
    print(f"- Finding top {n_layouts} solutions")
    
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


    

