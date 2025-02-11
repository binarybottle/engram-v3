import os
import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Set, Tuple
from itertools import permutations, islice
from multiprocessing import Pool, cpu_count
from math import factorial
from tqdm import tqdm
import psutil
import time
from datetime import timedelta
import numba
from numba import jit, float64, boolean

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

def process_chunk_optimized(args):
    """
    Process a chunk of permutations using updated scoring function.
    """
    chunk, arrays, weights = args
    key_LR, comfort_matrix, letter_freqs, bigram_freqs = arrays
    bigram_weight, letter_weight = weights
    
    chunk_results = []
    for perm in chunk:
        total_score, bigram_score, letter_score = calculate_layout_score_numba(
            np.array(perm, dtype=np.int64),
            key_LR,
            comfort_matrix,
            bigram_freqs,
            letter_freqs,
            bigram_weight,
            letter_weight
        )
        chunk_results.append((total_score, bigram_score, letter_score, perm))
    
    return chunk_results

def evaluate_layouts_parallel(
    letters: str,
    keys: str,
    right_keys: Set[str],
    norm_2key_scores: Dict[Tuple[str, str], float],
    norm_1key_scores: Dict[str, float],
    norm_bigram_freqs: Dict[Tuple[str, str], float],
    norm_letter_freqs: Dict[str, float],
    bigram_weight: float,
    letter_weight: float,
    batch_size: int = 100000,
    n_processes: int = None
) -> list:
    """
    Evaluate layouts with separate component scores.
    """    
    start_time = time.time()

    if n_processes is None:
        n_processes = cpu_count() - 1
    
    # Prepare arrays once
    arrays = prepare_arrays(
        letters, keys, right_keys,
        norm_2key_scores, norm_1key_scores,
        norm_bigram_freqs, norm_letter_freqs
    )
    
    weights = (bigram_weight, letter_weight)
    
    # Generate indices for permutations
    indices = list(range(len(letters)))
    perms_iterator = permutations(indices)
    
    total_perms = factorial(len(letters))
    with tqdm(total=total_perms, desc="Evaluating layouts") as pbar:
        results = []
        while True:
            batch = list(islice(perms_iterator, batch_size))
            if not batch:
                break
            
            # Split batch for parallel processing
            chunks = np.array_split(batch, n_processes)
            chunk_args = [(chunk, arrays, weights) for chunk in chunks if len(chunk) > 0]
            
            # Process chunks in parallel
            with Pool(n_processes) as pool:
                chunk_results = pool.map(process_chunk_optimized, chunk_args)
                for res in chunk_results:
                    results.extend(res)
                    pbar.update(len(res))  # Update progress bar
                
                # Update estimated time remaining
                if len(results) > batch_size:  # Wait for some data to get better estimate
                    elapsed = time.time() - start_time
                    perms_per_second = len(results) / elapsed
                    eta = (total_perms - len(results)) / perms_per_second
                    pbar.set_postfix({'ETA': str(timedelta(seconds=int(eta)))})
    
    # Sort and convert indices back to letters
    results.sort(reverse=True, key=lambda x: x[0])

    # Convert results, using permutation indices correctly
    final_results = []
    for total_score, bigram_score, letter_score, perm in results:
        # Map the permutation indices to actual keys
        key_mapping = dict(zip(letters, [keys[i] for i in perm]))
        
        # Add debug output
        print(f"\nLayout score: {total_score:.4f}")
        print(f"Permutation: {perm}")
        print(f"Mapping: {key_mapping}")
        print(f"Component scores - Bigram: {bigram_score:.4f}, Letter: {letter_score:.4f}")
        
        final_results.append((
            total_score,
            key_mapping,
            {'total': {
                'total_score': total_score,
                'bigram_score': bigram_score,
                'letter_score': letter_score
            }}
        ))
    
    final_results.sort(reverse=True, key=lambda x: x[0])
    return final_results

def optimize_layout(config: dict) -> None:
    """
    Main optimization function using the optimized implementation.
    """
    # Load and normalize scores as before
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
    
    # Run optimized evaluation
    results = evaluate_layouts_parallel(
        letters=letters,
        keys=keys,
        right_keys=right_keys,
        norm_2key_scores=norm_2key_scores,
        norm_1key_scores=norm_1key_scores,
        norm_bigram_freqs=norm_bigram_freqs,
        norm_letter_freqs=norm_letter_freqs,
        bigram_weight=bigram_weight,
        letter_weight=letter_weight,
        batch_size=100000,
        n_processes=None  # set elsewhere to CPU count - 1
    )
    
    # Process top results as before
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


    

