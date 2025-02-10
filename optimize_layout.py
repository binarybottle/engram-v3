import os
import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Tuple
from itertools import permutations
from tqdm import tqdm

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
    Load and normalize comfort scores for both bigrams and individual keys.
    Returns normalized scores for both left and right hand positions.
    """
    # Load raw data
    bigram_df = pd.read_csv(config['paths']['input']['two_key_comfort_scores_file'])
    key_df = pd.read_csv(config['paths']['input']['one_key_comfort_scores_file'])
    
    # Get position mappings for right-to-left
    right_to_left = config['layout']['position_mappings']['right_to_left']
    left_to_right = {v: k for k, v in right_to_left.items()}
    
    # Normalize comfort scores (lower raw scores are better)
    bigram_min = bigram_df['comfort_score'].min()
    bigram_max = bigram_df['comfort_score'].max()
    key_min = key_df['comfort_score'].min()
    key_max = key_df['comfort_score'].max()
    
    # Create normalized bigram comfort scores dictionary
    norm_bigram_scores = {}
    for _, row in bigram_df.iterrows():
        # Normalize score (flip so 1.0 is most comfortable)
        norm_score = 1 - ((row['comfort_score'] - bigram_min) / (bigram_max - bigram_min))
        
        # Add left hand bigrams
        key1, key2 = row['first_char'], row['second_char']
        norm_bigram_scores[(key1, key2)] = norm_score
        
        # Mirror to right hand if both keys have right-hand equivalents
        if key1 in left_to_right and key2 in left_to_right:
            right_key1 = left_to_right[key1]
            right_key2 = left_to_right[key2]
            norm_bigram_scores[(right_key1, right_key2)] = norm_score
    
    # Create normalized single key comfort scores dictionary
    norm_key_scores = {}
    for _, row in key_df.iterrows():
        # Normalize score (flip so 1.0 is most comfortable)
        norm_score = 1 - ((row['comfort_score'] - key_min) / (key_max - key_min))
        
        # Add left hand key
        key = row['key']
        norm_key_scores[key] = norm_score
        
        # Mirror to right hand if key has right-hand equivalent
        if key in left_to_right:
            right_key = left_to_right[key]
            norm_key_scores[right_key] = norm_score
    
    return norm_bigram_scores, norm_key_scores

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
    
    # Normalize letter frequencies to 0-1
    letter_min = min(log_letter_freqs.values())
    letter_max = max(log_letter_freqs.values())
    norm_letter_freqs = {
        k: (v - letter_min) / (letter_max - letter_min)
        for k, v in log_letter_freqs.items()
    }
    
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
    output_path = os.path.join(config['paths']['output']['layout_results_folder'], f"layout_results_{timestamp}.csv")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with configuration info
        writer.writerow(['Configuration'])
        writer.writerow(['Letters', config['optimization']['letters']])
        writer.writerow(['Keys', config['optimization']['keys']])
        writer.writerow(['Bigram Frequency Weight', config['optimization']['scoring']['bigram_frequency_weight']])
        writer.writerow(['Letter Frequency Weight', config['optimization']['scoring']['letter_frequency_weight']])
        writer.writerow(['Comfort Weight', config['optimization']['scoring']['comfort_weight']])
        writer.writerow([])  # Empty row for separation
        
        # Write results header
        writer.writerow([
            'Letters',
            'Keys',
            'Rank',
            'Total score',
            'Avg bigram component score',
            'Avg key component score',
            'Avg 2-key comfort score (both directions)',
            'Avg 1-key comfort score',
            'Avg bigram frequency score (both directions)',
            'Avg 1-gram frequency score'
        ])
        
        # Write results
        for rank, (score, mapping, detailed_scores) in enumerate(results, 1):
            arrangement = "".join(mapping.keys())
            positions = "".join(mapping.values())
            
            # Calculate component scores
            avg_bigram_score = np.mean([d['bigram_addend'] for d in detailed_scores.values()])
            avg_key_score = np.mean([d['key_addend'] for d in detailed_scores.values()])

            avg_2key_comfort_score1 = np.mean([d['components']['comfort_seq1'] for d in detailed_scores.values()])
            avg_2key_comfort_score2 = np.mean([d['components']['comfort_seq2'] for d in detailed_scores.values()])
            avg_2key_comfort_score = (avg_2key_comfort_score1 + avg_2key_comfort_score2) / 2

            avg_bigram_freq_score1 = np.mean([d['components']['freq_seq1'] for d in detailed_scores.values()])
            avg_bigram_freq_score2 = np.mean([d['components']['freq_seq2'] for d in detailed_scores.values()])
            avg_bigram_freq_score = (avg_bigram_freq_score1 + avg_bigram_freq_score2) / 2

            avg_1key_comfort_score = np.mean([d['components']['avg_comfort'] for d in detailed_scores.values()])
            avg_1gram_freq_score = np.mean([d['components']['avg_freq'] for d in detailed_scores.values()])

            writer.writerow([
                arrangement,
                positions,
                rank,
                f"{score:.4f}",
                f"{avg_bigram_score:.4f}",
                f"{avg_key_score:.4f}",
                f"{avg_2key_comfort_score:.4f}",
                f"{avg_1key_comfort_score:.4f}",
                f"{avg_bigram_freq_score:.4f}",
                f"{avg_1gram_freq_score:.4f}"
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
            layout_chars[key] = '_'

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
# Optimizing functions
#-----------------------------------------------------------------------------
def calculate_layout_score(letter_mapping: Dict[str, str], 
                         norm_bigram_scores: Dict[Tuple[str, str], float],
                         norm_key_scores: Dict[str, float],
                         norm_letter_freqs: Dict[str, float],
                         norm_bigram_freqs: Dict[Tuple[str, str], float],
                         config: dict,
                         right_keys: set,
                         left_keys: set) -> Tuple[float, Dict[str, dict]]:
    """
    Calculate the score for a given layout mapping using the equation:
    
    bigram_addend = weight_bigram_comfort * (norm_comfort_keys_sequence1 + norm_comfort_keys_sequence2) + 
                    weight_bigram_frequency * (norm_frequency_bigram_sequence1 + norm_frequency_bigram_sequence2)
    
    key_addend = weight_key_comfort * avg_norm_comfort_keys + 
                 weight_letter_frequency * norm_frequency_bigram_letters
    
    score = bigram_addend + key_addend

    Args:
        letter_mapping: Dictionary mapping letters to key positions
        norm_bigram_scores: Normalized comfort scores for key pairs
        norm_key_scores: Normalized comfort scores for individual keys
        norm_letter_freqs: Normalized letter frequencies
        norm_bigram_freqs: Normalized bigram frequencies
        config: Configuration dictionary
        right_keys: Set of keys on the right side
        left_keys: Set of keys on the left side
    
    Returns:
        total_score: Final weighted score
        detailed_scores: Dictionary containing component scores for analysis
    """
    weights = config['optimization']['scoring']
    bigram_freq_weight = weights['bigram_frequency_weight']
    letter_freq_weight = weights['letter_frequency_weight']
    comfort_weight = weights['comfort_weight']
    
    total_score = 0
    detailed_scores = {}
    
    # Process each letter pair (bigram)
    for letter1, pos1 in letter_mapping.items():
        for letter2, pos2 in letter_mapping.items():
            if letter1 != letter2:

                # Check if keys are on the same side
                pos1_right = pos1 in right_keys
                pos2_right = pos2 in right_keys
                same_side = (pos1_right and pos2_right) or \
                            (not pos1_right and not pos2_right)
                
                # Calculate bigram_addend
                if same_side:
                    # Get comfort scores for both sequences
                    comfort_seq1 = norm_bigram_scores.get((pos1, pos2), 0)
                    comfort_seq2 = norm_bigram_scores.get((pos2, pos1), 0)
                    
                    # Get frequency scores for both sequences
                    freq_seq1 = norm_bigram_freqs.get((letter1, letter2), 0)
                    freq_seq2 = norm_bigram_freqs.get((letter2, letter1), 0)
                    
                    # Calculate bigram_addend
                    bigram_addend = (comfort_weight * (comfort_seq1 + comfort_seq2) + 
                                    bigram_freq_weight * (freq_seq1 + freq_seq2))
                else:
                    bigram_addend = 0
                
                # Calculate key_addend
                avg_comfort = (norm_key_scores.get(pos1, 0) + norm_key_scores.get(pos2, 0)) / 2
                avg_freq = (norm_letter_freqs.get(letter1, 0) + norm_letter_freqs.get(letter2, 0)) / 2
                
                key_addend = (comfort_weight * avg_comfort + 
                              letter_freq_weight * avg_freq)
                
                score = bigram_addend + key_addend
                total_score += score
                
                detailed_scores[f"{letter1}{letter2}"] = {
                    'total_score': score,
                    'bigram_addend': bigram_addend,
                    'key_addend': key_addend,
                    'components': {
                        'comfort_seq1': comfort_seq1,
                        'comfort_seq2': comfort_seq2,
                        'freq_seq1': freq_seq1,
                        'freq_seq2': freq_seq2,
                        'avg_comfort': avg_comfort,
                        'avg_freq': avg_freq
                    }
                }
    
    return total_score, detailed_scores

def evaluate_layout_permutations(config: dict,
                                 norm_bigram_scores: Dict[Tuple[str, str], float],
                                 norm_key_scores: Dict[str, float],
                                 norm_letter_freqs: Dict[str, float],
                                 norm_bigram_freqs: Dict[Tuple[str, str], float],
                                 right_keys: set,
                                 left_keys: set) -> List[Tuple[float, Dict[str, str], Dict[str, dict]]]:
    """
    Evaluate all possible permutations of letters in the specified key positions.
    
    Args:
        config: Configuration dictionary containing 'letters' and 'keys'
        norm_bigram_scores: Normalized comfort scores for key pairs
        norm_key_scores: Normalized comfort scores for individual keys
        norm_letter_freqs: Normalized letter frequencies
        norm_bigram_freqs: Normalized bigram frequencies
        right_keys: Set of keys on the right side
        left_keys: Set of keys on the left side
    
    Returns:
        List of tuples: (score, letter_mapping, detailed_scores) sorted by score descending
    """
    # Get letters and keys from config
    letters = config['optimization']['letters']
    keys = config['optimization']['keys']
    
    # Validate input
    if len(letters) != len(keys):
        raise ValueError(f"Number of letters ({len(letters)}) must equal number of keys ({len(keys)})")
    
    if len(set(letters)) != len(letters):
        raise ValueError("Duplicate letters found in input letters")
    
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate keys found in input keys")
    
    # Generate all possible permutations of letters
    letter_perms = list(permutations(letters))
    total_perms = len(letter_perms)
    
    print(f"\nEvaluating {total_perms} possible arrangements of '{letters}' on keys '{keys}'")
    
    # Store results as (score, mapping, detailed_scores)
    results = []
    
    # Evaluate each permutation
    for letter_perm in tqdm(letter_perms, desc="Evaluating layouts"):
        # Create letter to key mapping
        letter_mapping = dict(zip(letter_perm, keys))
        
        # Calculate score for this arrangement
        score, detailed_scores = calculate_layout_score(
            letter_mapping=letter_mapping,
            norm_bigram_scores=norm_bigram_scores,
            norm_key_scores=norm_key_scores,
            norm_letter_freqs=norm_letter_freqs,
            norm_bigram_freqs=norm_bigram_freqs,
            config=config,
            right_keys=right_keys,
            left_keys=left_keys
        )
        
        results.append((score, letter_mapping, detailed_scores))
    
    # Sort results by score (descending)
    results.sort(reverse=True, key=lambda x: x[0])
    
    return results

def optimize_layout(config_path: str = "config.yaml") -> None:
    """
    Main function to run the layout optimization process.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Show initial keyboard with positions to fill
    print("\nOptimizing layout for the following positions:")
    visualize_keyboard_layout(None, "Keys to optimize", config)

    # Load and normalize comfort scores
    norm_bigram_scores, norm_key_scores = load_and_normalize_comfort_scores(config)
    
    # Load and normalize frequencies
    norm_letter_freqs, norm_bigram_freqs = load_and_normalize_frequencies(
        onegrams=onegrams,
        onegram_frequencies_array=onegram_frequencies_array,
        bigrams=bigrams,
        bigram_frequencies_array=bigram_frequencies_array
    )
    
    # Get keyboard sides
    right_keys = set()
    left_keys = set()
    for row in ['top', 'home', 'bottom']:
        right_keys.update(config['layout']['positions']['right'][row])
        left_keys.update(config['layout']['positions']['left'][row])
    
    # Run optimization
    try:
        results = evaluate_layout_permutations(
            config=config,
            norm_bigram_scores=norm_bigram_scores,
            norm_key_scores=norm_key_scores,
            norm_letter_freqs=norm_letter_freqs,
            norm_bigram_freqs=norm_bigram_freqs,
            right_keys=right_keys,
            left_keys=left_keys
        )
        
        # Show top layouts with visualization
        print_top_results(results, config)
        
        # Save results to CSV
        save_results_to_csv(results[:config['optimization'].get('nlayouts', 5)], config)

    except ValueError as e:
        print(f"Error: {e}")

#--------------------------------------------------------------------
# Pipeline
#--------------------------------------------------------------------
if __name__ == "__main__":
    try:
        optimize_layout(config_path='config.yaml')
    except Exception as e:
        print(f"Error: {e}")


    

