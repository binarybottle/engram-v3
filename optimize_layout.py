# optimize_layouts/optimize_layout.py
"""
Memory-efficient item-to-position layout optimization using branch and bound search.

This script uses a branch and bound algorithm to find optimal positions for items 
and item pairs by jointly considering two scoring components: 
item/item_pair scores and position/position_pair scores.

See README for more details.

Usage: python optimize_layout.py
"""
import os
import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Tuple
from math import perm
from tqdm import tqdm
import psutil
import time
from datetime import datetime, timedelta
from numba import jit
import csv
from collections import defaultdict
import pickle

#-----------------------------------------------------------------------------
# Loading, validating, and saving functions
#-----------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file and normalize item case and numeric types."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create necessary output directories
    output_dirs = [config['paths']['output']['layout_results_folder']]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    # Convert weights to float32
    config['optimization']['scoring']['item_weight'] = np.float32(
        config['optimization']['scoring']['item_weight'])
    config['optimization']['scoring']['item_pair_weight'] = np.float32(
        config['optimization']['scoring']['item_pair_weight'])
    
    # Normalize strings to lowercase with consistent access pattern
    optimization = config['optimization']
    for position in ['items_to_assign', 'positions_to_assign', 
                     'items_to_constrain', 'positions_to_constrain',
                     'items_assigned', 'positions_assigned']:
        # Use get() with empty string default for all positions
        optimization[position] = optimization.get(position, '').lower()
    
    # Validate constraints
    validate_config(config)

    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']

    # Convert to lowercase/uppercase for consistency
    items_to_assign = items_to_assign.lower()
    positions_to_assign = positions_to_assign.upper()
    items_to_constrain = items_to_constrain.lower()
    positions_to_constrain = positions_to_constrain.upper()
    items_assigned = items_assigned.lower()
    positions_assigned = positions_assigned.upper()

    print("\nConfiguration:")
    print(f"{len(items_to_assign)} items to assign: {items_to_assign}")
    print(f"{len(positions_to_assign)} available positions: {positions_to_assign}")
    print(f"{len(items_to_constrain)} items to constrain: {items_to_constrain}")
    print(f"{len(positions_to_constrain)} constraining positions: {positions_to_constrain}")
    print(f"{len(items_assigned)} items already assigned: {items_assigned}")
    print(f"{len(positions_assigned)} filled positions: {positions_assigned}")
    print(f"Item weight: {item_weight}")
    print(f"Item-pair weight: {item_pair_weight}")

    return config

def validate_config(config):
    """
    Validate optimization inputs from config, safely handling None values.
    """
    # Safely get and convert values
    items_to_assign = config['optimization'].get('items_to_assign', '')
    items_to_assign = set(items_to_assign.lower() if items_to_assign else '')

    positions_to_assign = config['optimization'].get('positions_to_assign', '')
    positions_to_assign = set(positions_to_assign.upper() if positions_to_assign else '')

    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    items_to_constrain = set(items_to_constrain.lower() if items_to_constrain else '')

    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    positions_to_constrain = set(positions_to_constrain.upper() if positions_to_constrain else '')

    items_assigned = config['optimization'].get('items_assigned', '')
    items_assigned = set(items_assigned.lower() if items_assigned else '')

    positions_assigned = config['optimization'].get('positions_assigned', '')
    positions_assigned = set(positions_assigned.upper() if positions_assigned else '')

    # Check for duplicates
    if len(items_to_assign) != len(config['optimization']['items_to_assign']):
        raise ValueError(f"Duplicate items in items_to_assign: {config['optimization']['items_to_assign']}")
    if len(positions_to_assign) != len(config['optimization']['positions_to_assign']):
        raise ValueError(f"Duplicate positions in positions_to_assign: {config['optimization']['positions_to_assign']}")
    if len(items_assigned) != len(config['optimization']['items_assigned']):
        raise ValueError(f"Duplicate items in items_assigned: {config['optimization']['items_assigned']}")
    if len(positions_assigned) != len(config['optimization']['positions_assigned']):
        raise ValueError(f"Duplicate positions in positions_assigned: {config['optimization']['positions_assigned']}")
    
    # Check that assigned items and positions have matching lengths
    if len(items_assigned) != len(positions_assigned):
        raise ValueError(
            f"Mismatched number of assigned items ({len(items_assigned)}) "
            f"and assigned positions ({len(positions_assigned)})"
        )

    # Check no overlap between assigned and to_assign
    overlap = items_assigned.intersection(items_to_assign)
    if overlap:
        raise ValueError(f"items_to_assign contains assigned items: {overlap}")
    overlap = positions_assigned.intersection(positions_to_assign)
    if overlap:
        raise ValueError(f"positions_to_assign contains assigned positions: {overlap}")

    # Check that we have enough positions
    if len(items_to_assign) > len(positions_to_assign):
        raise ValueError(
            f"More items to assign ({len(items_to_assign)}) "
            f"than available positions ({len(positions_to_assign)})"
        )

    # Check constraints are subsets
    if not items_to_constrain.issubset(items_to_assign):
        invalid = items_to_constrain - items_to_assign
        raise ValueError(f"items_to_constrain contains items not in items_to_assign: {invalid}")
    if not positions_to_constrain.issubset(positions_to_assign):
        invalid = positions_to_constrain - positions_to_assign
        raise ValueError(f"positions_to_constrain contains positions not in positions_to_assign: {invalid}")

    # Check if we have enough constraint positions for constraint items
    if len(items_to_constrain) > len(positions_to_constrain):
        raise ValueError(
            f"Not enough constraint positions ({len(positions_to_constrain)}) "
            f"for constraint items ({len(items_to_constrain)})"
        )
    
def prepare_arrays(
    items_to_assign: str,
    positions_to_assign: str,
    norm_item_scores: Dict[str, float],
    norm_item_pair_scores: Dict[Tuple[str, str], float],
    norm_position_scores: Dict[str, float],
    norm_position_pair_scores: Dict[Tuple[str, str], float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare input arrays and verify they are normalized to [0,1]."""
    n_items_to_assign = len(items_to_assign)
    n_positions_to_assign = len(positions_to_assign)
    
    # Create position score matrix
    position_score_matrix = np.zeros((n_positions_to_assign, n_positions_to_assign), dtype=np.float32)
    for i, k1 in enumerate(positions_to_assign):
        for j, k2 in enumerate(positions_to_assign):
            if i == j:
                position_score_matrix[i, j] = norm_position_scores.get(k1.lower(), 0.0)
            else:
                position_score_matrix[i, j] = norm_position_pair_scores.get((k1.lower(), k2.lower()), 0.0)
    
    # Create item score array
    item_scores = np.array([
        norm_item_scores.get(l.lower(), 0.0) for l in items_to_assign
    ], dtype=np.float32)
    
    # Create item_pair score matrix
    item_pair_score_matrix = np.zeros((n_items_to_assign, n_items_to_assign), dtype=np.float32)
    for i, l1 in enumerate(items_to_assign):
        for j, l2 in enumerate(items_to_assign):
            item_pair_score_matrix[i, j] = norm_item_pair_scores.get((l1.lower(), l2.lower()), 0.0)

    # Verify all scores are normalized [0,1]
    arrays_to_check = [
        (item_scores, "Item scores"),
        (item_pair_score_matrix, "Item pair scores"),
        (position_score_matrix, "Position scores")
    ]
    
    for arr, name in arrays_to_check:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} must be normalized to [0,1] range")
    
    return item_scores, item_pair_score_matrix, position_score_matrix

def detect_and_normalize_distribution(scores: np.ndarray, name: str = '') -> np.ndarray:
    """
    Automatically detect distribution type and apply appropriate normalization.
    Returns scores normalized to [0,1] range.
    """
    # Handle empty or constant arrays
    if len(scores) == 0 or np.all(scores == scores[0]):
        return np.zeros_like(scores)

    # Get basic statistics
    non_zeros = scores[scores != 0]
    if len(non_zeros) == 0:
        return np.zeros_like(scores)
        
    min_nonzero = np.min(non_zeros)
    max_val = np.max(scores)
    mean = np.mean(non_zeros)
    median = np.median(non_zeros)
    skew = np.mean(((non_zeros - mean) / np.std(non_zeros)) ** 3)
    
    # Calculate ratio between consecutive sorted values
    sorted_nonzero = np.sort(non_zeros)
    ratios = sorted_nonzero[1:] / sorted_nonzero[:-1]
    
    # Detect distribution type
    if len(scores[scores == 0]) / len(scores) > 0.3:
        # Sparse distribution with many zeros
        print(f"{name}: Sparse distribution detected")
        adjusted_scores = np.where(scores > 0, scores, min_nonzero / 10)
        log_scores = np.log10(adjusted_scores)
        return (log_scores - np.min(log_scores)) / (np.max(log_scores) - np.min(log_scores))
    
    elif skew > 2 or np.median(ratios) > 1.5:
        # Heavy-tailed/exponential/zipfian distribution
        print(f"{name}: Heavy-tailed distribution detected")
        log_scores = np.log10(scores + min_nonzero/10)
        return (log_scores - np.min(log_scores)) / (np.max(log_scores) - np.min(log_scores))
        
    elif abs(mean - median) / mean < 0.1:
        # Roughly symmetric distribution
        print(f"{name}: Symmetric distribution detected")
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
    else:
        # Default to robust scaling
        print(f"{name}: Using robust scaling")
        q1, q99 = np.percentile(scores, [1, 99])
        scaled = (scores - q1) / (q99 - q1)
        return np.clip(scaled, 0, 1)

def load_and_normalize_scores(config: dict):
    """Load and normalize scores."""
    # Load raw data
    item_df = pd.read_csv(config['paths']['input']['item_scores_file'])
    item_pair_df = pd.read_csv(config['paths']['input']['item_pair_scores_file'])
    position_df = pd.read_csv(config['paths']['input']['position_scores_file'])
    position_pair_df = pd.read_csv(config['paths']['input']['position_pair_scores_file'])
    
    #-------------------------------------------------------------------------
    # Create normalized item scores dictionary 
    #-------------------------------------------------------------------------
    norm_item_scores = {}
    scores = item_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Item scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    for idx, row in item_df.iterrows():
        norm_item_scores[row['item'].lower()] = np.float32(norm_scores[idx])
        
    #-------------------------------------------------------------------------
    # Create normalized item-pair scores dictionary 
    #-------------------------------------------------------------------------
    norm_item_pair_scores = {}
    scores = item_pair_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Item pair scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    for idx, row in item_pair_df.iterrows():
        item_pair = row['item_pair']
        if not isinstance(item_pair, str):
            print(f"Warning: non-string item_pair at index {idx}: {item_pair} of type {type(item_pair)}")
            continue
        chars = tuple(item_pair.lower())
        norm_item_pair_scores[chars] = np.float32(norm_scores[idx])

    #-------------------------------------------------------------------------
    # Create normalized position scores dictionary
    #-------------------------------------------------------------------------
    norm_position_scores = {}
    scores = position_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Position scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    
    for idx, row in position_df.iterrows():
        norm_position_scores[row['position'].lower()] = np.float32(norm_scores[idx])
        
    #-------------------------------------------------------------------------
    # Create normalized position-pair scores dictionary
    #-------------------------------------------------------------------------
    norm_position_pair_scores = {}
    scores = position_pair_df['score'].values
    norm_scores = detect_and_normalize_distribution(scores, 'Position pair scores')
    print("  - original:", min(scores), "to", max(scores))
    print("  - normalized:", min(norm_scores), "to", max(norm_scores))
    
    for idx, row in position_pair_df.iterrows():
        chars = tuple(c.lower() for c in row['position_pair'])
        norm_position_pair_scores[chars] = np.float32(norm_scores[idx])

    #print("\nScore ranges after normalization:")
    #print(f"Item scores: [{min(norm_item_scores.values()):.4f}, {max(norm_item_scores.values()):.4f}]")
    #print(f"Item pair scores: [{min(norm_item_pair_scores.values()):.4f}, {max(norm_item_pair_scores.values()):.4f}]")
    #print(f"Position scores: [{min(norm_position_scores.values()):.4f}, {max(norm_position_scores.values()):.4f}]")
    #print(f"Position pair scores: [{min(norm_position_pair_scores.values()):.4f}, {max(norm_position_pair_scores.values()):.4f}]")

    return norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores
   
def validate_mapping(mapping: np.ndarray, constrained_item_indices: set, constrained_positions: set) -> bool:
    """Validate that mapping follows all constraints."""
    for idx in constrained_item_indices:
        if mapping[idx] >= 0 and mapping[idx] not in constrained_positions:
            return False
    return True

def save_results_to_csv(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]], 
                        config: dict,
                        output_path: str = "layout_results.csv") -> None:
    """
    Save layout results to a CSV file with proper escaping of special characters.
    """
    # Generate timestamp and set output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config['paths']['output']['layout_results_folder'], 
                               f"layout_results_{timestamp}.csv")
    
    def escape_special_chars(text: str) -> str:
        """Helper function to escape special characters and ensure proper CSV formatting."""
        # Replace certain special characters with their character names to prevent CSV splitting
        replacements = {
            ';': '[semicolon]',
            ',': '[comma]',
            '.': '[period]',
            '/': '[slash]'
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields
        
        # Write header with configuration info
        opt = config['optimization']
        writer.writerow(['Items to assign', escape_special_chars(opt.get('items_to_assign', ''))])
        writer.writerow(['Available positions', escape_special_chars(opt.get('positions_to_assign', ''))])
        writer.writerow(['Items to constrain', escape_special_chars(opt.get('items_to_constrain', ''))])
        writer.writerow(['Constraint positions', escape_special_chars(opt.get('positions_to_constrain', ''))])
        writer.writerow(['Assigned items', escape_special_chars(opt.get('items_assigned', ''))])
        writer.writerow(['Assigned positions', escape_special_chars(opt.get('positions_assigned', ''))])
        writer.writerow(['Item-pair weight', opt['scoring']['item_pair_weight']])
        writer.writerow(['Item weight', opt['scoring']['item_weight']])
        writer.writerow([])  # Empty row for separation
        
        # Write results header
        writer.writerow([
            'Items',
            'Positions',
            'Rank',
            'Total score',
            'Item-pair score (unweighted)',
            'Item score (unweighted)'
        ])
        
        # Write results
        for rank, (score, mapping, detailed_scores) in enumerate(results, 1):
            arrangement = escape_special_chars("".join(mapping.keys()))
            positions = escape_special_chars("".join(mapping.values()))
            
            # Get first item_pair score and item score
            first_entry = next(iter(detailed_scores.values()))
            unweighted_item_pair_score = first_entry['unweighted_item_pair_score']
            unweighted_item_score = first_entry['unweighted_item_score']

            writer.writerow([
                arrangement,
                positions,    
                rank,
                f"{score:.4f}",
                f"{unweighted_item_pair_score:.4f}",
                f"{unweighted_item_score:.4f}"
            ])
    
    print(f"\nResults saved to: {output_path}")

#-----------------------------------------------------------------------------
# Visualizing functions
#-----------------------------------------------------------------------------
def visualize_keyboard_layout(mapping: Dict[str, str] = None, title: str = "Layout", config: dict = None, items_to_display: str = None, positions_to_display: str = None) -> None:
    """
    Print a visual representation of a keyboard layout showing assigned items.
    """
    # Templates
    KEYBOARD_TEMPLATE = """╭───────────────────────────────────────────────╮
│ Layout: {title:<34}    │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│ {q:^3} │ {w:^3} │ {e:^3} │ {r:^3} ║ {u:^3} │ {i:^3} │ {o:^3} │ {p:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│ {a:^3} │ {s:^3} │ {d:^3} │ {f:^3} ║ {j:^3} │ {k:^3} │ {l:^3} │ {sc:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│ {z:^3} │ {x:^3} │ {c:^3} │ {v:^3} ║ {m:^3} │ {cm:^3} │ {dt:^3} │ {sl:^3} │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯"""

    QWERTY_KEYBOARD_TEMPLATE = """╭───────────────────────────────────────────────╮
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

    # Position mapping for special characters
    position_mapping = {
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
    
    # First apply assigned items from config
    items_assigned = config['optimization'].get('items_assigned', '').lower()
    positions_assigned = config['optimization'].get('positions_assigned', '').lower()
    
    if items_assigned and positions_assigned:
        for item, position in zip(items_assigned, positions_assigned):
            converted_position = position_mapping.get(position.lower(), position.lower())
            layout_chars[converted_position] = item.lower()
    
    # Then mark positions to be filled from positions_to_assign
    if not mapping:
        positions_to_mark = config['optimization'].get('positions_to_assign', '').lower()
        for position in positions_to_mark:
            if position not in positions_assigned:  # Skip if already assigned
                converted_position = position_mapping.get(position, position)
                layout_chars[converted_position] = '-'
    
    # Fill in items from items_to_display and positions_to_display
    if items_to_display and positions_to_display:
        if len(items_to_display) != len(positions_to_display):
            raise ValueError("items_to_display and positions_to_display must have the same length")
            
        for item, position in zip(items_to_display, positions_to_display):
            converted_position = position_mapping.get(position.lower(), position.lower())
            layout_chars[converted_position] = item.lower()
    
    # Fill in items from mapping
    if mapping:
        for item, position in mapping.items():
            converted_position = position_mapping.get(position.lower(), position.lower())
            current_char = layout_chars[converted_position]
            
            # If there's already an assigned item, combine them
            if current_char.strip() and current_char != '_':
                # Put mapped item (uppercase) first, then assigned item (lowercase)
                layout_chars[converted_position] = f"{item.upper()}{current_char}"
            else:
                # Just show the mapped item in uppercase
                layout_chars[converted_position] = item.upper()

    # Use template from config if specified, otherwise use default
    template = KEYBOARD_TEMPLATE  # QWERTY_KEYBOARD_TEMPLATE
    print(template.format(**layout_chars))

def calculate_total_perms(
    n_items: int,
    n_positions: int,
    items_to_constrain: set,
    positions_to_constrain: set,
    items_assigned: set,
    positions_assigned: set
) -> dict:
    """
    Calculate exact number of permutations for two-phase search.
    
    Phase 1: Arrange constrained items within constrained positions.
    Phase 2: For each Phase 1 solution, arrange remaining items in remaining positions.
    """
    # Phase 1: Arrange constrained items in constrained positions
    n_constrained_items = len(items_to_constrain)
    n_constrained_positions = len(positions_to_constrain)
    
    if n_constrained_items == 0 or n_constrained_positions == 0:
        # No constraints - everything happens in Phase 2
        total_perms_phase1 = 1
        remaining_items = n_items - len(items_assigned)
        remaining_positions = n_positions - len(positions_assigned)
    else:
        # Calculate Phase 1 permutations
        total_perms_phase1 = perm(n_constrained_positions, n_constrained_items)
        # After Phase 1, we have used n_constrained_items positions
        remaining_items = n_items - n_constrained_items - len(items_assigned)
        remaining_positions = n_positions - n_constrained_items - len(positions_assigned)
    
    # Phase 2: For each Phase 1 solution, arrange remaining items
    perms_per_phase1 = perm(remaining_positions, remaining_items)
    total_perms_phase2 = perms_per_phase1 * total_perms_phase1
    
    return {
        'total_perms': total_perms_phase2,  # Total perms is just Phase 2 total
        'phase1_arrangements': total_perms_phase1,
        'phase2_arrangements': total_perms_phase2,
        'details': {
            'remaining_positions': remaining_positions,
            'remaining_items': remaining_items,
            'constrained_items': n_constrained_items,
            'constrained_positions': n_constrained_positions,
            'arrangements_per_phase1': perms_per_phase1
        }
    }

def update_progress_bar(pbar, processed_perms: int, start_time: float, total_perms: int) -> None:
    """Update progress bar with accurate statistics."""
    current_time = time.time()
    elapsed = current_time - start_time   
    if elapsed > 0:
        perms_per_second = processed_perms / elapsed
        percent_explored = (processed_perms / total_perms) * 100 if total_perms > 0 else 0
        remaining_perms = total_perms - processed_perms
        eta_minutes = (remaining_perms / perms_per_second) / 60 if perms_per_second > 0 else 0
        
        pbar.set_postfix({
            'Perms/sec': f"{perms_per_second:,.0f}",
            'Explored': f"{percent_explored:.1f}%",
            'ETA': 'Complete' if processed_perms >= total_perms else f"{eta_minutes:.1f}m",
            'Memory': f"{psutil.Process().memory_info().rss/1e9:.1f}GB"
        })

def print_top_results(results: List[Tuple[float, Dict[str, str], Dict[str, dict]]], 
                      config: dict,
                      n: int = None,
                      items_to_display: str = None,
                      positions_to_display: str = None,
                      print_keyboard: bool = True) -> None:
    """
    Print the top N results with their scores and mappings.
    
    Args:
        results: List of (score, mapping, detailed_scores) tuples
        config: Configuration dictionary
        n: Number of layouts to display (defaults to config['optimization']['nlayouts'])
        items_to_display: items that have already been assigned
        positions_to_display: positions that have already been assigned
        print_keyboard: Whether to print the keyboard layout
    """
    if n is None:
        n = config['optimization'].get('nlayouts', 5)
    
    print(f"\nTop {n} scoring layouts:")
    for i, (score, mapping, detailed_scores) in enumerate(results[:n], 1):
        print(f"\n#{i}: Score: {score:.4f}")

        if print_keyboard:
            visualize_keyboard_layout(
                mapping=mapping,
                title=f"Layout #{i}", 
                items_to_display=items_to_display,
                positions_to_display=positions_to_display,
                config=config
            )
        
#-----------------------------------------------------------------------------
# Optimizing functions
#-----------------------------------------------------------------------------
@jit(nopython=True, fastmath=True)
def calculate_layout_score(
    item_indices: np.ndarray,
    position_score_matrix: np.ndarray,  
    item_scores: np.ndarray,    
    item_pair_score_matrix: np.ndarray,    
    item_weight: float,
    item_pair_weight: float
) -> tuple:
    """Calculate layout score for normalized [0,1] scores."""
    if not np.any(item_indices >= 0):
        return -1.0, 0.0, 0.0
        
    # Single-item component - no special normalization needed since scores are [0,1]
    item_component = np.float32(0.0)
    valid_count = 0
    for i in range(len(item_indices)):
        pos = item_indices[i]
        if pos >= 0:
            item_component += position_score_matrix[pos, pos] * item_scores[i]
            valid_count += 1
    
    item_component = item_component / valid_count if valid_count > 0 else 0.0
    
    # Item-pair component
    item_pair_component = np.float32(0.0)
    pair_count = 0
    for i in range(len(item_indices)):
        pos1 = item_indices[i]
        if pos1 >= 0:
            for j in range(i + 1, len(item_indices)):
                pos2 = item_indices[j]
                if pos2 >= 0:
                    # Simple sum of products since all scores are [0,1]
                    score = (position_score_matrix[pos1, pos2] * item_pair_score_matrix[i, j] +
                            position_score_matrix[pos2, pos1] * item_pair_score_matrix[j, i])
                    item_pair_component += score
                    pair_count += 2

    item_pair_component = item_pair_component / pair_count if pair_count > 0 else 0.0
    
    return (float(item_weight * item_component + item_pair_weight * item_pair_component),
            float(item_component),
            float(item_pair_component))
            
@jit(nopython=True, fastmath=True)
def calculate_upper_bound(
    mapping: np.ndarray,
    depth: int,
    used: np.ndarray,
    position_score_matrix: np.ndarray,
    item_scores: np.ndarray,
    item_pair_score_matrix: np.ndarray,
    item_weight: float,
    item_pair_weight: float
) -> float:
    """Calculate accurate upper bound on best possible score from this node."""

    # Get current score from assigned items
    current_score, current_unweighted_item_score, current_unweighted_item_pair_score = calculate_layout_score(
        mapping, position_score_matrix, item_scores, item_pair_score_matrix, item_weight, item_pair_weight
    )
    
    # Return the current score if there are no remaining items
    n_remaining_items = len(mapping) - depth
    if n_remaining_items == 0:
        return current_score
        
    # Get unassigned items and available positions
    # These include item indices that have -1 in the mapping array, and 
    # include both constrained and unconstrained items that haven't been assigned
    assigned = np.where(mapping >= 0)[0]  # Already placed items
    unassigned = np.where(mapping < 0)[0]
    available_positions = np.where(~used)[0]
    
    #-------------------------------------------------------------------------
    # Item scoring component
    #-------------------------------------------------------------------------
    # Get single-item position scores (from diagonal)
    # numba doesn't accept np.diagonal
    #position_single_scores = np.diagonal(position_score_matrix)[available_positions]
    position_single_scores = np.zeros(len(available_positions))
    for i, pos in enumerate(available_positions):
        position_single_scores[i] = position_score_matrix[pos, pos]
    position_single_scores = np.sort(position_single_scores)[-n_remaining_items:][::-1]  # highest to lowest

    # Get scores for unassigned items
    remaining_scores = item_scores[unassigned]
    remaining_scores = np.sort(remaining_scores)[::-1]  # highest to lowest
    
    # Maximum possible item score contribution (unweighted)
    max_item_component = np.sum(position_single_scores * remaining_scores) / len(mapping)
    
    #-------------------------------------------------------------------------
    # Item-pair scoring component
    #-------------------------------------------------------------------------
    # Extract and sort position pair scores
    n_available = len(available_positions)
    position_pair_scores = np.zeros(n_available * (n_available - 1) // 2, dtype=np.float32)
    idx = 0
    for i in range(n_available):
        pos1 = available_positions[i]
        for j in range(i + 1, n_available):
            pos2 = available_positions[j]
            # Take maximum of both directions
            score = max(position_score_matrix[pos1, pos2],
                        position_score_matrix[pos2, pos1])
            position_pair_scores[idx] = score
            idx += 1
    position_pair_scores = np.sort(position_pair_scores)[::-1]

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 1. Unassigned-to-unassigned item_pairs
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    n_unassigned = len(unassigned)
    item_pair_scores = np.zeros(n_unassigned * (n_unassigned - 1) // 2, dtype=np.float32)
    idx = 0
    for i in range(n_unassigned):
        for j in range(i + 1, n_unassigned):
            l1, l2 = unassigned[i], unassigned[j]
            # Take maximum of both directions
            score = max(item_pair_score_matrix[l1, l2],
                        item_pair_score_matrix[l2, l1])
            item_pair_scores[idx] = score
            idx += 1
    item_pair_scores = np.sort(item_pair_scores)[::-1]
    
    # Match best pairs
    # The unassigned pairs loop only does the upper triangle (i,j) not (j,i), 
    # so each pair is counted once in idx, and the score is normalized by n_pairs:
    n_pairs = min(len(position_pair_scores), len(item_pair_scores))
    if n_pairs > 0:
        unassigned_score = np.sum(position_pair_scores[:n_pairs] * item_pair_scores[:n_pairs]) / n_pairs
    else:
        unassigned_score = 0.0

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 2. Assigned-to-unassigned item_pairs
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Assigned-to-unassigned scores (normalized)
    assigned_unassigned_score = 0.0
    n_assigned_unassigned_pairs = 0
    for i in assigned:
        pos_i = mapping[i]
        for j in unassigned:
            score = max(item_pair_score_matrix[i, j],
                        item_pair_score_matrix[j, i])
            
            best_position = 0.0
            for pos_j in available_positions:
                position = max(position_score_matrix[pos_i, pos_j],
                               position_score_matrix[pos_j, pos_i])
                best_position = max(best_position, position)
            
            # The assigned-unassigned loop does all combinations, 
            # multiplies the score by 2 for both directions,
            # and explicitly counts each direction by incrementing by 2:
            assigned_unassigned_score += 2 * (best_position * score)
            n_assigned_unassigned_pairs += 2  # Count both directions

    # Normalize by the number of pairs
    if n_assigned_unassigned_pairs > 0:
        assigned_unassigned_score /= n_assigned_unassigned_pairs

    max_item_pair_component = unassigned_score + assigned_unassigned_score
        
    #-------------------------------------------------------------------------
    # Weight and combine components
    #-------------------------------------------------------------------------
    total_item_score = (current_unweighted_item_score + max_item_component) * item_weight
    total_item_pair_score = (current_unweighted_item_pair_score + 
                             max_item_pair_component) * item_pair_weight

    return float(total_item_score + total_item_pair_score)

def get_next_unassigned_item(mapping: np.ndarray) -> int:
    """Find the index of the next unassigned item."""
    for i in range(len(mapping)):
        if mapping[i] < 0:
            return i
    return None

def branch_and_bound_optimal(
    arrays: tuple,
    weights: tuple,
    config: dict,
    n_solutions: int = 5
) -> List[Tuple[float, Dict[str, str], Dict]]:
    """
    Branch and bound implementation using depth-first search. 
    
    Uses DFS instead of best-first search because:
    1. With a mathematically sound upper bound for pruning, the search order 
       doesn't affect optimality
    2. DFS requires only O(depth) memory vs O(width^depth) for best-first
    3. Simpler implementation without heap management complexity
    
    Search is conducted in two phases:
    Phase 1:
      - Finds all valid arrangements of constrained items (e.g., 'e', 't') 
        in constrained positions (e.g., F, D, J, K).
      - Each arrangement marks positions as used/assigned.

    Phase 2: For each Phase 1 solution, arrange remaining items
      - For each valid arrangement from Phase 1
        - Uses ONLY the positions that weren't assigned during Phase 1.
        - In other words, if a Phase 1 solution put 'e' in F and 't' in J, 
          then Phase 2 would use remaining positions (not F or J)
          to arrange the remaining items.

    Args:
        arrays: Tuple of (item_scores, item_pair_score_matrix, position_score_matrix)
        weights: Tuple of (item_weight, item_pair_weight)
        config: Configuration dictionary with optimization parameters
        n_solutions: Number of top solutions to maintain

    Returns:
        Tuple containing:
        - List of (score, mapping, detailed_scores) tuples
        - Total number of permutations processed
    """
    # Get items and positions from config
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')

    # Initialize dimensions and arrays
    n_items_to_assign = len(items_to_assign)
    n_positions_to_assign = len(positions_to_assign)
    
    item_scores, item_pair_score_matrix, position_score_matrix = arrays
    item_weight, item_pair_weight = weights
    
    # Set up constraint tracking
    constrained_items = set(items_to_constrain.lower())
    constrained_positions = set(i for i, position in enumerate(positions_to_assign) 
                              if position.upper() in positions_to_constrain.upper())
    constrained_item_indices = set(i for i, item in enumerate(items_to_assign) 
                                   if item in constrained_items)
    
    # Initialize search structures
    solutions = []  # Will store (score, unweighted_scores, mapping) tuples
    worst_top_n_score = float('-inf')
    
    # Initialize mapping and used positions
    initial_mapping = np.full(n_items_to_assign, -1, dtype=np.int32)
    initial_used = np.zeros(n_positions_to_assign, dtype=bool)

    # Track statistics
    processed_perms = 0
    pruning_stats = {
        'depth': defaultdict(int),  # Count pruned branches by depth
        'margin': [],               # Track pruning margins (diff between upper bound and worst score)
        'total_pruned': 0,
        'total_explored': 0
    }

    # Handle pre-assigned items
    if items_assigned:
        item_to_idx = {item: idx for idx, item in enumerate(items_to_assign)}
        position_to_pos = {position: pos for pos, position in enumerate(positions_to_assign)}
        
        for item, position in zip(items_assigned, positions_assigned):
            if item in item_to_idx and position in position_to_pos:
                idx = item_to_idx[item]
                pos = position_to_pos[position]
                initial_mapping[idx] = pos
                initial_used[pos] = True
                print(f"Pre-assigned: {item} -> {position} (position {pos})")
    
    # Calculate phase perms
    total_perms_phase1 = perm(len(constrained_positions), len(constrained_items))
    phase2_remaining_items = n_items_to_assign - len(constrained_items) - len(items_assigned)
    phase2_available_positions = n_positions_to_assign - len(constrained_items) - len(items_assigned)
    perms_per_phase1_solution = perm(phase2_available_positions, phase2_remaining_items)
    total_perms_phase2 = perms_per_phase1_solution * total_perms_phase1

    print(f"\nPhase 1 (Constrained items): {total_perms_phase1:,} permutations")
    print(f"Phase 2 (Remaining items): {total_perms_phase2:,} permutations")
    
    # Track progress
    start_time = time.time()

    def phase1_dfs(mapping: np.ndarray, used: np.ndarray, depth: int, pbar: tqdm) -> List[Tuple[np.ndarray, np.ndarray]]:
        """DFS for Phase 1 (constrained items)."""
        solutions = []
        
        # Found a valid phase 1 arrangement
        if all(mapping[i] >= 0 for i in constrained_item_indices):
            solutions.append((mapping.copy(), used.copy()))
            pbar.update(1)
            return solutions
            
        # Try next constrained item
        current_item_idx = -1
        for i in constrained_item_indices:
            if mapping[i] < 0:
                current_item_idx = i
                break
                
        # Try each constrained position
        for pos in constrained_positions:
            if not used[pos]:
                new_mapping = mapping.copy()
                new_mapping[current_item_idx] = pos
                new_used = used.copy()
                new_used[pos] = True
                solutions.extend(phase1_dfs(new_mapping, new_used, depth + 1, pbar))
        
        return solutions

    def phase2_dfs(
        mapping: np.ndarray, 
        used: np.ndarray, 
        depth: int,
        pbar: tqdm
    ) -> None:
        """DFS for Phase 2 (remaining items)."""
        nonlocal solutions, worst_top_n_score, processed_perms
        
        # Process complete solutions
        if depth == n_items_to_assign:
            if not validate_mapping(mapping, constrained_item_indices, constrained_positions):
                return

            processed_perms += 1
            pbar.update(1)
            
            score_tuple = calculate_layout_score(
                mapping,
                position_score_matrix,
                item_scores,
                item_pair_score_matrix,
                item_weight,
                item_pair_weight
            )
            total_score, unweighted_item_score, unweighted_item_pair_score = score_tuple

            if len(solutions) < n_solutions or total_score > worst_top_n_score:
                solution = (
                    total_score,
                    unweighted_item_score,
                    unweighted_item_pair_score,
                    mapping.tolist()
                )
                solutions.append(solution)
                solutions.sort(key=lambda x: x[0])  # Sort by total_score
                if len(solutions) > n_solutions:
                    solutions.pop(0)  # Remove worst solution
                worst_top_n_score = solutions[0][0]
            return

        # Find next unassigned item
        current_item_idx = get_next_unassigned_item(mapping)
        if current_item_idx is None:
            return

        # Get valid positions for this item
        if current_item_idx in constrained_item_indices:
            valid_positions = [pos for pos in constrained_positions if not used[pos]]
        else:
            valid_positions = [pos for pos in range(n_positions_to_assign) if not used[pos]]
            
        # Try each valid position
        for pos in valid_positions:
            new_mapping = mapping.copy()
            new_mapping[current_item_idx] = pos
            new_used = used.copy()
            new_used[pos] = True
            
            # Only prune if we have n solutions to compare against
            if len(solutions) >= n_solutions:
                upper_bound = calculate_upper_bound(
                    new_mapping, depth + 1, new_used,
                    position_score_matrix, item_scores,
                    item_pair_score_matrix, item_weight, item_pair_weight
                )
                
                margin = upper_bound - (worst_top_n_score - np.abs(worst_top_n_score) * np.finfo(np.float32).eps)
                if margin < 0:  # We can safely prune
                    pruning_stats['depth'][depth] += 1
                    pruning_stats['margin'].append(margin)
                    pruning_stats['total_pruned'] += 1
                    continue
                pruning_stats['total_explored'] += 1

            # Recurse
            phase2_dfs(new_mapping, new_used, depth + 1, pbar)

    #-------------------------------------------------------------------------
    # Phase 1: Find all valid arrangements of constrained items
    #-------------------------------------------------------------------------
    phase1_solutions = []
    with tqdm(total=total_perms_phase1, desc="Phase 1", unit='perms') as pbar:
        phase1_solutions = phase1_dfs(initial_mapping, initial_used, 0, pbar)
    
    print(f"\nFound {len(phase1_solutions)} valid phase 1 arrangements")
    
    #-------------------------------------------------------------------------
    # Phase 2: For each Phase 1 solution, arrange remaining items
    #-------------------------------------------------------------------------
    current_phase1_solution_index = 0

    with tqdm(total=total_perms_phase2, desc="Phase 2", unit='perms') as pbar:
        for phase1_mapping, phase1_used in phase1_solutions:
            print(f"\nProcessing Phase 1 solution {current_phase1_solution_index + 1}/{len(phase1_solutions)}")
            
            # Calculate initial depth based on assigned items
            initial_depth = sum(1 for i in range(n_items_to_assign) if phase1_mapping[i] >= 0)
            
            # Start DFS from this phase 1 solution
            phase2_dfs(phase1_mapping, phase1_used, initial_depth, pbar)
            
            current_phase1_solution_index += 1

    # Convert final solutions to return format
    return_solutions = []
    for score, unweighted_item_score, unweighted_item_pair_score, mapping_list in reversed(solutions):
        mapping = np.array(mapping_list, dtype=np.int32)
        item_mapping = dict(zip(items_to_assign, [positions_to_assign[i] for i in mapping]))
        return_solutions.append((
            score,
            item_mapping,
            {'total': {
                'total_score': score,
                'unweighted_item_pair_score': unweighted_item_pair_score,
                'unweighted_item_score': unweighted_item_score
            }}
        ))

    # Print pruning statistics
    print("\nPruning statistics:")
    print(f"Total nodes explored: {pruning_stats['total_explored']:,}")
    print(f"Total branches pruned: {pruning_stats['total_pruned']:,}")
    print("Pruning by depth:")
    for depth in sorted(pruning_stats['depth'].keys()):
        count = pruning_stats['depth'][depth]
        print(f"  Depth {depth}: {count:,} branches pruned")
    if pruning_stats['margin']:  # Only if we have margins to report
        margins = np.array(pruning_stats['margin'])
        print("Pruning margins:")
        print(f"  Min: {np.min(margins):.6f}")
        print(f"  Max: {np.max(margins):.6f}")
        print(f"  Mean: {np.mean(margins):.6f}")

    return return_solutions, processed_perms

def optimize_layout(config: dict) -> None:
    """
    Main optimization function.
    """
    # Get parameters from config
    items_to_assign = config['optimization']['items_to_assign']
    positions_to_assign = config['optimization']['positions_to_assign']
    items_to_constrain = config['optimization'].get('items_to_constrain', '')
    positions_to_constrain = config['optimization'].get('positions_to_constrain', '')
    items_assigned = config['optimization'].get('items_assigned', '')
    positions_assigned = config['optimization'].get('positions_assigned', '')
    print_keyboard = config['visualization']['print_keyboard']

    # Convert to lowercase/uppercase for consistency
    items_to_assign = items_to_assign.lower()
    positions_to_assign = positions_to_assign.upper()
    items_to_constrain = items_to_constrain.lower()
    positions_to_constrain = positions_to_constrain.upper()
    items_assigned = items_assigned.lower()
    positions_assigned = positions_assigned.upper()

    # Validate configuration
    validate_config(config)
    
    # Calculate exact search space size
    search_space = calculate_total_perms(
        n_items=len(items_to_assign),
        n_positions=len(positions_to_assign),
        items_to_constrain=set(items_to_constrain),
        positions_to_constrain=set(positions_to_constrain),
        items_assigned=set(items_assigned),
        positions_assigned=set(positions_assigned)
    )
    
    # Print detailed search space analysis
    print("\nSearch space:")
    if search_space['phase1_arrangements'] > 1:
        print(f"Phase 1 ({search_space['details']['constrained_items']} items constrained to {search_space['details']['constrained_positions']} positions): {search_space['phase1_arrangements']:,} permutations")
        print(f"Phase 2 ({search_space['details']['remaining_items']} items to arrange in {search_space['details']['remaining_positions']} positions): {search_space['details']['arrangements_per_phase1']:,} permutations per Phase 1 solution")
        print(f"Total permutations: {search_space['total_perms']:,}")
    else:
        print("No constraints - running single-phase optimization")
        print(f"Total permutations: {search_space['total_perms']:,}")
        print(f"- Arranging {search_space['details']['remaining_items']} items in {search_space['details']['remaining_positions']} positions")

    # Show initial positionboard
    if print_keyboard:
        print("\n")
        visualize_keyboard_layout(
            mapping=None,
            title="positions to optimize",
            items_to_display=items_assigned,
            positions_to_display=positions_assigned,
            config=config
        )
        
    # Load and normalize scores
    print("\nNormalization of scores:")
    norm_item_scores, norm_item_pair_scores, norm_position_scores, norm_position_pair_scores = load_and_normalize_scores(config)
    
    # Get scoring weights
    item_weight = config['optimization']['scoring']['item_weight']
    item_pair_weight = config['optimization']['scoring']['item_pair_weight']
        
    # Prepare arrays and run optimization
    arrays = prepare_arrays(
        items_to_assign, positions_to_assign,
        norm_item_scores, norm_item_pair_scores, 
        norm_position_scores, norm_position_pair_scores
    )
    
    # Get number of layouts from config
    n_layouts = config['optimization'].get('nlayouts', 5)
    
    print("\nStarting Phase 1 optimization to find top {n_layouts} solutions:")
    print(f"  - {len(items_to_constrain)} constrained items: {items_to_constrain}")
    print(f"  - {len(positions_to_constrain)} constrained positions: {positions_to_constrain}")
    print(f"  - Finding top {n_layouts} solutions")

    # Run optimization
    weights = (item_weight, item_pair_weight)
    results, processed_perms = branch_and_bound_optimal(
        arrays=arrays,
        weights=weights,
        config=config,
        n_solutions=n_layouts
    )
    
    # Sort and save results
    sorted_results = sorted(
        results,
        key=lambda x: (
            x[0],                                         # use total_score as primary sort
            x[2]['total']['unweighted_item_score'],       # use item score as secondary sort
            x[2]['total']['unweighted_item_pair_score']   # use item_pair score as tertiary sort
        ),
        reverse=True
    )
    
    print_top_results(
        results=sorted_results,
        config=config,
        n=None,
        items_to_display=items_assigned,
        positions_to_display=positions_assigned,
        print_keyboard=print_keyboard
    )
    
    save_results_to_csv(sorted_results, config)

    # Final statistics reporting
    elapsed_time = time.time() - start_time
    if processed_perms >= search_space['total_perms']:
        print(f"Total permutations processed: {processed_perms:,} (100% of solution space explored) in {timedelta(seconds=int(elapsed_time))}")
    else:
        percent_explored = (processed_perms / search_space['total_perms']) * 100
        print(f"Total permutations processed: {processed_perms:,} ({percent_explored:.1f}% of solution space explored) in {timedelta(seconds=int(elapsed_time))}")

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
        print(f"Total runtime: {timedelta(seconds=int(elapsed))}")

    except Exception as e:
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()


    

