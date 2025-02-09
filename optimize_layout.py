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
from data.bigram_frequencies_english import (
    bigrams, bigram_frequencies,
    onegrams, onegram_frequencies
)

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_comfort_scores(config: dict = None) -> Dict[Tuple[str, str], float]:
    """Load comfort scores from CSV file."""
    if config is None:
        config = load_config()
    
    scores_file = Path(config['data']['bigram_scores_file'])
    df = pd.read_csv(scores_file)
    comfort_scores = {}
    for _, row in df.iterrows():
        bigram = (row['first_char'], row['second_char'])
        comfort_scores[bigram] = row['comfort_score']
    return comfort_scores

def print_keyboard_layout(positions: Dict[str, str], title: str = "Layout"):
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
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯
"""
    # Create a dictionary for all possible positions, with spaces as defaults
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
    
    # Update with provided positions, converting special characters
    for pos, letter in positions.items():
        layout_key = position_conversion.get(pos, pos)
        layout_chars[layout_key] = letter.upper()
    
    # Print the layout
    print(layout_template.format(title=title, **layout_chars))

def get_available_memory():
    """Get available system memory in bytes."""
    return psutil.virtual_memory().available

def estimate_memory_per_perm(letters: str, positions: str) -> int:
    """Estimate memory needed per permutation in bytes."""
    perm_size = len(positions) * 2  # tuple of position strings
    bigram_size = len(letters) * len(letters) * 20  # bigram scores dict
    return perm_size + bigram_size + 100  # add overhead

def process_chunk(args):
    """Process a chunk of permutations in parallel."""
    perms_chunk, letters, comfort_scores = args
    results = []
    
    # Convert numpy array to list if it's a numpy array
    if isinstance(perms_chunk, np.ndarray):
        perms_chunk = perms_chunk.tolist()
    
    # Right-to-left position mapping for scoring
    right_to_left = {
        'u': 'r', 'i': 'e', 'o': 'w', 'p': 'q',  # Top row
        'j': 'f', 'k': 'd', 'l': 's', ';': 'a',  # Home row
        'm': 'v', ',': 'c', '.': 'x', '/': 'z'   # Bottom row
    }
    
    for pos_perm in perms_chunk:
        # Convert tuple elements to strings if they're numpy string types
        pos_perm = tuple(str(p) for p in pos_perm)
        position_map = dict(zip(letters, pos_perm))
        total_score = 0
        bigram_scores = {}
        
        for l1 in letters:
            for l2 in letters:
                if l1 != l2:
                    bigram = f"{l1}{l2}"
                    freq = bigram_frequencies.get(bigram, 0)
                    
                    pos1, pos2 = position_map[l1], position_map[l2]
                    
                    # Map right-side positions to left-side equivalents
                    pos1 = right_to_left.get(pos1, pos1)
                    pos2 = right_to_left.get(pos2, pos2)
                    
                    comfort = comfort_scores.get((pos1, pos2), float('-inf'))
                    weighted_score = freq * comfort
                    bigram_scores[bigram] = weighted_score
                    total_score += weighted_score
        
        results.append((total_score, pos_perm, bigram_scores))
    
    # Sort and return as list
    return sorted(results, key=lambda x: x[0], reverse=True)[:3]

def optimize_letter_placement(
    letters: str,
    positions: str,
    top_n: int = 3,
    comfort_scores: Optional[Dict[Tuple[str, str], float]] = None,
    n_processes: int = None,
    memory_fraction: float = 0.7
) -> List[Tuple[float, List[str], Dict[str, float]]]:
    """Memory-optimized version of letter placement optimization."""
    if n_processes is None:
        n_processes = cpu_count() - 1

    # Calculate total permutations
    total_perms = factorial(len(positions)) // factorial(len(positions) - len(letters))
    
    # Calculate memory requirements
    available_memory = get_available_memory()
    memory_to_use = int(available_memory * memory_fraction)
    mem_per_perm = estimate_memory_per_perm(letters, positions)
    max_perms_in_memory = memory_to_use // mem_per_perm
    
    # Calculate estimated time
    estimated_time_per_perm = 0.000001  # 1 microsecond per permutation (estimate)
    estimated_total_seconds = total_perms * estimated_time_per_perm / n_processes  # Adjust for parallel processing
    
    print(f"\nOptimizing placement for letters: {letters}")
    print(f"Available positions: {positions}")
    print(f"Will return top {top_n} placements")
    print(f"Total permutations to evaluate: {total_perms:,}")
    print(f"Using {n_processes} processes")
    print(f"Available memory: {available_memory / (1024**3):.1f} GB")
    print(f"Memory per permutation: {mem_per_perm / 1024:.1f} KB")
    print(f"Max permutations in memory: {max_perms_in_memory:,}")
    print(f"Estimated time: {timedelta(seconds=int(estimated_total_seconds))}")
    
    if comfort_scores is None:
        comfort_scores = get_comfort_scores()
    
    start_time = time.time()
    best_placements = []
    perms_iterator = permutations(positions, len(letters))
    batch_size = min(max_perms_in_memory, total_perms)
    num_batches = (total_perms + batch_size - 1) // batch_size
    
    with Pool(n_processes) as pool:
        for batch_num in range(num_batches):
            batch_perms = list(itertools.islice(perms_iterator, batch_size))
            if not batch_perms:
                break
                
            chunks = np.array_split(batch_perms, n_processes)
            chunk_args = [(chunk, letters, comfort_scores) for chunk in chunks]
            
            print(f"\nProcessing batch {batch_num + 1}/{num_batches}")
            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                for chunk_results in pool.imap_unordered(process_chunk, chunk_args):
                    best_placements.extend(chunk_results)
                    best_placements.sort(reverse=True)
                    best_placements = best_placements[:top_n]
                    pbar.update(1)
                    
                    # Update time estimate
                    elapsed = time.time() - start_time
                    progress = (batch_num * len(chunks) + pbar.n) / (num_batches * len(chunks))
                    if progress > 0:
                        remaining = (elapsed / progress) - elapsed
                        pbar.set_postfix({'ETA': str(timedelta(seconds=int(remaining)))})
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {timedelta(seconds=int(elapsed))}")
    
    # Print results
    print("\nTop placements:")
    for i, (score, positions, bigram_scores) in enumerate(best_placements, 1):
        print(f"\nPlacement {i}:")
        layout = dict(zip(positions, letters))
        print_keyboard_layout(layout, f"Score: {score:.4f}")
        
        print("\nTop contributing bigrams:")
        sorted_bigrams = sorted(bigram_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for bigram, score in sorted_bigrams:
            if score != 0:
                print(f"{bigram}: {score:.6f}")
    
    return best_placements


if __name__ == "__main__":

    # total_perms = factorial(len(positions)) // factorial(len(positions) - len(letters))
    # number of cores (sysctl -n hw.ncpu): 14 -- use 14-1 cores 
    # E, T, A, O, I, N, S, R, H, L, D, C / U, M, F, P, G, W, Y, B, V, K, X, J
    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    ncores = multiprocessing.cpu_count() - 1

    config = load_config()
    results = optimize_letter_placement(
        "etaoinsrhldc",
        "qwerasdf;uiopjkl;",
        top_n=3,
        n_processes=ncores,
        memory_fraction=0.7,
        comfort_scores=get_comfort_scores(config)  # Pass comfort scores with config
    )

