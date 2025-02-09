from itertools import permutations
import heapq
from typing import List, Dict, Tuple, Optional
import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path
from input.bigram_frequencies_english import (
    bigrams, bigram_frequencies,
    onegrams, onegram_frequencies
)

class Config:
    """Configuration class for layout optimization."""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set up paths
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on settings."""
        log_config = self.config['logging']
        log_file = Path(log_config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

class LayoutOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.comfort_scores = self._load_comfort_scores()
        
        # Load layout settings
        layout_config = config.config['layout']
        self.right_positions = layout_config['right_positions']
        self.left_positions = layout_config['left_positions']
        self.position_mappings = layout_config['position_mappings']['right_to_left']

    def _load_comfort_scores(self) -> Dict[Tuple[str, str], float]:
        """Load comfort scores from CSV file."""
        scores_file = Path(self.config.config['data']['bigram_scores_file'])
        df = pd.read_csv(scores_file)
        return {(row['first_char'], row['second_char']): row['comfort_score'] 
                for _, row in df.iterrows()}

    def get_top_consonants(self, n: int) -> str:
        """Get the N most frequent consonants."""
        consonants = [c for c in 'bcdfghjklmnpqrstvwxz']
        consonant_freqs = {c: onegram_frequencies.get(c, 0) for c in consonants}
        
        self.logger.info("\nConsonant frequencies:")
        for c, freq in sorted(consonant_freqs.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"{c}: {freq:.6f}")
            
        return ''.join(sorted(consonants, key=lambda x: consonant_freqs[x], reverse=True)[:n])

    def score_permutation(self, letters: str, positions: List[str]) -> Tuple[float, Dict[str, float]]:
        """Score a permutation by frequency-weighted comfort of letter transitions."""
        position_map = dict(zip(letters, positions))
        total_score = 0
        bigram_scores = {}
        
        # For each possible letter pair
        for l1 in letters:
            for l2 in letters:
                if l1 != l2:
                    bigram = f"{l1}{l2}"
                    freq = bigram_frequencies.get(bigram, 0)
                    
                    # Get assigned positions
                    pos1, pos2 = position_map[l1], position_map[l2]
                    
                    # Map right-side positions to left-side equivalents
                    if pos1 in self.right_positions:
                        pos1 = self.position_mappings[pos1]
                    if pos2 in self.right_positions:
                        pos2 = self.position_mappings[pos2]
                    
                    # Get comfort score
                    comfort = self.comfort_scores.get((pos1, pos2), float('-inf'))
                    weighted_score = freq * comfort
                    bigram_scores[bigram] = weighted_score
                    total_score += weighted_score
                    
        return total_score, bigram_scores

    def find_best_placements(self, letters: str, positions: List[str], top_n: int) -> List[Tuple[float, List[str], Dict[str, float]]]:
        """Find optimal placements for letters."""
        self.logger.info(f"\nSearching for best placements of letters {letters}")
        self.logger.info(f"Available positions: {positions}")
        
        best_placements = []
        count = 0
        
        for pos_perm in permutations(positions, len(letters)):
            weighted_score, bigram_scores = self.score_permutation(letters, pos_perm)
            
            if len(best_placements) < top_n:
                heapq.heappush(best_placements, (weighted_score, pos_perm, bigram_scores))
            elif weighted_score > best_placements[0][0]:
                heapq.heappop(best_placements)
                heapq.heappush(best_placements, (weighted_score, pos_perm, bigram_scores))
            
            count += 1
            if count % 1000000 == 0:
                self.logger.info(f"Evaluated {count} permutations...")
                
        return sorted(best_placements, reverse=True)

    def optimize_consonant_layout(self) -> List[Tuple[float, List[str], Dict[str, float]]]:
        """Optimize consonant placement on the left side."""
        consonant_config = self.config.config['optimization']['consonants']
        num_consonants = consonant_config['num_consonants']
        top_n = consonant_config['top_n']
        
        consonants = self.get_top_consonants(num_consonants)
        self.logger.info(f"Optimizing placement for consonants: {consonants}")
        
        return self.find_best_placements(consonants, self.left_positions, top_n)

    def optimize_vowel_layout(self) -> List[Tuple[float, List[str], Dict[str, float]]]:
        """Optimize vowel placement on the right side."""
        vowel_config = self.config.config['optimization']['vowels']
        vowels = vowel_config['letters']
        top_n = vowel_config['top_n']
        
        self.logger.info(f"Optimizing placement for vowels: {vowels}")
        
        return self.find_best_placements(vowels, self.right_positions, top_n)

    def print_keyboard_layout(self, positions: Dict[str, str], title: str = "Layout"):
        """Print visual keyboard layout."""
        template = self.config.config['visualization']['keyboard_template']
        
        # Create layout chars dictionary with spaces as defaults
        layout_chars = {
            'q': ' ', 'w': ' ', 'e': ' ', 'r': ' ',
            'u': ' ', 'i': ' ', 'o': ' ', 'p': ' ',
            'a': ' ', 's': ' ', 'd': ' ', 'f': ' ',
            'j': ' ', 'k': ' ', 'l': ' ', 'sc': ' ',
            'z': ' ', 'x': ' ', 'c': ' ', 'v': ' ',
            'm': ' ', 'cm': ' ', 'dt': ' ', 'sl': ' '
        }
        
        # Fill in the positions from the layout
        for pos, letter in positions.items():
            if pos in layout_chars:
                layout_chars[pos] = letter.upper()
            elif pos == ';':
                layout_chars['sc'] = letter.upper()
            elif pos == ',':
                layout_chars['cm'] = letter.upper()
            elif pos == '.':
                layout_chars['dt'] = letter.upper()
            elif pos == '/':
                layout_chars['sl'] = letter.upper()
        
        # Print the layout
        print(template.format(title=title, **layout_chars))

def main():
    parser = argparse.ArgumentParser(description='Optimize keyboard layout.')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Initialize configuration and optimizer
    config = Config(args.config)
    optimizer = LayoutOptimizer(config)
    
    # Optimize consonants first
    consonant_results = optimizer.optimize_consonant_layout()
    consonants = optimizer.get_top_consonants(len(consonant_results[0][1]))
    
    # Optimize vowels
    vowel_results = optimizer.optimize_vowel_layout()
    vowels = config.config['optimization']['vowels']['letters']
    
    # Create layout mapping
    layout = {}
    
    # Map consonants
    for letter, pos in zip(consonants, consonant_results[0][1]):
        layout[pos] = letter
    
    # Map vowels
    for letter, pos in zip(vowels[:len(vowel_results[0][1])], vowel_results[0][1]):
        layout[pos] = letter
    
    # Print final layout
    optimizer.print_keyboard_layout(layout, "Optimized Layout")

if __name__ == "__main__":
    main()
