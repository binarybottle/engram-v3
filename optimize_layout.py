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
    """Main class for keyboard layout optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.comfort_scores = self._load_comfort_scores()
        
        # Load layout settings from config
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
        """Get top n most frequent consonants."""
        consonants = sorted(
            [(c, freq) for c, freq in onegram_frequencies.items() 
             if c.isalpha() and c.lower() not in 'aeiou'],
            key=lambda x: x[1],
            reverse=True
        )
        return ''.join(c for c, _ in consonants[:n])

    def find_best_placements(self, letters: str, positions: List[str], top_n: int) -> List[Tuple[float, List[str], Dict[str, float]]]:
        """Find the best placements for given letters in given positions."""
        self.logger.info(f"Searching for best placements of letters {letters}")
        self.logger.info(f"Available positions: {positions}")
        
        best_placements = []
        count = 0
        
        for pos_perm in permutations(positions, len(letters)):
            weighted_score = 0
            bigram_scores = {}
            
            # Calculate score for this permutation
            for i, letter in enumerate(letters):
                pos = pos_perm[i]
                for other_letter in letters:
                    bigram = (letter, other_letter)
                    if bigram in bigram_frequencies:
                        score = (bigram_frequencies[bigram] * 
                                self.comfort_scores.get((pos, self.position_mappings.get(pos, pos)), 0))
                        weighted_score += score
                        bigram_scores[bigram] = score
            
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
        
        # Convert special characters
        special_chars = {';': 'sc', ',': 'cm', '.': 'dt', '/': 'sl'}
        
        # Update layout with provided positions
        for pos, letter in positions.items():
            layout_key = special_chars.get(pos, pos)
            layout_chars[layout_key] = letter.upper()
        
        # Print the layout
        print(template.format(
            title=title,
            **layout_chars
        ))

def main():
    parser = argparse.ArgumentParser(description='Optimize keyboard layout.')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Initialize configuration and optimizer
    config = Config(args.config)
    optimizer = LayoutOptimizer(config)
    
    # Optimize consonants
    consonant_results = optimizer.optimize_consonant_layout()
    optimizer.logger.info("Top consonant placements:")
    for score, placement, details in consonant_results:
        optimizer.logger.info(f"Score: {score:.2f}, Placement: {placement}")
    
    # Optimize vowels
    vowel_results = optimizer.optimize_vowel_layout()
    optimizer.logger.info("Top vowel placements:")
    for score, placement, details in vowel_results:
        optimizer.logger.info(f"Score: {score:.2f}, Placement: {placement}")
    
    # Create and print best layout
    best_consonant_placement = consonant_results[0][1]
    best_vowel_placement = vowel_results[0][1]
    
    layout = {}
    for i, letter in enumerate(best_consonant_placement):
        layout[letter] = optimizer.left_positions[i]
    for i, letter in enumerate(best_vowel_placement):
        layout[letter] = optimizer.right_positions[i]
    
    optimizer.print_keyboard_layout(layout, "Optimized Layout")

if __name__ == "__main__":
    main()