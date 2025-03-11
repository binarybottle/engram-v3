#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is step 1 of:
1. Optimally arrange 20-letter layouts constraining the most frequent letter.
2. Optimally arrange letters in the least comfortable of 24 keys.

Constraints:
- items_to_assign is always "nsrhldcumfpgwyb" (15 letters)
- items_assigned is always "etaoi" (5 letters)
- e will always be assigned to qwerty key D or F
- taoi can be in any of the 20 most comfortable (qwerty) keys: "FDSVERAWCQJKLMIU;O,P"
- positions_to_assign: 15 of the remaining 20 keys

Step 1a: positions_assigned
---------------------------
e in qwerty key D or F:
╭───────────────────────────────────────────────╮
│     │     │     │     ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  D  │  F  ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │     │     ║     │     │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 1b: positions_assigned
---------------------------
taoi in any of the top-20 (most comfortable) keys, 
except where e is already assigned:
╭───────────────────────────────────────────────╮
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║  M  │  ,  │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 1c: positions_to_assign 
----------------------------
nsrhldcumfpgwyb (15 letters) in any remaining top-20 keys:
╭───────────────────────────────────────────────╮
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║  M  │  ,  │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

There are 186,048 valid configurations based on the constraints.

Example configuration:
  items_assigned: etaoi
  positions_assigned: DFSVE
  positions_to_assign: RAWCQJKLMIU;O,P
  Letter mappings:
    e -> D
    t -> F
    a -> S
    o -> V
    i -> E

"""
import os
import math
import yaml
import itertools

# Configuration
OUTPUT_DIR = 'configs'
ALL_KEYS = "FDSVERAWCQJKLMIU;O,P"  # 20 most comfortable keys

# Base configuration from the original config file
with open('config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

def generate_constraint_sets():
    """Generate all valid configurations based on the constraints."""
    # Fixed items for all configurations
    items_assigned  = "etaoi"            #  5 letters
    items_to_assign = "nsrhldcumfpgwyb"  # 15 letters
    n_items_to_assign = len(items_to_assign)
    
    # Position constraints
    e_positions = ["D", "F"]
    # taoi can be in any of the most comfortable keys
    taoi_positions = list(ALL_KEYS)
    
    # Generate all valid configurations
    configs = []
    
    # Loop through all possible position combinations
    for e_pos in e_positions:
            # Calculate remaining positions for t, a, o, i (excluding already assigned positions)
            remaining_positions = [pos for pos in taoi_positions if pos not in [e_pos]]
            
            # Generate all combinations of 4 positions from remaining_positions for t, a, o, i
            for taoi_combo in itertools.combinations(remaining_positions, 4):
                # Generate all permutations of these 4 positions for t, a, o, i
                for taoi_perm in itertools.permutations(taoi_combo):
                    t_pos, a_pos, o_pos, i_pos = taoi_perm
                    
                    # Create final position assignments
                    positions = {'e': e_pos, 't': t_pos, 'a': a_pos, 'o': o_pos, 'i': i_pos}
                    
                    # Create the positions_assigned string (must match the order of items_assigned)
                    positions_assigned = ''.join([positions[letter] for letter in items_assigned])
                    
                    # Create positions_to_assign (14 keys not used in positions_assigned)
                    used_positions = set(positions_assigned)
                    positions_to_assign = ''.join([pos for pos in ALL_KEYS if pos not in used_positions])
                    
                    # Add to configs if valid and positions_to_assign has the correct number of positions
                    if len(positions_to_assign) == n_items_to_assign:
                        configs.append({
                            'items_to_assign': items_to_assign,
                            'positions_to_assign': positions_to_assign,
                            'items_assigned': items_assigned,
                            'positions_assigned': positions_assigned,
                            'items_to_constrain': "",  # No constraints for these configurations
                            'positions_to_constrain': ""
                        })
    
    return configs

def create_config_files(configs, nlayouts=100):
    """Create individual config file for each configuration."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print the total number of configs we're working with
    print(f"Creating {len(configs)} configuration files...")
    
    for i, config_params in enumerate(configs, 1):
        # Create a copy of the base config
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
        
        # Update optimization parameters
        for param, value in config_params.items():
            config['optimization'][param] = value
        
        # Set nlayouts
        config['optimization']['nlayouts'] = nlayouts
        
        # Set up unique output path
        #config['paths']['output']['layout_results_folder'] = f"output/layouts/config_{i}"
        #os.makedirs(config['paths']['output']['layout_results_folder'], exist_ok=True)
        
        # Write the configuration to a YAML file
        config_filename = f"{OUTPUT_DIR}/config_{i}.yaml"
        with open(config_filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Print progress feedback for every 100 files
        if i % 100 == 0:
            print(f"  Created {i}/{len(configs)} configuration files...")
        
if __name__ == "__main__":
    nlayouts = 100
    print("Generating keyboard layout configurations...")
    configs = generate_constraint_sets()
    print(f"Found {len(configs)} valid configurations based on the constraints.")
    create_config_files(configs, nlayouts)
    print(f"All configuration files have been generated in the '{OUTPUT_DIR}' directory.")
    
    # Print details about the first few configs
    num_examples = min(3, len(configs))
    print(f"\nShowing details for first {num_examples} configurations:")
    for i in range(num_examples):
        config = configs[i]
        print(f"\nConfig {i+1}:")
        print(f"  items_assigned: {config['items_assigned']}")
        print(f"  positions_assigned: {config['positions_assigned']}")
        print(f"  positions_to_assign: {config['positions_to_assign']}")
        
        # Map letters to positions for clarity
        letter_positions = {config['items_assigned'][j]: config['positions_assigned'][j] 
                          for j in range(len(config['items_assigned']))}
        print("  Letter mappings:")
        for letter, pos in letter_positions.items():
            print(f"    {letter} -> {pos}")
            

    # Calculate the number of possible combinations 
    e_positions = 2  # D or F
    # Calculate "19 choose 4" directly using the binomial coefficient formula
    taoi_combinations = math.comb(19, 4)  # In Python 3.8+
    taoi_permutations = 24  # 4! = 24 ways to arrange t, a,o,i    
    print(f"\nTheoretical maximum configurations: {e_positions * taoi_combinations * taoi_permutations}")
    print(f"Actual configurations: {len(configs)}")
    print("Note: The actual number is lower due to filtering out invalid combinations")