#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is step 1 of:
1. Optimally arrange 20-letter layouts constraining the top 2 letters.
2. Optimally arrange letters in the least comfortable of 24 keys.

Constraints:
- items_to_assign is always "nsrhldcumfpgwyb" (15 letters)
- items_assigned is always "etaoi" (5 letters)
- e: assigned to qwerty key D or F
- t: any available key of the top 6 (most comfortable) keys: "FDSJKL"
- aoi: any available keys of the top 20 keys: "FDSVERAWCQJKLMIU;O,P"
- positions_to_assign: 15 remaining of the top 20 keys

Step 1a: constrain e in positions_assigned
-------------------------------------------------
e in (qwerty) key D/F:
╭───────────────────────────────────────────────╮
│     │     │     │     ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  D  │  F  ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │     │     ║     │     │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 1b: constrain t in positions_assigned
-------------------------------------------------
t in any of the top-6 (most comfortable) keys, 
except where e is already assigned:
╭───────────────────────────────────────────────╮
│     │     │     │     ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │  S  │  D  │  F  ║  J  │  K  │  L  │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │     │     ║     │     │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 1c: constrain aoi in positions_assigned
-------------------------------------------------
aoi in any of the top-20 keys, 
except where e and t are already assigned:
╭───────────────────────────────────────────────╮
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║  M  │  ,  │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 1d: assign 15 letters to positions_to_assign 
-------------------------------------------------
nsrhldcumfpgwyb (15 letters) in any remaining top-20 keys:
╭───────────────────────────────────────────────╮
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║  M  │  ,  │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

There are 48,960 valid configurations based on the constraints.
(If t were not constrained, there would be 186,048 valid configurations.)

Example configuration:
  items_assigned: etaoi
  positions_assigned: FDSVE
  positions_to_assign: RAWCQJKLMIU;O,P
  Letter mappings:
    e -> F
    t -> D
    a -> S
    o -> V
    i -> E

"""
import os
import yaml
import math
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
    e_positions = ["F", "D"]
    t_positions = ["F", "D", "S", "J", "K", "L"]
    # aoi can be in any of the most comfortable keys
    aoi_positions = list(ALL_KEYS)
    
    # Generate all valid configurations
    configs = []
    
    # Loop through all possible position combinations
    for e_pos in e_positions:
        for t_pos in [pos for pos in t_positions if pos != e_pos]:
            # Calculate remaining positions for a, o, i (excluding already assigned positions)
            remaining_positions = [pos for pos in aoi_positions if pos not in [e_pos, t_pos]]
            
            # Generate all combinations of 3 positions from remaining_positions for a, o, i
            for aoi_combo in itertools.combinations(remaining_positions, 3):
                # Generate all permutations of these 3 positions for a, o, i
                for aoi_perm in itertools.permutations(aoi_combo):
                    a_pos, o_pos, i_pos = aoi_perm
                    
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
    e_positions = 2
    t_valid_positions = 5  # 6 keys minus 1 where e is placed
    top_keys = len(ALL_KEYS)
    aoi_combinations = math.comb(18, 3)  # 816 ways to choose 3 from 18
    aoi_permutations = 6  # 3! = 6 ways to arrange a,o,i
    
    print(f"\nTheoretical maximum configurations: {e_positions * t_valid_positions * aoi_combinations * aoi_permutations}")
    print(f"Actual configurations: {len(configs)}")
