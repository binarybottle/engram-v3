#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is step 1 of:
1. Generate layouts to see whether vowels aggregate on one side of the keyboard.
2. Optimally arrange 20-letter layouts with vowels on the left.
3. Optimally arrange letters in the least comfortable of 24 keys.

Constraints:
- items_to_assign is always "nsrhldcumfpgw" (13 letters)
- items_assigned is always "etaoi" (5 letters)
- e will always be assigned to qwerty key D or F
- t will always be assigned to qwerty key J or K
- aoi can be in any of the 18 most comfortable keys: "FDSVERAWCJKLMIU;O,"
- positions_to_assign: 13 of the remaining 18 keys

Step 1a: positions_assigned
---------------------------
e in key D/F, t in key J/K:
╭───────────────────────────────────────────────╮
│     │     │     │     ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  D  │  F  ║  J  │  K  │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │     │     ║     │     │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 1b: positions_assigned
---------------------------
aoi in any of the top-18 (most comfortable) keys, 
except where e and t are already assigned:
╭───────────────────────────────────────────────╮
│     │  W  │  E  │  R  ║  U  │  I  │  O  │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║  M  │  ,  │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 1c: positions_to_assign 
----------------------------
nsrhldcumfpgw in any remaining top-18 keys:
╭───────────────────────────────────────────────╮
│     │  W  │  E  │  R  ║  U  │  I  │  O  │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║  M  │  ,  │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

There are 13,440 valid configurations based on the constraints.

Example configuration:
  items_assigned: etaoi
  positions_assigned: DJFSV
  positions_to_assign: ERAWCKLMIU;O,
  Letter mappings:
    e -> D
    t -> J
    a -> F
    o -> S
    i -> V

"""
import os
import yaml
import itertools

# Configuration
OUTPUT_DIR = 'configs'
ALL_KEYS = "FDSVERAWCJKLMIU;O,"  # 18 most comfortable keys

# Base configuration from the original config file
with open('config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

def generate_constraint_sets():
    """Generate all valid configurations based on the constraints."""
    # Fixed items for all configurations
    items_to_assign = "nsrhldcumfpgw"  # 13 letters
    items_assigned = "etaoi"  # 5 letters
    n_items_to_assign = len(items_to_assign)
    
    # Position constraints
    e_positions = ["D", "F"]
    t_positions = ["J", "K"]
    # aoi can be in any of the most comfortable keys
    aoi_positions = list(ALL_KEYS)
    
    # Generate all valid configurations
    configs = []
    
    # Loop through all possible position combinations
    for e_pos in e_positions:
        for t_pos in t_positions:
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
                    
                    # Add to configs if valid and positions_to_assign has exactly 14 positions
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

def create_config_files(configs):
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
        
        # Set nlayouts to 100
        config['optimization']['nlayouts'] = 100
        
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
    print("Generating keyboard layout configurations...")
    configs = generate_constraint_sets()
    print(f"Found {len(configs)} valid configurations based on the constraints.")
    create_config_files(configs)
    print(f"All configuration files have been generated in the '{OUTPUT_DIR}' directory.")
    
    # Print details about the first few configs
    num_examples = min(5, len(configs))
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
    t_positions = 2  # J or K
    top_keys = len(ALL_KEYS)
    aoiu_positions = top_keys - 2  # Excluding where e and t are placed
    aoiu_combinations = len(list(itertools.combinations(range(aoiu_positions), 4)))
    aoiu_permutations = 24  # 4! = 24 ways to arrange a,o,i,u
    
    print(f"\nTheoretical maximum configurations: {e_positions * t_positions * aoiu_combinations * aoiu_permutations}")
    print(f"Actual configurations: {len(configs)}")
    print("Note: The actual number is lower due to filtering out invalid combinations")