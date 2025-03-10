#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
Generate configuration files to run keyboard layout optimizations in parallel 
with specific letter-to-key constraints specified in each file.

This is step 2 of:
1. Generate layouts to see whether vowels aggregate on one side of the keyboard.
2. Optimally arrange 20-letter layouts with vowels on the left.
3. Optimally arrange letters in the least comfortable of 24 keys.

Constraints:
- items_to_assign is always "nsrhldcmfpgwyb" (14 letters)
- items_assigned is always "etaoiu" (6 letters)
- e will always be assigned to qwerty key D or F
- t will always be assigned to qwerty key J or K
- aoiu can be in any of the 10 most comfortable keys on the left: "QWERASDFCV"
- positions_to_assign: 14 of the remaining 20 keys

Step 2a: positions_assigned
---------------------------
e in key D/F, t in key J/K:
╭───────────────────────────────────────────────╮
│     │     │     │     ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  D  │  F  ║  J  │  K  │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │     │     ║     │     │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 2b: positions_assigned
---------------------------
aoiu in any of the top-10 (most comfortable) keys, 
except where e is already assigned:
╭───────────────────────────────────────────────╮
│  Q  │  W  │  E  │  R  ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║     │     │     │     │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║     │     │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

Step 2c: positions_to_assign 
----------------------------
nsrhldcmfpgwyb in any remaining top-20 keys:
╭───────────────────────────────────────────────╮
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│     │     │  C  │  V  ║  M  │  ,  │     │     │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

There are 12,096 valid configurations based on the constraints.

Example configuration:
  items_assigned: etaoiu
  positions_assigned: DJWERQ
  positions_to_assign: FSVACKLMIU;O,P
  Letter mappings:
    e -> D
    t -> J
    a -> W
    o -> E
    i -> R
    u -> Q

"""
import os
import yaml
import itertools

# Configuration
OUTPUT_DIR = 'configs'
ALL_KEYS = "FDSVERAWCQJKLMIU;O,P"

# Base configuration from the original config file
with open('config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

def generate_constraint_sets():
    """Generate all valid configurations based on the constraints."""
    # Fixed items for all configurations
    items_to_assign = "nsrhldcmfpgwyb"  # 14 letters
    items_assigned = "etaoiu"  # 6 letters
    
    # Position constraints
    e_positions   = ["D", "F"]
    t_positions   = ["J", "K"]
    u_positions   = ["Q", "W", "E", "R", "A", "S", "D", "F", "C", "V"]
    aoi_positions = ["Q", "W", "E", "R", "A", "S", "D", "F", "C", "V"]
    
    # Generate all valid configurations
    configs = []
    
    # Loop through all possible position combinations
    for e_pos in e_positions:
        for t_pos in t_positions:
            for u_pos in u_positions:
                # Skip invalid combinations (positions must be unique)
                if u_pos in [e_pos, t_pos]:
                    continue
                
                # Calculate remaining positions for a, o, i (excluding already assigned positions)
                remaining_positions = [pos for pos in aoi_positions if pos not in [e_pos, u_pos]]
                
                # If we don't have enough remaining positions, skip
                if len(remaining_positions) < 3:
                    continue
                
                # Generate all permutations of positions for a, o, i
                for aoi_pos in itertools.permutations(remaining_positions, 3):
                    # Get final position assignments
                    positions = {'e': e_pos, 't': t_pos, 'a': aoi_pos[0], 'o': aoi_pos[1], 'i': aoi_pos[2], 'u': u_pos}
                    
                    # Skip if we have any duplicates
                    if len(set(positions.values())) != 6:
                        continue
                    
                    # Create the positions_assigned string (must match the order of items_assigned)
                    positions_assigned = ''.join([positions[letter] for letter in items_assigned])
                    
                    # Create positions_to_assign (14 keys not used in positions_assigned)
                    used_positions = set(positions_assigned)
                    positions_to_assign = ''.join([pos for pos in ALL_KEYS[:20] if pos not in used_positions])[:14]
                    
                    # Add to configs if valid
                    if len(positions_to_assign) == 14:
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
        
if __name__ == "__main__":
    print("Generating keyboard layout configurations...")
    configs = generate_constraint_sets()
    print(f"Found {len(configs)} valid configurations based on the constraints.")
    create_config_files(configs)
    print(f"All configuration files have been generated in the '{OUTPUT_DIR}' directory.")
    
    # Print details about the first few configs
    for i in range(min(5, len(configs))):
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