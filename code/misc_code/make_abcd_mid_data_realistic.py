import pandas as pd
import numpy as np
import os
from scipy.stats import norm

def make_fmri_data_realistic(input_file, output_file):
    # Load the data, treating row 1 as header and row 2 as ABCD-style descriptions
    df = pd.read_csv(input_file, sep='\t', header=0)
    
    # Separate the description row from the actual subject data
    desc_row = df.iloc[0:1]
    data_df = df.iloc[1:].copy()
    
    # Identify the fMRI contrast columns
    contrast_cols = [col for col in df.columns if 'mid_ant_' in col]
    
    # Define realistic target distributions: (Mean, Standard Deviation)
    target_dists = {
        'mid_ant_rew_v_neut_l_nacc': (1.20, 0.75),
        'mid_ant_rew_v_neut_r_nacc': (1.25, 0.80),
        'mid_ant_rew_v_neut_vmpfc': (0.80, 0.65),
        'mid_ant_rew_v_neut_anterior_insula': (1.05, 0.60),
        'mid_ant_rew_v_neut_l_amygdala': (0.30, 0.45),
        'mid_ant_rew_v_neut_r_amygdala': (0.35, 0.50),
        'mid_ant_rew_v_neut_dacc': (0.85, 0.60),
        
        'mid_ant_loss_v_neut_l_nacc': (0.40, 0.70),
        'mid_ant_loss_v_neut_r_nacc': (0.45, 0.75),
        'mid_ant_loss_v_neut_vmpfc': (0.25, 0.60),
        'mid_ant_loss_v_neut_anterior_insula': (0.90, 0.65),
        'mid_ant_loss_v_neut_l_amygdala': (0.55, 0.50),
        'mid_ant_loss_v_neut_r_amygdala': (0.60, 0.55),
        'mid_ant_loss_v_neut_dacc': (0.70, 0.60)
    }
    
    # Apply Rank-Based Inverse Normal Transformation
    for col in contrast_cols:
        # Ensure data is float and handle any potential NaNs
        vals = pd.to_numeric(data_df[col], errors='coerce')
        valid_idx = vals.dropna().index
        valid_vals = vals.loc[valid_idx]
        
        if len(valid_vals) > 0:
            # 1. Rank the data
            ranks = valid_vals.rank(method='average')
            
            # 2. Convert ranks to percentiles: (r - 0.5) / N
            n = len(valid_vals)
            percentiles = (ranks - 0.5) / n
            
            # 3. Force into a standard normal distribution (mean=0, std=1)
            z_scores = norm.ppf(percentiles)
            
            # 4. Apply the new realistic mean and standard deviation
            target_mean, target_std = target_dists.get(col, (0, 1)) 
            transformed_vals = (z_scores * target_std) + target_mean
            
            # Put back into the dataframe, rounded to 4 decimals
            data_df.loc[valid_idx, col] = np.round(transformed_vals, 4)
        
    # Recombine the description row with the transformed data
    final_df = pd.concat([desc_row, data_df], ignore_index=True)
    
    # Export the realistic dataset
    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"Transformation complete. Saved to:\n{output_file}")

if __name__ == "__main__":
    base_dir = "/Users/jeremyhogeveen/Dropbox/Winter_2026/teaching/DSPN_Spring_2026/workdir"
    
    input_filepath = os.path.join(base_dir, "data", "abcd_synthetic", "midaparc03.txt")
    output_filepath = os.path.join(base_dir, "data", "abcd_synthetic", "midaparc03_redo.txt")
    
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at {input_filepath}")
    else:
        make_fmri_data_realistic(input_filepath, output_filepath)