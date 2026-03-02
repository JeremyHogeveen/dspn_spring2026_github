import os
import pandas as pd
import numpy as np

# 1. Setup Directories
data_dir = '/Users/jeremyhogeveen/Dropbox/Winter_2026/teaching/DSPN_Spring_2026/workdir/data/abcd_synthetic'
os.makedirs(data_dir, exist_ok=True)

# 2. Generate Synthetic Subject Pool
n_total_subjects = 2000 # Updated to 2000 for the full dataframes
n_subset = 150          # Subset count for subjectkeys.txt

baseline_keys = [f"NDAR_INV{str(i).zfill(6)}" for i in range(1, n_total_subjects + 1)]

# Create the longitudinal event list
event_list = [('baseline_year_1_arm_1', sk) for sk in baseline_keys] # Everyone
event_list += [('6_month_followup', sk) for sk in baseline_keys[:800]] # 800 subjects have a 6mo
event_list += [('12_month_followup', sk) for sk in baseline_keys[400:1000]] # 600 subjects have a 12mo

df_base_events = pd.DataFrame(event_list, columns=['eventname', 'subjectkey'])
n_total_rows = len(df_base_events)

# Helper function to generate ABCD-style text files with a description row
def save_abcd_format(df, filename):
    desc_row = {col: f"{col} description" for col in df.columns}
    desc_row['subjectkey'] = 'The subject identifier'
    df_out = pd.concat([pd.DataFrame([desc_row]), df], ignore_index=True)
    df_out.to_csv(os.path.join(data_dir, filename), sep='\t', index=False)

# 3. Create Basic Files (Demographics, Site, etc.)

# UPDATED: Generate and save only a subset of 150 keys to subjectkeys.txt
subset_keys = np.random.choice(baseline_keys, size=n_subset, replace=False)
pd.DataFrame({'subjectkey': subset_keys}).to_csv(
    os.path.join(data_dir, 'subjectkeys.txt'), sep='\t', index=False
)

# Generate demographic data for all events
demo_cols = ['sex', 'demo_gender_id_v2', 'interview_age', 'demo_comb_income_v2', 
             'demo_fam_exp1_v2', 'demo_prnt_marital_v2', 'demo_prnt_ethn_v2'] + \
            [f'demo_prnt_race_a_v2___{i}' for i in range(10, 25)]

df_demo = df_base_events.copy()
for col in demo_cols:
    df_demo[col] = np.random.choice([1, 2, 0, 777], n_total_rows) 
    
# Make age increase across visits
df_demo.loc[df_demo['eventname'] == '6_month_followup', 'interview_age'] += 6
df_demo.loc[df_demo['eventname'] == '12_month_followup', 'interview_age'] += 12

save_abcd_format(df_demo, 'pdem02.txt')

df_site = df_base_events.copy()
df_site['site_id_l'] = np.random.choice(['site01', 'site02', 'site03'], n_total_rows)
save_abcd_format(df_site, 'abcd_lt01.txt')

cbcl_cols = ['cbcl_scr_syn_anxdep_t', 'cbcl_scr_syn_withdep_t', 'cbcl_scr_syn_somatic_t', 
             'cbcl_scr_syn_internal_t', 'cbcl_scr_syn_aggressive_t', 'cbcl_scr_syn_attention_t', 
             'cbcl_scr_syn_rulebreak_t', 'cbcl_scr_syn_external_t']
df_cbcl = df_base_events.copy()
for col in cbcl_cols:
    df_cbcl[col] = np.random.randint(30, 80, n_total_rows)
save_abcd_format(df_cbcl, 'abcd_cbcls01.txt')

# 4. UPDATED: fMRI MID Task Volumes & Motivational ROIs
mid_qc_cols = ['mid_nvols_all', 'mid_nvols_clean', 'mid_mean_motion']

mid_conds = ['ant_rew_v_neut', 'ant_loss_v_neut', 'pos_v_neut_fb', 'neg_v_neut_fb']
mid_rois = ['l_nacc', 'r_nacc', 'vmpfc', 'anterior_insula', 'l_amygdala', 'r_amygdala', 'dacc']

mid_neural_cols = []
for cond in mid_conds:
    for roi in mid_rois:
        mid_neural_cols.append(f'mid_{cond}_{roi}')

# Introduce missingness specifically in the fMRI data (e.g., missed scans at follow-up)
df_mid_events = df_base_events.sample(frac=0.85, random_state=42).copy()
n_mid_rows = len(df_mid_events)

df_mid1 = df_mid_events.copy()
df_mid2 = df_mid_events.copy()

# Add QC variables to part 1
for col in mid_qc_cols:
    df_mid1[col] = np.random.uniform(0, 5, n_mid_rows) if "motion" in col else np.random.randint(150, 500, n_mid_rows)

# Split the neural columns across two files to mimic the split text files in ABCD
split_idx = len(mid_neural_cols) // 2
for col in mid_neural_cols[:split_idx]:
    df_mid1[col] = np.random.uniform(-2, 2, n_mid_rows)
for col in mid_neural_cols[split_idx:]:
    df_mid2[col] = np.random.uniform(-2, 2, n_mid_rows)

save_abcd_format(df_mid1, 'midaparc03.txt')
save_abcd_format(df_mid2, 'midaparcp203.txt')

# 5. UPDATED: fMRI MID Behavioral RT Data 
df_midrt = df_base_events.copy()
df_midrt['rt_neut_fb'] = np.random.uniform(200, 500, n_total_rows)
df_midrt['rt_rew_pos_fb'] = np.random.uniform(200, 500, n_total_rows)
df_midrt['rt_loss_neg_fb'] = np.random.uniform(200, 500, n_total_rows)
save_abcd_format(df_midrt, 'abcd_mid02.txt')

print(f"Synthetic longitudinal ABCD data generated successfully for {n_total_rows} total visits across {n_total_subjects} subjects!")
print(f"Subset of {n_subset} subjects successfully saved to subjectkeys.txt!")