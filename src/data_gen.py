import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


total_timestamps = 10000
qber_baseline = 0.015
jitter_baseline_mean = 0.0
jitter_std = 0.1
dark_count_lambda = 0.05

qber_attack = 0.15 # to 0.25
jitter_attack_shift = -1.7
dark_count_lambda_attack = 0.20

eavesdropper_status = np.zeros(total_timestamps,dtype=int)

eavesdropper_status[2000:3500] = 1
eavesdropper_status[7000:8500] = 1


qber_means = np.where(eavesdropper_status == 0, qber_baseline, qber_attack)
qber_noise = np.random.normal(loc=0, scale=0.01, size=total_timestamps)
qber_feature = np.clip(qber_means + qber_noise, 0, 1)


jitter_means = np.where(eavesdropper_status == 0, jitter_baseline_mean, jitter_baseline_mean + jitter_attack_shift)
jitter_feature = np.random.normal(loc=jitter_means, scale=jitter_std, size=total_timestamps)


dark_count_lambdas = np.where(eavesdropper_status == 0, dark_count_lambda, dark_count_lambda_attack)
dark_count_feature = np.random.poisson(lam=dark_count_lambdas, size=total_timestamps)

# --- 1. Assemble into DataFrame ---
df = pd.DataFrame({
    'timestamp': np.arange(total_timestamps),
    'qber': qber_feature,
    'jitter': jitter_feature,
    'dark_counts': dark_count_feature,
    'attack_label': eavesdropper_status
})

# --- 2. Ensure data directory exists & save CSV ---
os.makedirs('data', exist_ok=True)
csv_path = 'data/qkd_simulated_telemetry.csv'
df.to_csv(csv_path, index=False)
print(f"Dataset successfully generated and saved to {csv_path}")

# --- 3. Quick Visual Sanity Check ---
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

axes[0].plot(df['qber'], color='crimson', label='QBER')
axes[0].set_ylabel('QBER')
axes[0].legend(loc='upper right')

axes[1].plot(df['jitter'], color='navy', label='Timing Jitter (ns)')
axes[1].set_ylabel('Jitter (ns)')
axes[1].legend(loc='upper right')

axes[2].plot(df['dark_counts'], color='darkgreen', label='Dark Counts')
axes[2].set_ylabel('Counts')
axes[2].legend(loc='upper right')

axes[3].plot(df['attack_label'], color='black', linestyle='--', label='Attack Ground Truth (Eve)')
axes[3].set_ylabel('Status')
axes[3].set_xlabel('Timestep')
axes[3].legend(loc='upper right')

plt.suptitle('Simulated Multi-Modal QKD Physical Telemetry Stream', fontsize=14)
plt.tight_layout()

# Save plot for your thesis / inspection
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/telemetry_sanity_check.png', dpi=300)
plt.show()
        