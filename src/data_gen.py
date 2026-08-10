import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------------
# telemetry data parameters
#------------------------------------
total_timestamps = 10000

qber_baseline = 0.015 # expected fraction of bit errors during normal operation
jitter_baseline_mean = 0.0 # (ns) mean jitter for SPADs
jitter_std = 0.1
dark_count_lambda = 0.05 # lambda for poisson distribution representing normal dark counts

qber_attack = 0.15 # to 0.25. When eve eavesdrops the QBER raises 
jitter_attack_shift = -1.7 # eves interruption also shifts the time of a photons arrival
dark_count_lambda_attack = 0.20 #also increases dark counts

#-----------------------------------
# attack injection
#-----------------------------------

eavesdropper_status = np.zeros(total_timestamps,dtype=int)

# artificially injecting eavesdropping attacks at specific times
eavesdropper_status[2000:3500] = 1
eavesdropper_status[7000:8500] = 1

#---------------------------------
# feature generation
#---------------------------------

# setting the mean QBER for each timestep depending on attack status
qber_means = np.where(eavesdropper_status == 0, qber_baseline, qber_attack)
# adding Gaussian noise 
qber_noise = np.random.normal(loc=0, scale=0.01, size=total_timestamps)
# making sure qber is between 0 and 1
qber_feature = np.clip(qber_means + qber_noise, 0, 1)

# setting the mean jitter during the attack
jitter_means = np.where(eavesdropper_status == 0, jitter_baseline_mean, jitter_baseline_mean + jitter_attack_shift)
# generating jitter values using normal distribution around changing mean
jitter_feature = np.random.normal(loc=jitter_means, scale=jitter_std, size=total_timestamps)

# setting dark count rate bases on attack status
dark_count_lambdas = np.where(eavesdropper_status == 0, dark_count_lambda, dark_count_lambda_attack)
# generating photon count events using poisson distribution
dark_count_feature = np.random.poisson(lam=dark_count_lambdas, size=total_timestamps)

# store all data in a data frame and save to csv
df = pd.DataFrame({
    'timestamp': np.arange(total_timestamps),
    'qber': qber_feature,
    'jitter': jitter_feature,
    'dark_counts': dark_count_feature,
    'attack_label': eavesdropper_status
})


csv_path = 'data/qkd_simulated_telemetry.csv'
df.to_csv(csv_path, index=False)

# 4 panel subplot of QBER, timing jitter, dark counnts and attack status in that order
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

plt.savefig('plots/telemetry_sanity_check.png', dpi=300)
plt.show()
        