import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------------
# telemetry data parameters
#------------------------------------
TOTAL_TIMESTAMPS = 10000

QBER_BASELINE = 0.015  # expected fraction of bit errors during normal operation
QBER_NOISE_STD = 0.01
JITTER_BASELINE_MEAN = 0.0  # (ns) mean jitter for SPADs
JITTER_STD = 0.1
DARK_COUNT_LAMBDA = 0.05  # lambda for poisson distribution representing normal dark counts

QBER_ATTACK_CHANGE_MAX = 0.055
JITTER_ATTACK_SHIFT_MAX = -0.55
DARK_COUNT_LAMBDA_ATTACK_MAX = 0.16

# episode generation parameters
MIN_GAP = 120
MIN_DURATION = 60
MAX_DURATION = 200
N_EPISODES = 24
STEALTH_PROB = 0.4

# -----------------------------------
# telemetry data generation
#------------------------------------
def generate_telemetry_dataset(seed, total_timestamps=TOTAL_TIMESTAMPS):

    rng = np.random.default_rng(seed)

    episodes = []
    cursor = rng.integers(200, 500) #first attack window between these
    for i in range(N_EPISODES):
        duration = int(rng.integers(MIN_DURATION, MAX_DURATION)) # how long the attack lasts
        start = cursor
        end = start + duration
        if end >= total_timestamps - 200:
            break # stops if at the end of the dataset
        intensity = float(rng.uniform(0.05, 1.0))  # how strong the signature is
        stealthy = bool(rng.random() < STEALTH_PROB) # whether the attack is stealthy or not
        episodes.append({"start": start, "end": end, "intensity": intensity, "stealthy": stealthy}) # save the episode information
        cursor = end + int(rng.integers(MIN_GAP, MIN_GAP * 3)) # random gap before next episode

   # arrays for storing telemetry data and attack metadata
    eavesdropper_status = np.zeros(total_timestamps, dtype=int)
    intensity_trace = np.zeros(total_timestamps, dtype=float)
    qber_intensity_trace = np.zeros(total_timestamps, dtype=float)
    jitter_intensity_trace = np.zeros(total_timestamps, dtype=float)
    dark_intensity_trace = np.zeros(total_timestamps, dtype=float)
    stealthy_trace = np.zeros(total_timestamps, dtype=bool)

    # populate the arrays with the generated episodes
    for eps in episodes:
        s, e, inten, stealthy = eps["start"], eps["end"], eps["intensity"], eps["stealthy"]
        # mark ground truth labels
        eavesdropper_status[s:e] = 1
        intensity_trace[s:e] = inten
        stealthy_trace[s:e] = stealthy
       
       # if stealthy suppress QBER and jitte signatures
        qber_intensity_trace[s:e] = inten * (rng.uniform(0.15, 0.4) if stealthy else 1.0)
        jitter_intensity_trace[s:e] = inten * (rng.uniform(0.5, 0.7) if stealthy else 1.0)
        dark_intensity_trace[s:e] = inten

    # ---------------------------------
    # feature generation
    # ---------------------------------

    # setting the mean QBER for each timestep depending on attack status
    qber_means = QBER_BASELINE + qber_intensity_trace * QBER_ATTACK_CHANGE_MAX
    # adding Gaussian noise
    qber_noise = rng.normal(loc=0.0, scale=QBER_NOISE_STD, size=total_timestamps)
    # making sure qber is between 0 and 1
    qber_feature = np.clip(qber_means + qber_noise, 0, 1)

    # setting the mean jitter during the attack
    jitter_means = JITTER_BASELINE_MEAN + jitter_intensity_trace * JITTER_ATTACK_SHIFT_MAX
    # generating jitter values using normal distribution around changing mean
    jitter_feature = rng.normal(loc=jitter_means, scale=JITTER_STD, size=total_timestamps)

    # setting dark count rate based on attack status
    dark_count_lambdas = DARK_COUNT_LAMBDA + dark_intensity_trace * (DARK_COUNT_LAMBDA_ATTACK_MAX - DARK_COUNT_LAMBDA)
    # generating photon count events using poisson distribution
    dark_count_feature = rng.poisson(lam=dark_count_lambdas, size=total_timestamps)

    # store all data in a dataframe
    df = pd.DataFrame({
        'timestamp': np.arange(total_timestamps),
        'qber': qber_feature,
        'jitter': jitter_feature,
        'dark_counts': dark_count_feature,
        'attack_label': eavesdropper_status,
        'attack_intensity': intensity_trace,
        'is_stealthy': stealthy_trace
    })

    return df


def plot_sanity_check(df, out_path='plots/telemetry_sanity_check.png'):
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # plot 1: QBER physical feature
    axes[0].plot(df['qber'], color='crimson', label='QBER')
    axes[0].set_ylabel('QBER')
    axes[0].legend(loc='upper right')

    # plot 2: Timing Jitter physical feature
    axes[1].plot(df['jitter'], color='navy', label='Timing Jitter (ns)')
    axes[1].set_ylabel('Jitter (ns)')
    axes[1].legend(loc='upper right')

    # plot 3: Dark Counts physical feature
    axes[2].plot(df['dark_counts'], color='darkgreen', label='Dark Counts')
    axes[2].set_ylabel('Counts')
    axes[2].legend(loc='upper right')

    # plot 4: Ground truth labels showing where Eve is active
    axes[3].plot(df['attack_label'], color='black', linestyle='--', label='Attack Ground Truth (Eve)')
    axes[3].set_ylabel('Status')
    axes[3].set_xlabel('Timestep')
    axes[3].legend(loc='upper right')

    plt.suptitle('Simulated Multi-Modal QKD Physical Telemetry channels', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


if __name__ == '__main__':
    # saving data and figures
    df = generate_telemetry_dataset(seed=42)

    csv_path = 'data/qkd_simulated_telemetry.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(csv_path, index=False)

    plot_sanity_check(df)
    print(f"Saved {csv_path} and plots/telemetry_sanity_check.png")