import numpy as np
import matplotlib.pyplot as plt

days = np.arange(0, 61)

def simulate_group(days, peak, decay_rate, noise_std, participants):
    group_data = []
    for i in range(participants):
        growth = 1 / (1 + np.exp(-0.3 * (days - 10)))
        decay = np.exp(-decay_rate * (days - 20))
        decay[days < 20] = 1
        levels = growth * decay * peak

        np.random.seed(42 + i + int(peak))
        levels += np.random.normal(0, noise_std, size=levels.shape)

        group_data.append(levels)

    return np.array(group_data)

young_group = simulate_group(days, peak=100, decay_rate=0.05, noise_std=3, participants=10)
elderly_group = simulate_group(days, peak=80, decay_rate=0.08, noise_std=3, participants=10)

young_mean = young_group.mean(axis=0)
elderly_mean = elderly_group.mean(axis=0)

plt.figure(figsize=(10, 5))
for participant in young_group:
    plt.scatter(days, participant, color='blue', alpha=0.3, s=10)
for participant in elderly_group:
    plt.scatter(days, participant, color='orange', alpha=0.3, s=10)

plt.plot(days, young_mean, color='blue', label='Young Group Mean', linewidth=2)
plt.plot(days, elderly_mean, color='orange', label='Elderly Group Mean', linewidth=2)
plt.axvline(x=20, color='r', linestyle='--', label='Peak Response')

plt.xlabel("Days After Vaccination")
plt.ylabel("Antibody Level (AU/mL)")
plt.title("Simulated Antibody Response - Young vs Elderly (10 Participants Each)")
plt.legend()
plt.grid(True)
plt.show()
