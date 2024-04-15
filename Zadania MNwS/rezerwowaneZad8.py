import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kstest, norm, t, chi2

# Parametry symulacji
sample_sizes = [50, 100, 200, 500, 1000, 2000]  # Rozmiary próbek
df_values = np.arange(1, 31, 5)  # Stopnie swobody dla rozkładu t-Studenta i chi-kwadrat

# Liczba powtórzeń symulacji
num_simulations = 1000

# Funkcja do przeprowadzenia symulacji i obliczenia mocy testu KS
def compute_power(sample_size, df, dist):
    power = 0
    for _ in range(num_simulations):
        # Generowanie próbki z rozkładu t-Studenta lub chi-kwadrat
        if dist == 't':
            sample = t.rvs(df, size=sample_size)
        elif dist == 'chi2':
            sample = chi2.rvs(df, size=sample_size)
        
        # Standaryzacja danych
        sample = (sample - np.mean(sample)) / np.std(sample)
        
        # Transformacja PIT
        sample = norm.cdf(sample)
        
        # Wykonanie testu KS
        _, p_value = kstest(sample, 'uniform')
        
        # Jeśli otrzymano istotny wynik (p_value < 0.05), zwiększ moc testu
        if p_value < 0.05:
            power += 1
            
    return power / num_simulations

# Przeprowadzenie symulacji dla wszystkich kombinacji rozmiarów próbek i stopni swobody
powers_t = np.zeros((len(sample_sizes), len(df_values)))
powers_chi2 = np.zeros((len(sample_sizes), len(df_values)))

for i, sample_size in enumerate(sample_sizes):
    for j, df in enumerate(df_values):
        powers_t[i, j] = compute_power(sample_size, df, 't')
        powers_chi2[i, j] = compute_power(sample_size, df, 'chi2')

# Create a colormap
cmap = plt.cm.hot

# Create a legend
colors = [cmap(i) for i in np.linspace(0, 1, 6)]
labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]

# Wykresy
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

ax[0].imshow(powers_t, cmap='hot', interpolation='nearest', aspect='auto')
ax[0].set_xticks(np.arange(len(df_values)))
ax[0].set_yticks(np.arange(len(sample_sizes)))
ax[0].set_xticklabels(df_values)
ax[0].set_yticklabels(sample_sizes)
ax[0].set_xlabel('Stopnie swobody')
ax[0].set_ylabel('Rozmiar próbki')
ax[0].set_title('Moc testu KS dla rozkładu t-Studenta')
ax[0].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Power')

ax[1].imshow(powers_chi2, cmap='hot', interpolation='nearest', aspect='auto')
ax[1].set_xticks(np.arange(len(df_values)))
ax[1].set_yticks(np.arange(len(sample_sizes)))
ax[1].set_xticklabels(df_values)
ax[1].set_yticklabels(sample_sizes)
ax[1].set_xlabel('Stopnie swobody')
ax[1].set_ylabel('Rozmiar próbki')
ax[1].set_title('Moc testu KS dla rozkładu chi-kwadrat')
ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Power')

plt.tight_layout()
plt.show()
