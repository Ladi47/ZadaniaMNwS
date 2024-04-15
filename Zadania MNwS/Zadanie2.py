import numpy as np
from scipy.stats import chi2, kstest, chisquare
import matplotlib.pyplot as plt

# Funkcja do generowania danych z rozkładu chi-kwadrat
def generate_chi_squared_data(k, n):
    return np.random.chisquare(k, n)

# Funkcja do przeprowadzenia testu Kołmogorowa
def perform_ks_test(data, k):
    return kstest(data, 'chi2', args=(k,))

# Funkcja do przeprowadzenia testu chi-kwadrat
def perform_chi_squared_test(data, k):
    observed_freq = np.bincount(np.digitize(data, np.arange(0, max(data)+1)))
    expected_freq = chi2.pdf(np.arange(len(observed_freq)), k) * np.sum(observed_freq)
    expected_freq = expected_freq / np.sum(expected_freq) * np.sum(observed_freq)  # Normalizacja oczekiwanych częstości
    epsilon = 1e-10  # Mała wartość, którą dodamy do oczekiwanych częstości, aby uniknąć dzielenia przez zero
    expected_freq[expected_freq == 0] = epsilon  # Zastąpienie zerowych wartości oczekiwanych częstości przez epsilon
    return chisquare(observed_freq, expected_freq, ddof=k)

# Funkcja do wyliczenia odsetka odrzuceń hipotezy głównej
def calculate_rejection_rate(test_results):
    return np.mean(np.array(test_results) < 0.05)

# Parametry symulacji
num_samples = [100, 500, 1000, 5000]  # różne liczby danych
df_values = [2, 5, 10]  # różne stopnie swobody

# Liczba powtórzeń symulacji
num_simulations = 1000

# Wyniki testów
ks_rejection_rates_classic = np.zeros((len(num_samples), len(df_values)))
chi_squared_rejection_rates_classic = np.zeros((len(num_samples), len(df_values)))
ks_rejection_rates_pit = np.zeros((len(num_samples), len(df_values)))
chi_squared_rejection_rates_pit = np.zeros((len(num_samples), len(df_values)))

# Symulacja
for i, n in enumerate(num_samples):
    for j, df in enumerate(df_values):
        ks_classic_results = []
        chi_squared_classic_results = []
        ks_pit_results = []
        chi_squared_pit_results = []

        for _ in range(num_simulations):
            # Generowanie danych
            data = generate_chi_squared_data(df, n)

            # Testowanie z użyciem klasycznego testu
            ks_classic_results.append(perform_ks_test(data, df).pvalue)
            chi_squared_classic_results.append(perform_chi_squared_test(data, df).pvalue)

            # Testowanie z użyciem PIT
            data_sorted = np.sort(data)
            uniform_samples = np.random.uniform(0, 1, n)
            transformed_data = chi2.ppf(uniform_samples, df)
            ks_pit_results.append(perform_ks_test(transformed_data, df).pvalue)
            chi_squared_pit_results.append(perform_chi_squared_test(transformed_data, df).pvalue)

        # Wyliczenie odsetka odrzuceń hipotezy głównej
        ks_rejection_rates_classic[i, j] = calculate_rejection_rate(ks_classic_results)
        chi_squared_rejection_rates_classic[i, j] = calculate_rejection_rate(chi_squared_classic_results)
        ks_rejection_rates_pit[i, j] = calculate_rejection_rate(ks_pit_results)
        chi_squared_rejection_rates_pit[i, j] = calculate_rejection_rate(chi_squared_pit_results)

# Wykresy
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

colors = ['blue', 'red', 'green', 'orange']

for ax, rejection_rates, title in zip(axes.flatten(),
                                      [ks_rejection_rates_classic, chi_squared_rejection_rates_classic,
                                       ks_rejection_rates_pit, chi_squared_rejection_rates_pit],
                                      ['Kolmogorow-Smirnow (Classic)', 'Chi-kwadrat (Classic)',
                                       'Kolmogorow-Smirnow (PIT)', 'Chi-kwadrat (PIT)']):
    for i in range(len(num_samples)):
        if 'Chi-kwadrat' in title:
            ax.plot(df_values, rejection_rates[i], label=f'{num_samples[i]} próbek', linestyle='-', marker='o', color=colors[i])
        else:
            ax.plot(df_values, rejection_rates[i], label=f'{num_samples[i]} próbek', linestyle='--', marker='x', color=colors[i])

    ax.set_title(title)
    ax.set_xlabel('Stopnie swobody') 
    ax.set_ylabel('Odsetek odrzuceń')
    ax.legend()

plt.tight_layout()
plt.show()
