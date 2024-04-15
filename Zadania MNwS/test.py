import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Funkcja do generowania danych z rozkładu NIG
def generate_nig_data(params, size):
    mu, alpha, beta, delta = params
    Z = np.random.normal(size=size)
    V = np.random.chisquare(beta, size=size)
    X = mu + delta * (Z / np.sqrt(V)) + alpha * Z
    return X

# Funkcja do przeprowadzania testu Kołmogorowa-Smirnowa
def perform_ks_test(data):
    ks_stat, p_value = stats.kstest(data, 'norm')
    return p_value

# Parametry symulacji
num_samples = [50, 100, 500, 1000]  # różne liczby danych
n_trials = 1000  # liczba prób w każdej symulacji
params_NIG = [(1, 2, 3, 4), (2, 3, 2, 1), (3, 2, 1, 0.5)]  # różne parametry NIG
params_gamma = [(2, 0), (5, 0), (2, 0.5)]  # różne parametry gamma

# Wykonanie symulacji
results_no_pit = np.zeros((len(num_samples), len(params_NIG) + len(params_gamma)))
results_pit = np.zeros((len(num_samples), len(params_NIG) + len(params_gamma)))

for i, n in enumerate(num_samples):
    for j, params in enumerate(params_NIG):
        for k in range(n_trials):
            data = generate_nig_data(params, n)
            p_value = perform_ks_test(data)
            results_no_pit[i, j] += p_value
            transformed_data = stats.norm.ppf(stats.norm.cdf(data))
            p_value_pit = stats.kstest(transformed_data, 'norm')[1]
            results_pit[i, j] += p_value_pit
        results_no_pit[i, j] /= n_trials
        results_pit[i, j] /= n_trials
    
    for j, params in enumerate(params_gamma):
        for k in range(n_trials):
            data = stats.gamma.rvs(*params, size=n)
            p_value = perform_ks_test(data)
            results_no_pit[i, len(params_NIG) + j] += p_value
            transformed_data = stats.norm.ppf(stats.gamma.cdf(data, *params))
            p_value_pit = stats.kstest(transformed_data, 'norm')[1]
            results_pit[i, len(params_NIG) + j] += p_value_pit
        results_no_pit[i, len(params_NIG) + j] /= n_trials
        results_pit[i, len(params_NIG) + j] /= n_trials

# Tworzenie wykresów
plt.figure(figsize=(12, 6))

# Wykres bez PIT
plt.subplot(1, 2, 1)
for j, params in enumerate(params_NIG):
    plt.plot(num_samples, results_no_pit[:, j], label=f"NIG {params}")
for j, params in enumerate(params_gamma):
    plt.plot(num_samples, results_no_pit[:, len(params_NIG) + j], label=f"Gamma {params}")
plt.title('Wykres bez PIT')
plt.xlabel('Liczba danych')
plt.ylabel('Średni p-value')
plt.legend()

# Wykres z PIT
plt.subplot(1, 2, 2)
for j, params in enumerate(params_NIG):
    plt.plot(num_samples, results_pit[:, j], label=f"NIG {params}")
for j, params in enumerate(params_gamma):
    plt.plot(num_samples, results_pit[:, len(params_NIG) + j], label=f"Gamma {params}")
plt.title('Wykres z PIT')
plt.xlabel('Liczba danych')
plt.ylabel('Średni p-value')
plt.legend()

plt.tight_layout()
plt.show()
