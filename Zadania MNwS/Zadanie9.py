import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm, gamma

# Definicja funkcji do generowania danych z rozkładu NIG i Gamma
def generate_data(n, dist_type, params):
    if dist_type == 'NIG':
        # Parametry rozkładu NIG: mu, alpha, beta, delta
        mu, alpha, beta, delta = params
        # Generowanie danych z rozkładu NIG
        data = np.random.normal(mu, delta, n)
    elif dist_type == 'Gamma':
        # Parametry rozkładu Gamma: k (shape), theta (scale)
        k, theta = params
        # Generowanie danych z rozkładu Gamma
        data = np.random.gamma(k, theta, n)
    return data

# Definicja funkcji do przeprowadzenia testu Kołmogorowa
def perform_ks_test(data, dist_type, params):
    if dist_type == 'NIG':
        # Parametry rozkładu NIG: mu, alpha, beta, delta
        mu, alpha, beta, delta = params
        # Przeprowadzenie testu Kołmogorowa
        d, p = kstest(data, 'norm', args=(mu, delta))
    elif dist_type == 'Gamma':
        # Parametry rozkładu Gamma: k (shape), theta (scale)
        k, theta = params
        # Przeprowadzenie testu Kołmogorowa
        d, p = kstest(data, 'gamma', args=(k, theta))
    return d, p

# Generowanie danych
n = 1000  # liczba danych
dist_types = ['NIG', 'Gamma']  # typy rozkładów
params_values = [(0, 1, 0, 1), (2, 2)]  # różne wartości parametrów rozkładów
p_values = []  # lista do przechowywania wartości p

# Przeprowadzenie testu Kołmogorowa dla różnych wartości parametrów
for dist_type, params in zip(dist_types, params_values):
    data = generate_data(n, dist_type, params)
    d, p = perform_ks_test(data, dist_type, params)
    p_values.append(p)

# Wygenerowanie wykresu
plt.figure(figsize=(10, 6))
plt.plot(dist_types, p_values, marker='o')
plt.xlabel('Typ rozkładu')
plt.ylabel('Wartość p')
plt.title('Moc testu Kołmogorowa w zależności od typu rozkładu')
plt.grid(True)
plt.show()
