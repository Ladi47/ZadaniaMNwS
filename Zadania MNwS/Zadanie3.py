import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2

# Parametry symulacji
num_simulations = 1000
sample_sizes = [10, 50, 100, 500, 1000]
degrees_of_freedom = [1, 2, 3, 5, 10, 50]

# Przechowuje wyniki
results = {}

# Pętla przez różne rozmiary próbek i stopnie swobody
for df in degrees_of_freedom:
    for n in sample_sizes:
        p_values = []
        for _ in range(num_simulations):
            # Generowanie próbek z rozkładu chi-kwadrat
            sample1 = chi2.rvs(df, size=n)
            sample2 = chi2.rvs(df, size=n)
            
            # Przesunięcie próbek, aby miały tę samą wartość oczekiwaną
            sample1 = sample1 - np.mean(sample1) + df
            sample2 = sample2 - np.mean(sample2) + df
            
            # Przeprowadzenie testu Kołmogorowa-Smirnowa
            _, p_value = ks_2samp(sample1, sample2)
            p_values.append(p_value)
        
        # Obliczenie mocy testu
        power = np.mean([p < 0.05 for p in p_values])
        
        # Zapisanie wyników
        results[(df, n)] = power

# Wyświetlanie wyników na wykresie
for i, df in enumerate(degrees_of_freedom):
    powers = [results[(df, n)] for n in sample_sizes]
    if i == len(degrees_of_freedom) - 1:  # dla ostatniego stopnia swobody
        plt.plot(sample_sizes, powers, label=f'df={df}', color='lime')  # jaskrawozielony kolor
    else:
        plt.plot(sample_sizes, powers, label=f'df={df}')
plt.xlabel('Rozmiar próbki')
plt.ylabel('Moc testu')
plt.legend()
plt.show()
