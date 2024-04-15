import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, kstest, chisquare

def symuluj_testy_pit(liczba_probek, stopnie_swobody):
    stopy_odrzucenia_ks = np.zeros((len(liczba_probek), len(stopnie_swobody)))
    stopy_odrzucenia_chi2 = np.zeros((len(liczba_probek), len(stopnie_swobody)))

    for i, n_probek in enumerate(liczba_probek):
        for j, df in enumerate(stopnie_swobody):
            p_wartosci_ks = []
            p_wartosci_chi2 = []
            for _ in range(num_symulacji):
                dane = t.rvs(df, size=n_probek)
                dane_pit = t.cdf(dane, df)  # Transformacja PIT
                _, p_ks = kstest(dane_pit, 'uniform')
                # Podział danych na kategorie
                dane_pit_kategorie = np.histogram(dane_pit, bins=10)[0]
                _, p_chi2 = chisquare(dane_pit_kategorie)
                p_wartosci_ks.append(p_ks)
                p_wartosci_chi2.append(p_chi2)
            stopy_odrzucenia_ks[i, j] = np.mean(np.array(p_wartosci_ks) < 0.05)
            stopy_odrzucenia_chi2[i, j] = np.mean(np.array(p_wartosci_chi2) < 0.05)

    return stopy_odrzucenia_ks, stopy_odrzucenia_chi2

# Parametry
liczba_probek = [50, 100, 200, 500]
stopnie_swobody = [1, 2, 5, 10, 20]  

num_symulacji = 1000

stopy_odrzucenia_ks, stopy_odrzucenia_chi2 = symuluj_testy_pit(liczba_probek, stopnie_swobody)

# Tworzenie wykresów
plt.figure(figsize=(18, 6))

colors = plt.cm.tab10(np.linspace(0, 1, len(liczba_probek)))

for i, df in enumerate(stopnie_swobody):
    plt.subplot(1, len(stopnie_swobody), i+1)
    plt.title(f'Stopnie Swobody = {df} dla Testów PIT')
    plt.xlabel('Liczba Próbek')
    plt.ylabel('Stopa Odrzuceń')
    for j, n_probek in enumerate(liczba_probek):
        plt.plot(liczba_probek, stopy_odrzucenia_ks[:, j], label=f'Test KS (PIT), N = {n_probek}', color=colors[j], marker='o')
        plt.plot(liczba_probek, stopy_odrzucenia_chi2[:, j], label=f'Test Chi2 (PIT), N = {n_probek}', color=colors[j], linestyle='--')

# Przeniesienie legendy poza wykresy
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
