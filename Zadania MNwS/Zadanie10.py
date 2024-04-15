import numpy as np
from scipy.stats import chi2, t
import matplotlib.pyplot as plt

def generuj_dane_studenta(n, k):
    return np.random.standard_t(df=k, size=n)

def test_chi_kwadrat(data, bins, dof):
    obserwowane, _ = np.histogram(data, bins=bins)
    oczekiwane = np.full_like(obserwowane, fill_value=len(data) / bins)
    statystyka_chi2 = np.sum((obserwowane - oczekiwane)**2 / oczekiwane)
    wartosc_p = 1 - chi2.cdf(statystyka_chi2, df=dof)
    return wartosc_p

def symuluj_testy_chi_kwadrat(num_symulacji, n, k, liczba_klas, dof):
    wartosci_p = np.zeros(num_symulacji)
    for i in range(num_symulacji):
        dane = generuj_dane_studenta(n, k)
        wartosci_p[i] = test_chi_kwadrat(dane, liczba_klas, dof)
    return wartosci_p

def rysuj_wykres_wartosci_p(ax, wartosci_klas, avg_wartosci_p, n, k):
    ax.plot(wartosci_klas, avg_wartosci_p, label=f'n={n}, k={k}')
    ax.set_xlabel('Liczba klas')
    ax.set_ylabel('Średnia wartość P')
    ax.set_title('Średnia wartość P vs Liczba klas')
    ax.legend()

# Parametry
num_symulacji = 1000
wartosci_n = [50, 100, 200]
wartosci_k = [5, 10, 15]
wartosci_klas = np.arange(2, 11)

# Tworzenie subplotów
fig, ax = plt.subplots()

# Wykonanie symulacji i rysowanie wyników
for n in wartosci_n:
    for k in wartosci_k:
        avg_wartosci_p = []
        for liczba_klas in wartosci_klas:
            dof = liczba_klas - 1
            wartosci_p = symuluj_testy_chi_kwadrat(num_symulacji, n, k, liczba_klas, dof)
            avg_wartosci_p.append(np.mean(wartosci_p))
        rysuj_wykres_wartosci_p(ax, wartosci_klas, avg_wartosci_p, n, k)

plt.show()
