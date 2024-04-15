import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, chisquare, t

def generuj_dane(rozklad, rozmiar_probki, df=None):
    if rozklad == 't':
        if df is None:
            raise ValueError("Stopnie swobody 'df' muszą być określone dla rozkładu t.")
        return t.rvs(df, size=rozmiar_probki)  
    elif rozklad == 'normal':
        return np.random.normal(size=rozmiar_probki)
    elif rozklad == 'uniform':
        return np.random.uniform(-1, 1, size=rozmiar_probki)
    else:
        raise ValueError("Nieprawidłowy rozkład. Wybierz spośród 't', 'normal', lub 'uniform'.")

def analiza_mocy(rozklad, rozmiary_probek, dfs=None, num_symulacji=1000):
    if rozklad == 't' and dfs is None:
        raise ValueError("Stopnie swobody 'df' muszą być określone dla rozkładu t.")
    
    moce = []

    if dfs is None:
        dfs = [None] * len(rozmiary_probek)  

    for rozmiar_probki, df in zip(rozmiary_probek, dfs):
        moc = 0

        for _ in range(num_symulacji):
            dane = generuj_dane(rozklad, rozmiar_probki, df=df)  
            p_wartosc = shapiro(dane)[1] if rozklad == 'normal' else kstest(dane, 'norm')[1]
            if p_wartosc > 0.05:
                moc += 1

        moce.append(moc / num_symulacji)

    return moce

# Parametry
rozmiary_probek = [20, 50, 100, 200]
dfs_dla_t = [3, 5, 10, 30]  # Stopnie swobody dla rozkładu t
num_symulacji = 1000

# Analiza mocy dla rozkładu t
moce_t_shapiro = [analiza_mocy('t', rozmiary_probek, [df]*len(rozmiary_probek), num_symulacji) for df in dfs_dla_t]
moce_t_kolmogorov = [analiza_mocy('t', rozmiary_probek, [df]*len(rozmiary_probek), num_symulacji) for df in dfs_dla_t]
moce_t_chi2 = [analiza_mocy('t', rozmiary_probek, [df]*len(rozmiary_probek), num_symulacji) for df in dfs_dla_t]

# Tworzenie wykresów dla każdego testu osobno
plt.figure(figsize=(18, 12))

# Test Shapiro-Wilka
plt.subplot(2, 2, 1)
for i, df in enumerate(dfs_dla_t):
    plt.plot(rozmiary_probek, moce_t_shapiro[i], label=f'df={df}')
plt.title('Moc testu Shapiro-Wilka (rozkład t)')
plt.xlabel('Rozmiar próbki')
plt.ylabel('Moc')
plt.legend(title='Stopnie swobody')
plt.grid(True)

# Test Kołmogorowa-Smirnowa
plt.subplot(2, 2, 2)
for i, df in enumerate(dfs_dla_t):
    plt.plot(rozmiary_probek, moce_t_kolmogorov[i], label=f'df={df}')
plt.title('Moc testu Kołmogorowa-Smirnowa (rozkład t)')
plt.xlabel('Rozmiar próbki')
plt.ylabel('Moc')
plt.legend(title='Stopnie swobody')
plt.grid(True)

# Test chi-kwadrat
plt.subplot(2, 2, 3)
for i, df in enumerate(dfs_dla_t):
    plt.plot(rozmiary_probek, moce_t_chi2[i], label=f'df={df}')
plt.title('Moc testu chi-kwadrat (rozkład t)')
plt.xlabel('Rozmiar próbki')
plt.ylabel('Moc')
plt.legend(title='Stopnie swobody')
plt.grid(True)

plt.tight_layout()
plt.show()
