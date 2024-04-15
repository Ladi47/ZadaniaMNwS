import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, jarque_bera, shapiro
import statsmodels.api as sm

# Funkcja do generowania próbek z rozkładu t-Studenta
def generuj_próbki_t_studenta(liczba_danych, stopnie_swobody):
    return t.rvs(stopnie_swobody, size=liczba_danych)

# Funkcja do obliczania mocy testów dla danych próbek
def oblicz_moc_testów(liczba_danych, stopnie_swobody, liczba_symulacji=1000):
    odrzucenia_JB = 0
    odrzucenia_SW = 0
    odrzucenia_LF = 0
    
    for _ in range(liczba_symulacji):
        próbki = generuj_próbki_t_studenta(liczba_danych, stopnie_swobody)
        
        # Test Jarque-Bera
        statystyka_JB, p_wartość_JB = jarque_bera(próbki)
        if p_wartość_JB < 0.05:
            odrzucenia_JB += 1
        
        # Test Shapiro-Wilk
        _, p_wartość_SW = shapiro(próbki)
        if p_wartość_SW < 0.05:
            odrzucenia_SW += 1
        
        # Test Lilliefors
        _, p_wartość_LF = sm.stats.lilliefors(próbki)
        if p_wartość_LF < 0.05:
            odrzucenia_LF += 1
    
    moc_JB = odrzucenia_JB / liczba_symulacji
    moc_SW = odrzucenia_SW / liczba_symulacji
    moc_LF = odrzucenia_LF / liczba_symulacji
    
    return moc_JB, moc_SW, moc_LF

# Parametry
liczby_danych = [20, 50, 100, 200]
stopnie_swobody = [5, 10, 20, 30, 50]  # Dodaliśmy dwie dodatkowe wartości stopni swobody

# Obliczanie mocy testów dla różnych wielkości próbki i stopni swobody
moce_testów = np.zeros((len(liczby_danych), len(stopnie_swobody), 3))  # macierz do przechowywania mocy testów

for i, liczba_danych in enumerate(liczby_danych):
    for j, df in enumerate(stopnie_swobody):  # Zmieniamy nazwę zmiennej iterowanej w pętli
        moce_testów[i, j, :] = oblicz_moc_testów(liczba_danych, df)

# Wykresy
plt.figure(figsize=(12, 12))

# Wykres mocy testu Jarque-Bera
plt.subplot(3, 1, 1)
for j, df in enumerate(stopnie_swobody):
    plt.plot(liczby_danych, moce_testów[:, j, 0], label=f'df={df}')
plt.title('Moc testu Jarque-Bera')
plt.xlabel('Liczba danych')
plt.ylabel('Moc testu')
plt.legend()

# Wykres mocy testu Shapiro-Wilk
plt.subplot(3, 1, 2)
for j, df in enumerate(stopnie_swobody):
    plt.plot(liczby_danych, moce_testów[:, j, 1], label=f'df={df}')
plt.title('Moc testu Shapiro-Wilk')
plt.xlabel('Liczba danych')
plt.ylabel('Moc testu')
plt.legend()

# Wykres mocy testu Lilliefors
plt.subplot(3, 1, 3)
for j, df in enumerate(stopnie_swobody):
    plt.plot(liczby_danych, moce_testów[:, j, 2], label=f'df={df}')
plt.title('Moc testu Lilliefors')
plt.xlabel('Liczba danych')
plt.ylabel('Moc testu')
plt.legend()

plt.tight_layout()
plt.show()
