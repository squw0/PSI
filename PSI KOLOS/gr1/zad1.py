import pandas as pd

# Zadanie 1: Wczytanie pliku data.csv
# Wczytanie danych
data = pd.read_csv('data.csv')

# Wyświetlenie pierwszych 5 wierszy
print(data.head())

# Wyświetlenie nazw kolumn
print(data.columns)

# Sprawdzenie liczby wierszy i kolumn
print(f'Liczba wierszy (próbek): {data.shape[0]}')
print(f'Liczba kolumn (cech): {data.shape[1]}')

# Sprawdzenie, czy są kolumny zbędne lub zawierające NaN
if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)
    print("Usunięto kolumnę 'id'")

if 'Unnamed: 32' in data.columns:
    data.drop('Unnamed: 32', axis=1, inplace=True)
    print("Usunięto kolumnę 'Unnamed: 32'")

# Sprawdzenie kolumn z brakującymi danymi
nan_columns = data.columns[data.isna().any()].tolist()
if nan_columns:
    print(f'Kolumny zawierające NaN: {nan_columns}')
    data.drop(columns=nan_columns, inplace=True)
else:
    print('Brak kolumn zawierających NaN.')
