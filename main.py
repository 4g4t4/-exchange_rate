# importowanie potrzebnych bibliotek

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons



# wczytanie danych z pliku csv
data = pd.read_csv('archiwum_tab_a_2022.csv', sep=';')
# zmiana typu kolumny 'data' na datę, bez tego jest traktowana jako liczba.
data['okres'] = pd.to_datetime(data['okres'], format='%Y%m%d')

data = data.set_index('okres')
print(data.describe())

# zmiana przecinków na kropki w wartościach walut
for col in data.columns:
    if "1" in col:
        data[col]=data[col].str.replace(',','.')
        data[col]=data[col].astype(float)

# kolumna "1USD" jest przekształcana do analizy sezonowości, wykorzystując model mnożenia i częstotliwości 12,
# ponieważ mamy dane z 12 miesięcy.
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['1USD'], model='multiplicative', period=12)

# rysowanie wykresu dla kolumny "1USD", "1EUR"
data[['1USD', '1EUR']].plot()
# dodanie opisów osi i legendy oraz rysowanie wykresu
plt.xlabel('Data')
plt.ylabel('Kurs')
plt.title('Kurs USD i EUR')
plt.show()


# utworzenie modelu
model = LinearRegression()
data['1USD_next'] = data['1USD'].shift(-1)
data['1EUR_next'] = data['1EUR'].shift(-1)

# przygotowanie danych do modelowania, podział na zestaw treningowy i testowy
from sklearn.model_selection import train_test_split
data= data.dropna()
X = data[['1USD', '1EUR']].values
y = data[['1USD_next', '1EUR_next']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# trenowanie modelu
model.fit(X_train, y_train)
# korzystanie z wytrenowanego modelu do dokonywania prognoz
y_pred = model.predict(X_test)

# import miary błędy średniokwadratowego z biblioteki sklearn
# oraz użycie jej do obliczenia błędu średniokwadratowego
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

# wyświetla jej wartość oraz prognozy
print("Mean Squared Error: ", mse)
print(y_pred)

# wykres
# przewidywane wartości
future_dates = pd.date_range(data.index[-1], periods=len(y_pred), freq='D')
plt.plot(future_dates, y_pred[:,0], label='Przewidywany kurs USD')
plt.plot(future_dates, y_pred[:,1], label='Przewidywany kurs EUR')

# dodanie opisów osi i legendy oraz rysowanie wykresu
plt.xlabel('Data')
plt.ylabel('Kurs')
plt.legend()
plt.show()











