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
data['year'] = data['okres'].dt.year
data['month'] = data['okres'].dt.month
data['day'] = data['okres'].dt.day
data = data.set_index('okres')
print(data.describe())
#zmiana przecinków na kropki w wartościach walut
for col in data.columns:
    if "1" in col:
        data[col]=data[col].str.replace(',','.')
        data[col]=data[col].astype(float)

# kolumna "1USD" jest przekształcana do analizy sezonowości, wykorzystując model mnożenia i częstotliwości 12,
# ponieważ mamy dane z 12 miesięcy.
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['1USD'], model='multiplicative', period=12)

#rysowanie wykresu dla kolumny "1USD", "1EUR"
data[['1USD', '1EUR']].plot()
# Dodanie opisów osi i legendy
plt.xlabel('Data')
plt.ylabel('Kurs')
plt.title('Kurs USD i EUR')
plt.show()


model = LinearRegression()
data['1USD_next'] = data['1USD'].shift(-1)
data['1EUR_next'] = data['1EUR'].shift(-1)
from sklearn.model_selection import train_test_split
data= data.dropna()
X = data[['1USD', '1EUR']].values
y = data[['1USD_next', '1EUR_next']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_new = X_test

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
print(y_pred)


import matplotlib.pyplot as plt

# Utworzenie wykresu
data = data.set_index('okres')
plt.plot(y_test, label='Rzeczywisty kurs')
plt.plot(y_pred, label='Przewidywany kurs')

# Dodanie opisów osi i legendy
plt.xlabel('Data')
plt.ylabel('Kurs')
plt.legend()

# Wyświetlenie wykresu
plt.show()




