import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

veri = pd.read_csv(
    "C:\\Users\\Oytun\\Desktop\\GitHub Projeleri\\Yapay Zeka Ölüm Tahmini\\olumsayisi1.csv")

x = veri["gun"]
y = veri["olumsayisi"]

x = x.values.reshape(-1, 1)
y = y.values.reshape(-1, 1)

plt.scatter(x, y)
plt.show()

# Lineer Reg.
predictlineer = LinearRegression()
predictlineer.fit(x, y)
predictlineer.predict(x)

plt.plot(x, predictlineer.predict(x), c="red")

# Polinom Reg.
predcitpolinom = PolynomialFeatures(degree=3)
Xyeni = predcitpolinom.fit_transform(x)

polimodel = LinearRegression()
polimodel.fit(Xyeni, y)
polimodel.predict(Xyeni)

plt.plot(x, polimodel.predict(Xyeni))
plt.show()

hatakaresilineer = 0
hatakaresipolinom = 0

for i in range(len(Xyeni)):
    hatakaresipolinom = hatakaresipolinom + (int(y[i]) - int(polimodel.predict(Xyeni)[i])) ** 2

for i in range(len(y)):
    hatakaresilineer = hatakaresilineer + (int(y[i]) - int(predictlineer.predict(x)[i])) ** 2

hatakaresipolinom = 0

for a in range(150):

    predcitpolinom = PolynomialFeatures(degree=a + 1)
    Xyeni = predcitpolinom.fit_transform(x)

    polimodel = LinearRegression()
    polimodel.fit(Xyeni, y)
    polimodel.predict(Xyeni)
    for i in range(len(Xyeni)):
        hatakaresipolinom = hatakaresipolinom + (float(y[i]) - float(polimodel.predict(Xyeni)[i])) ** 2
    print(a + 1, "inci dereceden fonksiyonda hata,", hatakaresipolinom)

    hatakaresipolinom = 0

predcitpolinom8 = PolynomialFeatures(degree=11)
Xyeni = predcitpolinom8.fit_transform(x)

polimodel8 = LinearRegression()
polimodel8.fit(Xyeni, y)
polimodel8.predict(Xyeni)

plt.plot(x, polimodel8.predict(Xyeni))

plt.show()

print((int(y[69]) - int(polimodel8.predict(Xyeni)[69])))
