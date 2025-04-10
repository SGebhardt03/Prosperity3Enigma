import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameter: Diese kannst du anpassen.
window = 80  # Fenstergröße für gleitenden Durchschnitt und Standardabweichung
rsi_period = 14  # Periode für den RSI

# CSV-Datei einlesen. Passe den Pfad bzw. Dateinamen ggf. an.
data = pd.read_csv('../data_25/Round1/prices_round_1_day_0.csv', sep=';')

data = data.query("product == 'SQUID_INK'")


# Setze das Datum als Index, falls sinnvoll
data.set_index('timestamp', inplace=True)

# Stelle sicher, dass die Preisspalte richtig benannt ist. Falls es einen anderen Namen gibt, passe es an.
price = data['mid_price']

# Berechnung des gleitenden Durchschnitts und der Standardabweichung
data['MA'] = price.rolling(window=window).mean()
data['RollingStd'] = price.rolling(window=window).std()

# Z-Score: Wie viele Standardabweichungen weicht der aktuelle Preis vom gleitenden Durchschnitt ab.
data['ZScore'] = (price - data['MA']) / data['RollingStd']

# Bollinger Bands berechnen:
# obere Band: MA + 2*Std
# untere Band: MA - 2*Std
data['UpperBand'] = data['MA'] + 2 * data['RollingStd']
data['LowerBand'] = data['MA'] - 2 * data['RollingStd']


# RSI-Berechnung
def compute_RSI(series, period=14):
    delta = series.diff()
    # Vermeide NaN in der ersten Zeile
    delta = delta.dropna()
    # Gains und Losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Erste Berechnung: einfacher Durchschnitt
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # RSI-Berechnung
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Füge NaN für die ersten Elemente hinzu, wo keine RSI-Berechnung möglich ist
    rsi = rsi.reindex(series.index)
    return rsi


data['RSI'] = compute_RSI(price, period=rsi_period)

# Ausgabe der Ergebnisse (optional)
print(data.tail(10))

# Visualisierung der Ergebnisse
plt.figure(figsize=(14, 8))

# Erster Plot: Price, MA und Bollinger Bands
plt.subplot(2, 1, 1)
plt.plot(data.index, price, label='Preis', color='black')
plt.plot(data.index, data['MA'], label=f'{window}-Perioden MA', color='blue', linestyle='--')
plt.plot(data.index, data['UpperBand'], label='Upper Bollinger Band', color='red', linestyle='--')
plt.plot(data.index, data['LowerBand'], label='Lower Bollinger Band', color='green', linestyle='--')
plt.fill_between(data.index, data['LowerBand'], data['UpperBand'], color='gray', alpha=0.2)
plt.title('Preis, MA & Bollinger Bands')
plt.legend()

# Zweiter Plot: Z-Score und RSI
plt.subplot(2, 1, 2)
plt.plot(data.index, data['ZScore'], label='Z-Score', color='purple')
plt.axhline(3, color='red', linestyle='--', label='Z=2')
plt.axhline(-3, color='green', linestyle='--', label='Z=-2')
plt.legend(loc='upper left')
plt.title('Z-Score')

# Falls du auch den RSI separat visualisieren möchtest:
plt.figure(figsize=(14, 4))
plt.plot(data.index, data['RSI'], label='RSI', color='orange')
plt.axhline(70, color='red', linestyle='--', label='Überkauft (70)')
plt.axhline(30, color='green', linestyle='--', label='Überverkauft (30)')
plt.title('Relative Strength Index')
plt.legend()

plt.show()
