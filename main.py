import flet as ft
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Funktion zur Erstellung des LSTM-Modells
def create_lstm_model(X_train, y_train, epochs=10, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Funktion zur Vorhersage
def predict_stock_price(ticker, start_date, end_date):
    # 1. Daten abrufen
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]

    # 2. Normalisieren der Daten
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 3. Train-Test-Daten erstellen
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # 4. LSTM-Modell erstellen
    model = create_lstm_model(X_train, y_train)

    # 5. Vorhersage für Testdaten
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform([y_test])

    return y_test[0], y_pred[:, 0], data

# Funktion zur Erstellung des Graphen
def create_plot(y_test, y_pred, data):
    plt.figure(figsize=(10, 6))

    # Plot für die tatsächlichen und vorhergesagten Kurse
    plt.plot(data.index[-len(y_test):], y_test, label='Tatsächliche Kurse')
    plt.plot(data.index[-len(y_pred):], y_pred, label='Vorhergesagte Kurse', linestyle='--')

    plt.title("Aktienkurs Vorhersage")
    plt.xlabel("Datum")
    plt.ylabel("Kurs")
    plt.legend()

    # Das Diagramm in ein Bild umwandeln
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64

# Flet-App mit Eingabefeldern und Vorhersage-Button
def main(page: ft.Page):
    page.title = "Kursvorhersagen mit LSTM"
    page.vertical_alignment = ft.MainAxisAlignment.START

    # Eingabefelder
    ticker_input = ft.TextField(label="Ticker (z.B. TSLA, AAPL)", autofocus=True)
    start_date_input = ft.TextField(label="Startdatum (YYYY-MM-DD)")
    end_date_input = ft.TextField(label="Enddatum (YYYY-MM-DD)")

    # Button für Vorhersage
    result_text = ft.Text("")
    predict_button = ft.ElevatedButton("Vorhersage erstellen", on_click=lambda e: on_predict_click(e, ticker_input, start_date_input, end_date_input, result_text, page))

    # Info-Button für Ticker-Liste
    info_button = ft.IconButton(ft.icons.INFO_OUTLINE, on_click=lambda e: show_ticker_info(page))

    page.add(
        ticker_input,
        start_date_input,
        end_date_input,
        predict_button,
        result_text,
        info_button
    )

# Funktion, die bei Klick auf "Vorhersage erstellen" ausgeführt wird
def on_predict_click(event, ticker_input, start_date_input, end_date_input, result_text, page):
    ticker = ticker_input.value
    start_date = start_date_input.value
    end_date = end_date_input.value

    try:
        y_test, y_pred, data = predict_stock_price(ticker, start_date, end_date)
        
        # Vorhersageergebnisse anzeigen
        result_text.value = f"Ergebnisse für {ticker}:\n"
        for real, pred in zip(y_test[:10], y_pred[:10]):  # Zeige nur die ersten 10 Werte
            result_text.value += f"Real: {real:.2f} - Vorhersage: {pred:.2f}\n"
        result_text.update()

        # Erstelle Graphen und zeige ihn an
        img_base64 = create_plot(y_test, y_pred, data)
        page.add(ft.Image(src=f"data:image/png;base64,{img_base64}"))
        page.update()

    except Exception as e:
        result_text.value = f"Fehler: {e}"
        result_text.update()

# Funktion, die beim Klick auf den Info-Button ausgeführt wird
def show_ticker_info(page: ft.Page):
    # Ticker aus einer Textdatei lesen
    ticker_list = []
    if os.path.exists('tickers.txt'):
        with open('tickers.txt', 'r') as file:
            ticker_list = file.readlines()

    page.dialog = ft.AlertDialog(
        title="Übersicht der Ticker-Symbole",
        content=ft.Column([ft.Text(ticker.strip()) for ticker in ticker_list]),
        actions=[ft.TextButton("Schließen", on_click=lambda e: page.dialog.close())]
    )
    page.dialog.open = True
    page.update()

# Flet-App ausführen
if __name__ == "__main__":
    ft.app(target=main)
