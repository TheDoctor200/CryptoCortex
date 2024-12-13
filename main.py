import flet as ft
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Funktion zur Vorhersage

def predict_stock_price(ticker, start_date, end_date):
    # 1. Daten abrufen
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("Keine Daten für den angegebenen Ticker oder Zeitraum gefunden.")

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

    # 4. Modell erstellen und trainieren
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Vorhersage für Testdaten
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    return y_test.flatten(), y_pred.flatten(), data

# Funktion zur Erstellung des Graphen
def create_plot(y_test, y_pred, data):
    plt.figure(figsize=(10, 6))

    # Plot für die tatsächlichen und vorhergesagten Kurse
    plt.plot(data.index[-len(y_test):], y_test, label='Tatsächliche Kurse', color='blue')
    plt.plot(data.index[-len(y_pred):], y_pred, label='Vorhergesagte Kurse', linestyle='--', color='orange')

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
    plt.close()
    return img_base64

# Flet-App mit Eingabefeldern und Vorhersage-Button
def main(page: ft.Page):
    page.title = "Kursvorhersagen mit Random Forest"
    page.scroll = ft.ScrollMode.AUTO

    # Eingabefelder
    ticker_input = ft.TextField(label="Ticker (z.B. TSLA, AAPL)", autofocus=True)
    start_date_input = ft.TextField(label="Startdatum (YYYY-MM-DD)")
    end_date_input = ft.TextField(label="Enddatum (YYYY-MM-DD)")

    # Ergebnisanzeige
    result_text = ft.Text("", size=14, multiline=True)
    image_container = ft.Container()

    # Button für Vorhersage
    def on_predict_click(e):
        ticker = ticker_input.value
        start_date = start_date_input.value
        end_date = end_date_input.value

        try:
            y_test, y_pred, data = predict_stock_price(ticker, start_date, end_date)

            # Vorhersageergebnisse anzeigen
            result_text.value = f"Ergebnisse für {ticker} (Zeige nur die ersten 10 Werte):\n"
            for real, pred in zip(y_test[:10], y_pred[:10]):
                result_text.value += f"Real: {real:.2f} - Vorhersage: {pred:.2f}\n"
            result_text.update()

            # Erstelle Graphen und zeige ihn an
            img_base64 = create_plot(y_test, y_pred, data)
            image_container.content = ft.Image(src=f"data:image/png;base64,{img_base64}", width=700)
            page.update()

        except Exception as ex:
            result_text.value = f"Fehler: {ex}"
            result_text.update()

    predict_button = ft.ElevatedButton("Vorhersage erstellen", on_click=on_predict_click)

    # Button für Info
    def show_ticker_info(e):
        ticker_list = []
        if os.path.exists('tickers.txt'):
            with open('tickers.txt', 'r') as file:
                ticker_list = file.readlines()

        page.dialog = ft.AlertDialog(
            title=ft.Text("Übersicht der Ticker-Symbole"),
            content=ft.Column([ft.Text(ticker.strip()) for ticker in ticker_list]),
            actions=[ft.TextButton("Schließen", on_click=lambda e: page.dialog.close())]
        )
        page.dialog.open = True
        page.update()

    info_button = ft.IconButton(ft.icons.INFO_OUTLINE, on_click=show_ticker_info)

    # Page Layout
    page.add(
        ft.Column([
            ft.Row([ticker_input]),
            ft.Row([start_date_input, end_date_input]),
            ft.Row([predict_button, info_button]),
            ft.Divider(),
            result_text,
            image_container
        ])
    )

# Flet-App ausführen
if __name__ == "__main__":
    ft.app(target=main)


