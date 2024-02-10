# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit import experimental_singleton as st_singleton
from streamlit.logger import get_logger
import yfinance as yf
import pandas as pd

LOGGER = get_logger(__name__)


# Seitentitel und Icon
page_title="Erste Schritte"
page_icon="👋"

# Importieren Sie Ihre Seitenmodule
from pages import FocusOneStock, LinearRegression

# Ein Wörterbuch, das die Seiten repräsentiert
pages = {
    "Seite 2": FocusOneStock,
    "Seite 3": LinearRegression,
}

# Preambel
st.markdown("Das ist ein normaler Text.")


stock_yfinance = st.text_input('Ticker Symbol:', 'AAPL')
annual_statements = st.selectbox(
     'Welchen Bericht möchtest du anzeigen lassen?',
     ('financials', 'balance_sheet', 'cashflow', 'income_stmt'))
st.write('Deine Auswahl:', annual_statements)


# Variablendefinition
#stock_yfinance = 'TSLA'
#annual_statements = 'income_stmt'


# Tickerobjekt erstellen
ticker_obj = yf.Ticker(stock_yfinance)

#yf spezifische Daten übergeben (Zeitraum, Ticker)
data = yf.download(stock_yfinance, start='2023-01-01', end='2023-12-31')

#Ausgabe der Schlusskurse über Zeitraum
st.write("\nAusgabe Überschriften & ersten 7 Datensätze:")
st.write(data.head(7)) # DataFrame testweise ausgeben (Überschriften & erste 7 Datensätze)

st.write("Ausgabe nur ersten 7 Schlusskurse:")
st.write(data['Close'].head(7)) # Nur Spalte Close aus DataFRame Data ausgeben

st.write("Ausgabe 7. Datensatz aus Close-Spalte:")
st.write(data['Close'].iloc[6]) # 7. Datensatz aus Spalte Close

st.write("Ausgabe Close-Datensatz vom Datum 03.07.2023:")
st.write(data['Close'].loc['2023-07-03']) # Datensatz Close mit konkreten Datum




# Variablendefinition für Finanzkennzahlen
financials = ticker_obj.financials
balance_sheet = ticker_obj.balance_sheet
cashflow = ticker_obj.cashflow
income_stmt = ticker_obj.income_stmt

#dynamische Konvertierung des Dropdown als String in ein Attribut für yfinance ticker_obj Abfrage
annual_data = getattr(ticker_obj, annual_statements, None)


#Ausgabe von Finanzkennzahlen unterschiedlicher Berichte
pd.set_option('display.max_columns', None)  # Zeigt alle Spalten an
pd.set_option('display.max_rows', None)  # Zeigt alle Zeilen an
st.write(f"Sie haben folgende Berichte ausgewählt {annual_statements} des Ticker Symbols {stock_yfinance} ausgewählt:")
st.write(annual_data)