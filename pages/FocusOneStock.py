########################## FokusOneStock version 2
#import openai
import streamlit as st
import requests
import urllib.parse
import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import yfinance as yf
import scipy.stats
import pandas as pd
import numpy as np

from datetime import date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from matplotlib.dates import AutoDateLocator
from translate import Translator
from termcolor import colored
from bs4 import BeautifulSoup

########################################### separate Klasse als fix für .info
# Quelle: https://github.com/ranaroussi/yfinance/issues/1729
class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "quoteType,"
                         "assetProfile,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
########################################### Ende separate Klasse

def app():
    st.title('Seite 1')
    st.write('Willkommen auf Seite 1 - FokusOneStock.')



###########################VARIABLEN-DEFINITION START--------------------------------
# Parameter für KAUF Signal RSI
min_rsi = 35  # Minimum-RSI-Wert
days = 2      # Anzahl der Tage für den Trend (steigend)

# Parameter für KAUF Signal MACD
nulllinie = 0 # Wert der "Nulllinie"
days_macd = 2  # Anzahl der Tage, die der MACD über dem Signal bleiben soll
range_days = 5 # Zeitspanne, in der alle Signale stattfinden sollen

# Parameter für Verkaufssignale RSI
max_rsi_verkaufen = 70  # Maximaler RSI-Wert für Verkaufssignal
days_rsi_verkaufen = 2  # Anzahl der Tage für den Trend (fallend) bei RSI

# Parameter für Verkaufssignale MACD
nulllinie_verkaufen = 0  # Wert der "Nulllinie" für MACD Verkaufssignal
days_macd_verkaufen = 3  # Anzahl der Tage, die der MACD unter dem Signal bleiben soll
range_days_verkaufen = 5  # Zeitspanne, in der alle Verkaufssignale stattfinden sollen


# Anzahl der Nachkommastellen
n = 3

# den heutigen Tag abfragen und Format konvertieren
today = date.today()
today_str = today.strftime('%Y-%m-%d')

# Aktie auswählen
## Klasse 1 - im Depot
stock_yfinance = 'TSLA' #Manufacturer
#stock_yfinance = 'HFG.DE' #E-Commerce
#stock_yfinance = 'ZIP' #Plattform
#stock_yfinance = 'ARM' #Semiconductors
#stock_yfinance = 'ZAL.DE' #E-Commerce
#stock_yfinance = 'YOU.DE' #E-Commerce

## Klasse 2 - Watchlist

#stock_yfinance = 'TRMD' #Oil & Gas E&P
#stock_yfinance = 'ONTO' #Semiconductors
#stock_yfinance = 'IRT' #Real Estate
#stock_yfinance = 'KNSL' #Insurance
#stock_yfinance = ''




## Noch prüfen
#stock_yfinance = 'C3.AI' #ETR: 724
#stock_yfinance = 'Cloudflare' #FRA: 8CF
#stock_yfinance = 'Snowflake Inc.' #ETR: 5Q5
#stock_yfinance = 'DigitalOcean Holdings Inc' #FRA: 0SU
#stock_yfinance = 'DataDog' #ETR: 3QD
#stock_yfinance = 'Hims' #FRA: 82W

#------------------------
# Periode definieren
#data = yf.download(stock_yfinance, start='2016-07-29', end=today_str)
#today_str = "2024-01-29"
data = yf.download(stock_yfinance, start='2023-04-01', end=today_str)

# Ersten Handelstag ermitteln und in Variable schreiben
data_trade_date = yf.download(stock_yfinance, period="max")
first_trade_date = data_trade_date.index[0]

## 1. Daten von yfinance abrufen
ticker_info = yf.Ticker(stock_yfinance).fast_info
ticker_obj = yf.Ticker(stock_yfinance)

## 2. Daten von YFinance (die Klasse als bugfix) abrufen
yfinance_obj = YFinance(stock_yfinance) # aus Klasse FYinance
info2 = yfinance_obj.info # aus Klasse FYinance

# Die letzten n Nachrichtenartikel abrufen
news = ticker_obj.news
last_n_articles = news[:3]

###########################VARIABLEN-DEFINITION ENDE#################################




###########################DEFINITION VON FUNKTIONEN START###########################
## Teil 0 - Übersetzungsmodul von openAI_____________________________________
def translate_to_german(text):
    translator = Translator(to_lang="de")
    translated_text = ""

    # Teile den Text in 500-Zeichen-Blöcke
    for i in range(0, len(text), 500):
        text_block = text[i:i+500]
        translated_block = translator.translate(text_block)
        translated_text += translated_block #hier muss ein Leerzeichen eingefügt werden

    return translated_text
#_____________________________________________________________________________


##Scoringmodell nach OKR (ambitionierte Score +10%)_________________________________
def calculate_score_rohmarge(ergebnis_20, sollwert_20, threshold_high_20, threshold_low_20):
    if ergebnis_20 > threshold_high_20:
        return 5
    elif ergebnis_20 > sollwert_20:
        return 4
    elif ergebnis_20 == sollwert_20:
        return 3
    elif ergebnis_20 >= threshold_low_20:
        return 2
    else:
        return 1

def calculate_score_gewinnmarge(ergebnis_21, sollwert_21, threshold_high_21, threshold_low_21):
    if ergebnis_21 > threshold_high_21:
        return 5
    elif ergebnis_21 > sollwert_21:
        return 4
    elif ergebnis_21 == sollwert_21:
        return 3
    elif ergebnis_21 >= threshold_low_21:
        return 2
    else:
        return 1

def calculate_score_kapitaleffizienz(ergebnis_3, sollwert_3, threshold_high_3, threshold_low_3):
    if ergebnis_3 > threshold_high_3:
        return 5
    elif ergebnis_3 > sollwert_3:
        return 4
    elif ergebnis_3 == sollwert_3:
        return 3
    elif ergebnis_3 >= threshold_low_3:
        return 2
    else:
        return 1

def calculate_score_entwicklung_stock_ipo(ergebnis_4, sollwert_4, threshold_high_4, threshold_low_4):
    if ergebnis_4 > threshold_high_4:
        return 5
    elif ergebnis_4 > sollwert_4:
        return 4
    elif ergebnis_4 == sollwert_4:
        return 3
    elif ergebnis_4 >= threshold_low_4:
        return 2
    else:
        return 1

def calculate_score_umsatzwachstum(ergebnis_50, sollwert_50, threshold_high_50, threshold_low_50):
    if ergebnis_50 > threshold_high_50:
        return 5
    elif ergebnis_50 > sollwert_50:
        return 4
    elif ergebnis_50 == sollwert_50:
        return 3
    elif ergebnis_50 >= threshold_low_50:
        return 2
    else:
        return 1

def calculate_score_eps_wachstum(ergebnis_51, sollwert_51, threshold_high_51, threshold_low_51):
    if ergebnis_51 > threshold_high_51:
        return 5
    elif ergebnis_51 > sollwert_51:
        return 4
    elif ergebnis_51 == sollwert_51:
        return 3
    elif ergebnis_51 >= threshold_low_51:
        return 2
    else:
        return 1

def calculate_score_eps_vergleich(ergebnis_52, sollwert_52, threshold_high_52, threshold_low_52):
    if ergebnis_52 > threshold_high_52:
        return 5
    elif ergebnis_52 > sollwert_52:
        return 4
    elif ergebnis_52 == sollwert_52:
        return 3
    elif ergebnis_52 >= threshold_low_52:
        return 2
    else:
        return 1

def calculate_score_peter_lynch_fair_value(ergebnis_6, sollwert_6, threshold_high_6, threshold_low_6):
    if ergebnis_6 > threshold_high_6:
        return 5
    elif ergebnis_6 > sollwert_6:
        return 4
    elif ergebnis_6 == sollwert_6:
        return 3
    elif ergebnis_6 >= threshold_low_6:
        return 2
    else:
        return 1

#comment_1 = "Geschäftsmodell"
#comment_20 = "Rohmarge (%)"
#comment_21 = "Gewinnmarge (%)"
#comment_3 = "Kapitaleffizienz (%)"
#comment_4 = "CAGR (%) stock seit IPO"
#comment_50 = "Umsatzwachstum (CAGR %)"
#comment_51 = "EPSWachstum (CAGR %)"
#comment_52 = "EPS LT Growth Est (3-5yr %)"
#comment_6 = "Ziel: Ergebnis > Sollwert"






########################################VARIABLEN-DEFINITION ENDE##############################



## 1. Daten für das Diagramm abrufen und zusammenstellen
# Calculate RSI
delta = data['Close'].diff()
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0
average_gain = up.rolling(window=14).mean()
average_loss = abs(down.rolling(window=14).mean())
rs = average_gain / average_loss
data['RSI'] = 100 - (100 / (1 + rs))


# Schlusskurs für Diagramm und Intro-Text über Diagramm
# Den jüngsten Aktienkurs und RSI ausgeben
latest_date = data.index[-1]
latest_close_price = data['Close'].iloc[-1]
latest_rsi = data['RSI'].iloc[-1]

# Calculate EMA 50 and EMA 200 (exponential moving average); rates young datapoint higher
data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

# Calculate MACD aus 1) Differenzlinie aus 12- & 26-Tage EMA + Signal aus 9-Tage EMA
# MACD und Signal berechnen
short_ema = data['Close'].ewm(span=12, adjust=False).mean()
long_ema = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = short_ema - long_ema
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()



############ Initialisierung der Kaufsignaltabellen
rsi_kaufsignale = pd.Series(index=data.index, dtype='object')
macd_kaufsignale = pd.Series(index=data.index, dtype='object')

### RSI - Überprüfe für jeden Tag, ob Kaufsignal vorliegt
for i in range(len(data) - days + 1):
    if all(data['RSI'][i:i+days] <= min_rsi) and data['RSI'][i+days-1] > data['RSI'][i]:
        rsi_kaufsignale[i+days-1] = 'KAUF'

### MACD - Überprüfe für jeden Tag, ob Kaufsignal vorliegt
for i in range(1, len(data) - days_macd + 1):
    if data['MACD'][i-1] < data['Signal'][i-1] and data['MACD'][i] > data['Signal'][i] \
       and all(data['MACD'][i:i+days_macd] > data['Signal'][i:i+days_macd]) \
       and all(data['MACD'][i-1:i+days_macd] < nulllinie):
        macd_kaufsignale[i+days_macd-1] = 'KAUF'
        
############ befüllen der beiden Tabellen Kaufsignale
rsi_kaufsignale_df = pd.DataFrame({'RSI-Signal': rsi_kaufsignale[rsi_kaufsignale == 'KAUF']})
rsi_kaufsignale_df.index = pd.to_datetime(rsi_kaufsignale_df.index)

macd_kaufsignale_df = pd.DataFrame({'MACD-Signal': macd_kaufsignale[macd_kaufsignale == 'KAUF']})
macd_kaufsignale_df.index = pd.to_datetime(macd_kaufsignale_df.index)



# Finde übereinstimmende Kaufsignale innerhalb der Range
matched_signals = []
for rsi_date in rsi_kaufsignale_df.index:
    for macd_date in macd_kaufsignale_df.index:
        # Berechnen Sie die Differenz in Tagen zwischen den Daten
        date_diff = abs((rsi_date - macd_date).days)
        if date_diff <= range_days:
            matched_signals.append(rsi_date)
            matched_signals.append(macd_date)


# Erstelle den finalen DataFrame mit übereinstimmenden Signalen und merge zu einem einzigen DataFrame
matched_signals_index = pd.to_datetime(matched_signals)
matched_signals_df = pd.DataFrame(index=matched_signals_index.unique()).sort_index()
matched_signals_df = matched_signals_df.merge(rsi_kaufsignale_df, left_index=True, right_index=True, how='left')
matched_signals_df = matched_signals_df.merge(macd_kaufsignale_df, left_index=True, right_index=True, how='left')
matched_signals_df = matched_signals_df.merge(data[['RSI', 'MACD', 'Signal']], left_index=True, right_index=True, how='left')

# Füge die Schlusskurse zum DataFrame hinzu - KAUFSIGNAL
close_kaufdatum = yf.download(stock_yfinance, start=matched_signals_df.index.min(), end=matched_signals_df.index.max())
matched_signals_df.index = matched_signals_df.index.strftime('%d.%m.%Y') #Datumsformat anpassen
matched_signals_df['Close'] = matched_signals_df.apply(lambda row: close_kaufdatum.loc[pd.to_datetime(row.name, format='%d.%m.%Y'), 'Close'] if pd.to_datetime(row.name, format='%d.%m.%Y') in close_kaufdatum.index else None, axis=1)
matched_signals_df = matched_signals_df.round(2)



############ Initialisierung der Verkaufsignaltabellen
#rsi_verkaufsignale = pd.Series(index=data.index, dtype='object')
#macd_verkaufsignale = pd.Series(index=data.index, dtype='object')

### RSI - Überprüfe für jeden Tag, ob Verkaufsignal vorliegt
#for i in range(len(data) - days_rsi_verkaufen + 1):
#    if all(data['RSI'][i:i+days_rsi_verkaufen] >= max_rsi_verkaufen) and data['RSI'][i+days_rsi_verkaufen-1] < data['RSI'][i]:
#        rsi_verkaufsignale[i+days_rsi_verkaufen-1] = 'VERKAUF'

### MACD - Überprüfe für jeden Tag, ob Verkaufsignal vorliegt
#for i in range(1, len(data) - days_macd_verkaufen + 1):
#    if data['MACD'][i-1] > data['Signal'][i-1] and data['MACD'][i] < data['Signal'][i] \
#       and all(data['MACD'][i:i+days_macd_verkaufen] < data['Signal'][i:i+days_macd_verkaufen]) \
#       and all(data['MACD'][i-1:i+days_macd_verkaufen] > nulllinie_verkaufen):
#        macd_verkaufsignale[i+days_macd_verkaufen-1] = 'VERKAUF'

### Befüllen der Tabellen mit Verkaufsignalen
#rsi_verkaufsignale_df = pd.DataFrame({'RSI-Signal': rsi_verkaufsignale[rsi_verkaufsignale == 'VERKAUF']})
#rsi_verkaufsignale_df.index = pd.to_datetime(rsi_verkaufsignale_df.index)

#macd_verkaufsignale_df = pd.DataFrame({'MACD-Signal': macd_verkaufsignale[macd_verkaufsignale == 'VERKAUF']})
#macd_verkaufsignale_df.index = pd.to_datetime(macd_verkaufsignale_df.index)


# Finde übereinstimmende Verkaufssignale innerhalb der Range
#matched_verkauf_signals = []
#for rsi_date in rsi_verkaufsignale_df.index:
#    for macd_date in macd_verkaufsignale_df.index:
#        date_diff = abs((rsi_date - macd_date).days)
#        if date_diff <= range_days_verkaufen:
#            matched_verkauf_signals.append(rsi_date)
#            matched_verkauf_signals.append(macd_date)

# Erstelle den finalen DataFrame mit übereinstimmenden Verkaufsignalen
#matched_verkauf_signals_index = pd.to_datetime(matched_verkauf_signals)
#matched_verkauf_signals_df = pd.DataFrame(index=matched_verkauf_signals_index.unique()).sort_index()
#matched_verkauf_signals_df = matched_verkauf_signals_df.merge(rsi_verkaufsignale_df, left_index=True, right_index=True, how='left')
#matched_verkauf_signals_df = matched_verkauf_signals_df.merge(macd_verkaufsignale_df, left_index=True, right_index=True, how='left')
#matched_verkauf_signals_df = matched_verkauf_signals_df.merge(data[['RSI', 'MACD', 'Signal']], left_index=True, right_index=True, how='left')

# Füge die Schlusskurse zum DataFrame hinzu - VERKAUFSIGNAL
#close_verkaufsdatum = yf.download(stock_yfinance, start=matched_verkauf_signals_df.index.min(), end=matched_verkauf_signals_df.index.max())
#matched_verkauf_signals_df.index = matched_verkauf_signals_df.index.strftime('%d.%m.%Y') #Datumsformat anpassen
#matched_verkauf_signals_df['Close'] = matched_verkauf_signals_df.apply(lambda row: close_verkaufsdatum.loc[pd.to_datetime(row.name, format='%d.%m.%Y'), 'Close'] if pd.to_datetime(row.name, format='%d.%m.%Y') in close_verkaufsdatum.index else None, axis=1)
#matched_verkauf_signals_df = matched_verkauf_signals_df.round(2)



## TEIL DIGRAMM__________________________________________________
## DIAGRAMM und TEXT
# Plotting Fig 1 - PLOT 1
fig = plt.figure(figsize=(15, 12)) # Figur definieren
gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3]) # Grid definieren, um Höhe anzupassen



# Plot price in subplot 1 & EMA50 and EMA200 curves
ax1 = fig.add_subplot(gs[0])  # 2 Reihen, 1 Spalte, 1. Subplot

ax1.plot(data.index, data['Close'], label=f'Price (Latest: {latest_close_price:.2f})', color='blue')
ax1.plot(data.index, data['EMA50'], label='EMA 50', color='purple', linestyle='dotted')
ax1.plot(data.index, data['EMA200'], label='EMA 200', color='#DAA520', linestyle='dotted')
ax1.set_title('closing Price with RSI')
ax1.set_ylabel('Price in local currency')
ax1.legend(loc='upper left')

# Plot RSI in subplot 1
ax1b = ax1.twinx()
ax1b.plot(data.index, data['RSI'], label=f'RSI (Latest: {latest_rsi:.2f})', color='purple')
ax1b.set_ylabel('RSI')
ax1b.legend(loc='upper right')



# Plotting Fig 1 - PLOT 2
ax2 = fig.add_subplot(gs[1])  # 2 Reihen, 1 Spalte, 2. Subplot

# Plot MACD & Signal curve
ax2.plot(data.index, data['MACD'], label='MACD', color='red', linestyle='dotted')
ax2.plot(data.index, data['Signal'], label='Signal', color='green', linestyle='dotted')
ax2.axhline(0, color='orange', linestyle='--', label='Nulllinie')
ax2.set_title(f"MACD und Signal für {stock_yfinance}")
ax2.set_ylabel("MACD/Signal")
ax2.legend(loc='upper left')


## TEIL UWE'S FINANCE TABLE_QUARTALBERICHTE_________________________________
# Variablen für yfinance definieren
quarterly_financials = ticker_obj.quarterly_financials
quarterly_balance = ticker_obj.quarterly_balance_sheet
quarterly_cashflow = ticker_obj.quarterly_cashflow

# Überprüfen, ob Quartalsdaten verfügbar sind
if not quarterly_financials.empty:
    available_quarters = quarterly_financials.shape[1]
    num_quarters = available_quarters
    
    # Kennzahlen für Quartale abrufen
    for i in range(num_quarters):


        # DataFrame initialisieren und Überschriften setzen
        kennzahlen = pd.DataFrame(columns=['Umsatz (bln)', 'Rohmarge (%)', 'EBIT (bln)', 'Nettoergebnis (bln)', 'FCF (bln)', 'D/E ratio', 'Verschuldung (bln)', 'Eigenkapital (bln)', 'Inventar (bln)', 'anderes Inventar (bln)'])
                       
        ## Kennzahlen abrufen
        for i in range(num_quarters):
            # Fiscal Quartal
            fiscal_quarter = quarterly_financials.columns[i].strftime('%d.%m.%y')

            # Umsatz
            total_revenue = quarterly_financials.loc['Total Revenue'][i] / 1e9 if 'Total Revenue' in quarterly_financials.index else np.nan

            # Rohmarge in %
            gross_profit = quarterly_financials.loc['Gross Profit'][i] / 1e9 if 'Gross Profit' in quarterly_financials.index else np.nan
            rohmarge = (gross_profit / total_revenue) * 100 if total_revenue else np.nan

            # EBIT
            ebit = quarterly_financials.loc['EBIT'][i] / 1e9 if 'EBIT' in quarterly_financials.index else np.nan

            # Nettoergebnis
            net_income = quarterly_financials.loc['Net Income'][i] / 1e9 if 'Net Income' in quarterly_financials.index else np.nan

            # Free Cash Flow
            fcf = (quarterly_cashflow.loc['Operating Cash Flow'][i] - quarterly_cashflow.loc['Capital Expenditure'][i]) / 1e9 if 'Operating Cash Flow' in quarterly_cashflow.index and 'Capital Expenditure' in quarterly_cashflow.index else np.nan

            # Verschuldungsgrad nach D/E
            total_debt = quarterly_balance.loc['Total Debt'][i] / 1e9 if 'Total Debt' in quarterly_balance.index else np.nan
            total_equity = quarterly_balance.loc['Stockholders Equity'][i] / 1e9 if 'Stockholders Equity' in quarterly_balance.index else np.nan
            de_ratio = total_debt / total_equity if total_equity else np.nan

            # Debt (Schulden) abrufen
            total_debt = quarterly_balance.loc['Total Debt'][i] / 1e9 if 'Total Debt' in quarterly_balance.index else np.nan

            # Stockholders Equity (Eigenkapital) abrufen
            total_equity = quarterly_balance.loc['Stockholders Equity'][i] / 1e9 if 'Stockholders Equity' in quarterly_balance.index else np.nan
            
            # Inventar abrufen
            inventory = quarterly_balance.loc['Inventories'][i] / 1e9 if 'Inventories' in quarterly_balance.index else np.nan
            
            # anderes Inventar
            inventory_other = quarterly_balance.loc['Other Inventories'][i] / 1e9 if 'Other Inventories' in quarterly_balance.index else np.nan
            
            # summiertes Inventar

            # Daten dem DataFrame hinzufügen
            kennzahlen.loc[fiscal_quarter] = [total_revenue, rohmarge, ebit, net_income, fcf, de_ratio, total_debt, total_equity, inventory, inventory_other]                        
        ## Daten transponieren
        kennzahlen = kennzahlen.transpose() # transponieren
        kennzahlen = kennzahlen.iloc[:, ::-1] # Spalten umkehren
        kennzahlen = kennzahlen.round(n) # alle Zahlen auf n Nachkommastellen runden

else:
    st.write("Keine Quartalsdaten verfügbar.")
# Tabelle ausgeben                          
## am Ende transponieren_____________________________________________________








## TEIL UWE'S FINANCE TABLE_JAHRESBERICHTE_________________________________
# Variablen für yfinance definieren
annual_financials = ticker_obj.financials
annual_balance = ticker_obj.balance_sheet
annual_cashflow = ticker_obj.cashflow
annual_income = ticker_obj.income_stmt

# Ermitteln der verfügbaren Jahre in jedem DataFrame
#years_financials = set(annual_financials.columns.year)
#years_balance = set(annual_balance.columns.year)
#years_income = set(annual_income.columns.year)
#years_cashflow = set(annual_cashflow.columns.year)

# Kleinsten gemeinsammen Nenner finden (intersection)
#common_years = years_financials.intersection(years_balance, years_income, years_cashflow)

# Überprüfen, ob Jahresdaten verfügbar sind
if not annual_financials.empty:
    available_years = annual_financials.shape[1]
    num_years = available_years

    # Kennzahlen für Jahresberichte abrufen
    for i in range(num_years):

        # Überprüfen, ob der Index i innerhalb der Grenzen der DataFrame liegt
        #if i < len(annual_financials.columns) and i < len(annual_balance.columns) and i < len(annual_cashflow.columns) and i < len(annual_income.columns):
            # DataFrame initialisieren und Überschriften setzen
        kennzahlen_jahresberichte = pd.DataFrame(columns=['Umsatz (bln)', 'Rohmarge (%)', 'EBITDA (bln)', 'EBIT (bln)', 'Nettoergebnis (bln)', 'FCF (bln)', 'D/E ratio', 'Verschuldung (bln)', 'Eigenkapital (bln)', 'Inventar (bln)', 'anderes Inventar (bln)', 'Vertrieb & Marketing', 'Verwaltung'])

        for i in range(num_years):
        
            ## Kennzahlen für Jahresberichte abrufen
            # Fiscal Jahr
            fiscal_year = annual_financials.columns[i].strftime('%Y')

                # Umsatz
            total_revenue = annual_financials.loc['Total Revenue'][i] / 1e9 if 'Total Revenue' in annual_financials.index else np.nan

                # Rohmarge in %
            gross_profit = annual_financials.loc['Gross Profit'][i] / 1e9 if 'Gross Profit' in annual_financials.index else np.nan
            rohmarge = (gross_profit / total_revenue) * 100 if total_revenue else np.nan

                # EBITDA
            ebitda = annual_financials.loc['EBITDA'][i] / 1e9 if 'EBITDA' in annual_financials.index else np.nan

                # EBIT
            ebit = annual_financials.loc['EBIT'][i] / 1e9 if 'EBIT' in annual_financials.index else np.nan

                # Nettoergebnis
            net_income = annual_financials.loc['Net Income'][i] / 1e9 if 'Net Income' in annual_financials.index else np.nan

                # Free Cash Flow
            fcf = (annual_cashflow.loc['Operating Cash Flow'][i] - annual_cashflow.loc['Capital Expenditure'][i]) / 1e9 if 'Operating Cash Flow' in annual_cashflow.index and 'Capital Expenditure' in annual_cashflow.index else np.nan

                # Verschuldungsgrad nach D/E
            total_debt = annual_balance.loc['Total Debt'][i] / 1e9 if 'Total Debt' in annual_balance.index else np.nan
            total_equity = annual_balance.loc['Stockholders Equity'][i] / 1e9 if 'Stockholders Equity' in annual_balance.index else np.nan
            de_ratio = total_debt / total_equity if total_equity else np.nan

                # Inventar abrufen
            inventory = annual_balance.loc['Inventories'][i] / 1e9 if 'Inventories' in annual_balance.index else np.nan

                # anderes Inventar
            inventory_other = annual_balance.loc['Other Inventories'][i] / 1e9 if 'Other Inventories' in annual_balance.index else np.nan

                # Verkauf und Marketing Ausgaben
            selling_and_marketing = annual_income.loc['Selling And Marketing Expense'][i] / 1e9 if 'Selling And Marketing Expense' in annual_income.index else np.nan

                # Verwaltungsausgaben
            general_and_administration = annual_income.loc['General And Administrative Expense'][i] / 1e9 if 'General And Administrative Expense' in annual_income.index else np.nan

                # Daten dem DataFrame hinzufügen
            kennzahlen_jahresberichte.loc[fiscal_year] = [total_revenue, rohmarge, ebitda, ebit, net_income, fcf, de_ratio, total_debt, total_equity, inventory, inventory_other, selling_and_marketing, general_and_administration]

        ## Daten transponieren
        kennzahlen_jahresberichte = kennzahlen_jahresberichte.transpose()
        kennzahlen_jahresberichte = kennzahlen_jahresberichte.iloc[:, ::-1] # Spalten umkehren
        kennzahlen_jahresberichte = kennzahlen_jahresberichte.round(n) # alle Zahlen auf n Nachkommastellen runden                        
else:
    st.write("Keine Jahresdaten verfügbar.")
######################################################################################




###############################___________________________Variablendefintion der Details
# Unternehmensinformationen abrufen
text_to_translate = info2['longBusinessSummary']
translated_company_info = translate_to_german(text_to_translate)
translated_company_info = textwrap.fill(translated_company_info, width=120)

# Marktkapitalisierung raussuchen
market_cap = ticker_info['marketCap'] if 'marketCap' in ticker_info else 0
market_cap = market_cap / 1e9

# aktuelles currency aus info abrufen
currency = ticker_info['currency']

#KGV (letzte 12 Monate) aus info2 holen und als integer umrechnen
t_kgv = info2.get('trailingPE', np.nan)
if not pd.isna(t_kgv) and not math.isinf(float(t_kgv)):
    t_kgv = int(round(float(t_kgv)))
else:
    t_kgv = np.nan

#KGV (zukünftige 12 Monate) aus info2 holen und als integer umrechnen
f_kgv = info2.get('forwardPE', np.nan)
if not pd.isna(f_kgv) and not math.isinf(float(f_kgv)):
    f_kgv = int(round(float(f_kgv)))
else:
    f_kgv = np.nan

#Sektor und Industrie aus info2 holen
sector = info2.get('sector', np.nan) 
industry = info2.get('industry', np.nan)

#######################################___________________________Ausgabe der Details


## Unternehmensbeschreibung in Deutsch
st.write("SCHRITT 1 - Was ist das Geschäftsmodell (Deutsch) und allgemeine Informationen:")
st.write()
st.write(translated_company_info)
st.write()

# Calculate the correlation and the p-value
data = data.dropna() # Remove missing values
correlation, p_value = scipy.stats.pearsonr(data['Close'].dropna(), data['RSI'].dropna())
st.write(f"correlation between stock price & RSI: {correlation:.5f}")
st.write(f"p-value: {p_value:.20f}")
st.write()
st.write(f"Erstes Handelsdatum von {stock_yfinance}: {first_trade_date}")
st.write(f"Sektor: {sector}")
st.write(f"Industrie: {industry}")

try:
    latest_fiscal_quarter_end = quarterly_financials.columns.max()
    new_date = latest_fiscal_quarter_end + relativedelta(months=3)
    formatted_new_date = new_date.strftime('%d.%m.%y')
    st.write(f"Nächstes Quartalsende: {new_date.strftime('%d.%m.%Y')}")
    fr_period_start = new_date + relativedelta(days=26)
    fr_period_end = new_date + relativedelta(days=26+14)
    st.write(f"Zeitfenster Financial Release: {fr_period_start.strftime('%d.%m.%Y')} bis {fr_period_end.strftime('%d.%m.%Y')}")
except Exception as e:
    st.write(f"Fehler: {e}")

st.write()
st.write(f"RSI (14): {latest_rsi:.2f}")
st.write(f"MarketCap (bln): {market_cap:.2f}")



################################################################################
## SCHRITT 2 - PROFITABILITÄT
# gross margin > 40%
# profit margin > 10%
st.write("\nSCHRITT 2 - Profitabilität (Verdient das Unternehmen Geld?):")
st.write("Rohmarge > 40%")
st.write("Gewinnmarge > 10%")

# Überprüfen, ob Jahresdaten verfügbar sind
if not annual_financials.empty:
    available_years = annual_financials.shape[1]
    num_years = available_years

    # Kennzahlen für Profitabilität abrufen
    for i in range(num_years):

        # DataFrame initialisieren und Überschriften setzen
        kennzahlen_profitabilitaet = pd.DataFrame(columns=['Rohmarge (%)', 'Gewinnmarge (%)'])

        ## Kennzahlen für Profitabilität abrufen
        for i in range(num_years):
            # Fiscal Jahr
            fiscal_year = annual_financials.columns[i].strftime('%Y')

            # Rohmarge in % (Rohertrag / Revenue)
            gross_profit = annual_financials.loc['Gross Profit'][i] / 1e9 if 'Gross Profit' in annual_financials.index else np.nan
            total_revenue = annual_financials.loc['Total Revenue'][i] / 1e9 if 'Total Revenue' in annual_financials.index else np.nan
            gross_margin = (gross_profit / total_revenue) * 100 if total_revenue else np.nan

            # Gewinnmarge in % (Net Income / Revenue)
            net_income = annual_financials.loc['Net Income'][i] / 1e9 if 'Net Income' in annual_financials.index else np.nan
            #total_revenue bereits als Parameter vorhanden
            profit_margin = (net_income / total_revenue) * 100 if total_revenue else np.nan

            # Daten dem DataFrame hinzufügen
            kennzahlen_profitabilitaet.loc[fiscal_year] = [gross_margin, profit_margin]

        ## Daten transponieren
        kennzahlen_profitabilitaet = kennzahlen_profitabilitaet.transpose()
        kennzahlen_profitabilitaet = kennzahlen_profitabilitaet.iloc[:, ::-1] # Spalten umkehren
        kennzahlen_profitabilitaet = kennzahlen_profitabilitaet.round(n) # alle Zahlen auf n Nachkommastellen runden                         

else:
    st.write("Keine Daten zur Profitabilität verfügbar.")
st.write(kennzahlen_profitabilitaet)
st.write()
######################################################################################




#####################################################################################
## STEP 3 - Kapitalzuweisung (wichtigste Aufgabe des Managements)
#Return On Invested Capital (ROIC) > 15%
st.write("\nSCHRITT 3 - Kapitalzuweisung/ Kapitaleffizienz (wichtigste Aufgabe des Managements):")
st.write("Rendite des investierten Kapitals (ROIC) > 15%")

# Überprüfen, ob Jahresdaten verfügbar sind
if not annual_balance.empty:
    available_years = annual_balance.shape[1]
    num_years = available_years

    # Kennzahlen für Kapitaleffizienz abrufen
    for i in range(num_years):

        # DataFrame initialisieren und Überschriften setzen
        kennzahlen_kapitaleffizienz = pd.DataFrame(columns=['Net Operating Profit After Tax (bln)', 'Steuerrate (%)', 'Kurzfristige Schulden (bln)', 'Langfristige Schulden (bln)', 'Eigenkapital d. Aktionären (bln)', 'Barmittel (bln)', 'Goodwill (bln)', 'ROIC (%)'])

        ## Kennzahlen für Kapitaleffizienz abrufen
        for i in range(num_years):
            # Fiscal Jahr
            fiscal_year = annual_balance.columns[i].strftime('%Y')
            #########ROIC = EBIT * (1 - Tax Rate) / Invested Capital
            ###NOPAT = EBIT * (1 - Tax Rate)
            
            #EBIT
            ebit_roic = annual_financials.loc['EBIT'][i] / 1e9 if 'EBIT' in annual_financials.index else np.nan

            #Tax Rate (annual_financials --> = Tax Provision / Pretax Income)
            tax_provision = annual_financials.loc['Tax Provision'][i] / 1e9  if 'Tax Provision' in annual_financials.index else np.nan
            pretax_income = annual_financials.loc['Pretax Income'][i] / 1e9  if 'Pretax Income' in annual_financials.index else np.nan
            tax_rate = tax_provision / pretax_income if pretax_income else np.nan
            tax_rate_2 = tax_rate * 100

            nopat = ebit_roic * (1 - tax_rate) if (ebit_roic and tax_rate) else np.nan

            ###Invested Capital (IC) = Short-term debt + Long-term debt + Shareholder equity - Cash/equivalents - Goodwill
            #Short-term debt (annual_balance --> Current Debt)
            #short_term_debt = annual_balance.loc['Current Debt'][i] / 1e9  if 'Current Debt' in annual_balance.index else 0
            short_term_debt = annual_balance.loc['Current Debt'][i] / 1e9 if 'Current Debt' in annual_balance.index and not pd.isna(annual_balance.loc['Current Debt'][i]) else 0


            #Long-term debt (annual_balance --> Long Term Debt)
            #long_term_debt = annual_balance.loc['Long Term Debt'][i] / 1e9  if 'Long Term Debt' in annual_balance.index else 0
            long_term_debt = annual_balance.loc['Long Term Debt'][i] / 1e9 if 'Long Term Debt' in annual_balance.index and not pd.isna(annual_balance.loc['Long Term Debt'][i]) else 0

            #Shareholder equity (annual_balance --> Stockholders Equity)
            stockholders_equity = annual_balance.loc['Stockholders Equity'][i] / 1e9  if 'Stockholders Equity' in annual_balance.index else 0

            #Cash/equivalents (annual_balance --> Cash And Cash Equivalents)
            cash_and_cash_equivalents = annual_balance.loc['Cash And Cash Equivalents'][i] / 1e9  if 'Cash And Cash Equivalents' in annual_balance.index else 0

            #Goodwill (annual_balance --> Goodwill)
            #goodwill = annual_balance.loc['Goodwill'][i] if 'Goodwill' in annual_balance.index else np.nan
            goodwill = annual_balance.loc['Goodwill'][i] / 1e9  if 'Goodwill' in annual_balance.index else 0
            if pd.isna(goodwill):
                goodwill = 0

            invested_capital = short_term_debt + long_term_debt + stockholders_equity - cash_and_cash_equivalents - goodwill

            roic = nopat / invested_capital if (nopat and invested_capital) else np.nan
            roic = roic * 100 #in %

            #Daten dem DataFrame hinzufügen
            kennzahlen_kapitaleffizienz.loc[fiscal_year] = [nopat, tax_rate_2, short_term_debt, long_term_debt, stockholders_equity, cash_and_cash_equivalents, goodwill, roic]

        ## Daten transponieren
        kennzahlen_kapitaleffizienz = kennzahlen_kapitaleffizienz.transpose()
        kennzahlen_kapitaleffizienz = kennzahlen_kapitaleffizienz.iloc[:, ::-1] # Spalten umkehren
        kennzahlen_kapitaleffizienz = kennzahlen_kapitaleffizienz.round(n) # alle Zahlen auf n Nachkommastellen runden                         

else:
    st.write("Keine Daten zur Kapitaleffizienz verfügbar.")
st.write(f"Anzahl verfügbarer Jahre aus den Bilanzen: {num_years}")
st.write(kennzahlen_kapitaleffizienz)
st.write()


####################################################################################

####################################################################################
## STEP 4 - Ivestiere nur in Gewinner
#Entwicklung des Aktienkurs seit IPO > CAGR von 10%
st.write("\nSCHRITT 4 - Investiere nur in Gewinner:")
st.write("Entwicklung des Aktienkurs seit IPO > CAGR von 10%")

def calculate_cagr(first_value, last_value, num_years):
    """Berechnet die jährliche Wachstumsrate (CAGR)."""
    return (last_value / first_value) ** (1 / num_years) - 1

# Tickersymbole der Indizes und der zu untersuchenden Aktie
symbols = {
    "S&P 500": "^GSPC",
    "S&P 100": "^OEX",
    "Russell 1000": "^RUI",
    "Russell 2000": "^RUT",
    "DAX": "^GDAXI",
    "TecDAX": "^TECDAX",
    "NIFTY 50": "^NSEI",
    stock_yfinance: stock_yfinance
}

# Liste für die gesammelten Daten
data = []

# Daten für jeden Index abrufen
for name, symbol in symbols.items():
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="max")

    # Erstes und letztes Datum sowie die Schlusskurse abrufen
    first_date = hist.index[0]
    last_date = hist.index[-1]
    first_close = hist['Close'].iloc[0]
    last_close = hist['Close'].iloc[-1]

    # Jahre zwischen erstem und letztem Datum berechnen
    num_years = (last_date - first_date).days / 365.25

    # CAGR berechnen in Prozent
    cagr = calculate_cagr(first_close, last_close, num_years)
    cagr = round(cagr * 100, 2)

    # Daten hinzufügen
    data.append([name, first_date.strftime("%d.%m.%Y"), last_date.strftime("%d.%m.%Y"), cagr])

# Erstellen eines DataFrames
benchmark = pd.DataFrame(data, columns=["Index Name", "Erstes Close Datum", "Letztes Close Datum", "CAGR (%)"])

st.write(benchmark)
####################################################################################

####################################################################################
## STEP 5 - Strukturelles Wachstum auf allen Ebenen?
#Wie ist das strukturelle Wachstum des Unternehmens?
#Alpha Vantage integrieren? --> später testen (Problem: ticker symbol konsistent anpassen)
st.write("\nSCHRITT 5 - Strukturelles Wachstum auf allen Ebenen?:")
st.write("Wie ist das strukturelle Wachstum des Unternehmens?")

#Umsatzwachstum (CAGR) > 8%
#EPS-Wachstum (CAGR) > 10%
#EPS LT Growth Estimate (3 bis 5 yr)
#letztes EPS
#aktuelles EPS
#zukünftiges EPS
available_years = annual_financials.shape[1]
total_revenue_beginning = annual_financials.loc['Total Revenue'][-1] / 1e9 if 'Total Revenue' in annual_financials.index else np.nan
total_revenue_ending = annual_financials.loc['Total Revenue'][0] / 1e9 if 'Total Revenue' in annual_financials.index else np.nan
total_revenue_cagr = (((total_revenue_ending / total_revenue_beginning) ** (1 / (available_years))) - 1) * 100
st.write(f"Umsatzwachstum (CAGR %): {total_revenue_cagr:.2f}")

diluted_eps_beginning = annual_financials.loc['Basic EPS'][-1] / 1e9 if 'Basic EPS' in annual_financials.index else np.nan
diluted_eps_ending = annual_financials.loc['Basic EPS'][0] / 1e9 if 'Basic EPS' in annual_financials.index else np.nan
diluted_eps_cagr = (((diluted_eps_ending / diluted_eps_beginning) ** (1 / (available_years))) - 1) * 100
st.write(f"EPS Wachstum (CAGR %): {diluted_eps_cagr:.2f}")

##URL-SCRAPING Yahoo Finance - Earnings Estimate
url = "https://finance.yahoo.com/quote/{}/analysis/" 
url = url.format(stock_yfinance)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
page = requests.get(url, headers=headers)
soup = BeautifulSoup(page.content, 'html.parser')

#rows = soup.find('tbody').find_all('tr')
#rows = soup.find('table', class_='W(100%) M(0) BdB Bdc($seperatorColor) Mb(25px)').find_all('tr')
rows = soup.find('tbody')

if rows is not None:
    data = []
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])
    eps_kennzahlen = pd.DataFrame(data, columns=['Metric', 'Current Qtr.', 'Next Qtr.', 'Current Year', 'Next Year'])
    #st.write(eps_estimate_kennzahlen)
    #st.write(f"Wert der Variable stock_yfinance: {stock_yfinance}")
    eps_current_year = eps_kennzahlen.loc[2, 'Current Year']
    eps_current_year = float(eps_current_year) #in Fließkommazahl umwandeln    
    st.write(f"Aktueller EPS-Wert: {eps_current_year}")
    
    avg_eps_estimate = eps_kennzahlen.loc[2, 'Next Year']
    avg_eps_estimate = float(avg_eps_estimate) #in Fließkommazahl umwandeln
    st.write(f"Avg. EPS-Projektion für die nächsten 12 Monate: {avg_eps_estimate}")
    if diluted_eps_ending != 0:
        eps_estimate_development = ((avg_eps_estimate - diluted_eps_ending) / diluted_eps_ending) * 100
        st.write(f"Entwicklung EPS-Wert (%): {eps_estimate_development:.2f}%")
    else:
        st.write("Aktueller EPS-Wert ist != 0, Division durch 0 nicht möglich.")
else:
    st.write(f"Keine EPS Estimate Daten auf Yahoo Finance für {stock_yfinance} gefunden.")




####################################################################################

####################################################################################
## STEP 6 - Nicht zu viel bezahlen
#Die Aktie muss eine faire Bewertung haben
st.write("\nSCHRITT 6 - Bezahle nicht zu viel für die Aktie:")
st.write("Aktuelle und zukünftig faire Bewerung des Unternehmens")
###Peter Lynch Faire Value
########## PETER LYNCH FAIR VALUE------------------------------------------------ 
peg_ratio = info2.get('pegRatio', np.nan)
trailing_eps = info2.get('trailingEps', np.nan)
# EBITDA CAGR ausrechnen
ebitda_beginning = annual_financials.loc['EBITDA'][available_years - 1] if 'EBITDA' in annual_financials.index else np.nan
ebitda_ending = annual_financials.loc['EBITDA'][0] if 'EBITDA' in annual_financials.index else np.nan
ebitda_cagr = (((ebitda_ending / ebitda_beginning) ** (1 / (available_years))) - 1) * 100
#peter_lynch_fair_value = peg_ratio * ebitda_cagr * trailing_eps

if isinstance(peg_ratio, complex) or isinstance(ebitda_cagr, complex) or isinstance(trailing_eps, complex):
    peter_lynch_fair_value = np.nan
else:
    peter_lynch_fair_value = peg_ratio * ebitda_cagr * trailing_eps



st.write(f"Aktueller Aktienkurs: {latest_close_price:.2f} zum Close Datum: {latest_date}")
st.write(f"Währung: {currency}")
st.write(f"Peter Lynch fair value (local currency): {peter_lynch_fair_value:.2f}")

###forwardPE
st.write(f"KGV (letzte 12 Monate): {t_kgv}")
st.write(f"KGV (zukünftige 12 Monate): {f_kgv}")

###DCF Modell
# komplexes Modell, später machen (Sonntag?) oder in separates Notebook outsourcen?
####################################################################################



####################################################################################
## STEP 7 - Quality Scoring (Zusammenfassung)
#Kennzahlen zusammenfassen, gegenüberstellen und ein Scoring ableiten
#Alpha Vantage integrieren?
st.write("\nSCHRITT 7 - Quality Scoring (Zusammenfassung):")
st.write("Kennzahlen zusammenfassen, gegenüberstellen und ein Scoring ableiten")
st.write("\n")

# Variablen deklarieren
schritt_1 = "1"
schritt_20 = "2.1"
schritt_21 = "2.2"
schritt_3 = "3"
schritt_4 = "4"
schritt_50 = "5.1"
schritt_51 = "5.2"
schritt_52 = "5.3"
schritt_6 = "6"

ergebnis_1 = "n/a"
juengstes_gross_margin = kennzahlen_profitabilitaet.loc['Rohmarge (%)'].iloc[-1]
ergebnis_20 = int(round(juengstes_gross_margin, 0)) if not pd.isna(juengstes_gross_margin) else np.nan
juengstes_profit_margin = kennzahlen_profitabilitaet.loc['Gewinnmarge (%)'].iloc[-1]
ergebnis_21 = int(round(juengstes_profit_margin, 0)) if not pd.isna(juengstes_profit_margin) else np.nan
juengster_roic = kennzahlen_kapitaleffizienz.loc['ROIC (%)'].iloc[-1]
ergebnis_3 = np.nan if pd.isna(juengster_roic) else int(round(juengster_roic, 0))
juengstes_cagr_stock = benchmark.loc[benchmark['Index Name'] == stock_yfinance, 'CAGR (%)'].iloc[0]
ergebnis_4 = int(round(juengstes_cagr_stock, 0)) if not pd.isna(juengstes_cagr_stock) else np.nan
ergebnis_50 = int(round(total_revenue_cagr, 0)) if not pd.isna(total_revenue_cagr) else np.nan
ergebnis_51 = int(round(diluted_eps_cagr, 0)) if not pd.isna(diluted_eps_cagr) else np.nan
ergebnis_52 = round(avg_eps_estimate, 2) if not pd.isna(avg_eps_estimate) else np.nan
ergebnis_6 = int(round(peter_lynch_fair_value, 2)) if not pd.isna(peter_lynch_fair_value) else np.nan

sollwert_1 = "n/a"
sollwert_20 = 40
sollwert_21 = 10
sollwert_3 = 15
sollwert_4 = 10
sollwert_50 = 8
sollwert_51 = 10
sollwert_52 = round(diluted_eps_ending, 2) if not pd.isna(diluted_eps_ending) else np.nan
sollwert_6 = int(round(latest_close_price, 2)) if not pd.isna(latest_close_price) else np.nan

comment_1 = "Geschäftsmodell"
comment_20 = "Rohmarge (%)"
comment_21 = "Gewinnmarge (%)"
comment_3 = "Kapitaleffizienz ROIC (%)"
comment_4 = "CAGR (%) stock seit IPO"
comment_50 = "Umsatzwachstum (CAGR %)"
comment_51 = "EPS-Wachstum (CAGR %)"
comment_52 = "Ziel: Ergebnis > Sollwert"
comment_6 = "Ziel: Ergebnis > Sollwert"

#Scoringmodell --> 5 Punkte: Exzellent; 4 Punkte: Sehr gut; 3 Punkte: Gut; 2 Punkte: Ausreichend; 1 Punkt: Schlecht
threshold_high_20 = 40 * 1.2 #AA
threshold_low_20 = 40 * 0.8
threshold_high_21 = 10 * 1.2 #AAA++
threshold_low_21 = 10 * 0.8
threshold_high_3 = 15 * 1.2 #AAA
threshold_low_3 = 15 * 0.8
threshold_high_4 = 10 * 1.2 #A
threshold_low_4 = 10 * 0.8
threshold_high_50 = 8 * 1.2 #AAA
threshold_low_50 = 8 * 0.8
threshold_high_51 = 10 * 1.2
threshold_low_51 = 10 * 0.8
threshold_high_52 = round(diluted_eps_ending, 2) * 1.2 if not pd.isna(diluted_eps_ending) else np.nan
threshold_low_52 = round(diluted_eps_ending, 2) * 0.8 if not pd.isna(diluted_eps_ending) else np.nan
threshold_high_6 = int(round(latest_close_price, 2)) * 3 if not pd.isna(latest_close_price) else np.nan #A
threshold_low_6 = int(round(latest_close_price, 2)) * 0.8 if not pd.isna(latest_close_price) else np.nan

score_1 = "n/a"
score_20 = f"{calculate_score_rohmarge(ergebnis_20, sollwert_20, threshold_high_20, threshold_low_20)}/5"
score_21 = f"{calculate_score_gewinnmarge(ergebnis_21, sollwert_21, threshold_high_21, threshold_low_21)}/5"
score_3 = f"{calculate_score_kapitaleffizienz(ergebnis_3, sollwert_3, threshold_high_3, threshold_low_3)}/5"
score_4 = f"{calculate_score_entwicklung_stock_ipo(ergebnis_4, sollwert_4, threshold_high_4, threshold_low_4)}/5"
score_50 = f"{calculate_score_umsatzwachstum(ergebnis_50, sollwert_50, threshold_high_50, threshold_low_50)}/5"
score_51 = f"{calculate_score_eps_wachstum(ergebnis_51, sollwert_51, threshold_high_51, threshold_low_51)}/5"
score_52 = f"{calculate_score_eps_vergleich(ergebnis_52, sollwert_52, threshold_high_52, threshold_low_52)}/5"
score_6 = f"{calculate_score_peter_lynch_fair_value(ergebnis_6, sollwert_6, threshold_high_6, threshold_low_6)}/5"

weight_1 = "n/a"
weight_20 = "n/a"
weight_21 = "n/a"
weight_3 = "n/a"
weight_4 = "n/a"
weight_50 = "n/a"
weight_51 = "n/a"
weight_52 = "n/a"
weight_6 = "n/a"

sum_1 = "n/a"
sum_20 = "n/a"
sum_21 = "n/a"
sum_3 = "n/a"
sum_4 = "n/a"
sum_50 = "n/a"
sum_51 = "n/a"
sum_52 = "n/a"
sum_6 = "n/a"

# Dictionary mit Variablen erstellen
data = {"step": [schritt_1, schritt_20, schritt_21,schritt_3, schritt_4, schritt_50, schritt_51, schritt_52, schritt_6], 
        "Ergebnis": [ergebnis_1, ergebnis_20, ergebnis_21, ergebnis_3, ergebnis_4, ergebnis_50, ergebnis_51, ergebnis_52, ergebnis_6],
        "Sollwert": [sollwert_1, sollwert_20, sollwert_21, sollwert_3, sollwert_4, sollwert_50, sollwert_51, sollwert_52, sollwert_6],
        "comment": [comment_1, comment_20, comment_21, comment_3, comment_4, comment_50, comment_51, comment_52, comment_6],
        "score": [score_1, score_20, score_21, score_3, score_4, score_50, score_51, score_52, score_6],
        "weight": [weight_1, weight_20, weight_21, weight_3, weight_4, weight_50, weight_51, weight_52, weight_6],
        "sum": [sum_1, sum_20, sum_21, sum_3, sum_4, sum_50, sum_51, sum_52, sum_6]}

# DataFrame erzeugen
quality_scoring = pd.DataFrame(data)
total_sum_quality_scoring_simple = calculate_score_rohmarge(ergebnis_20, sollwert_20, threshold_high_20, threshold_low_20) + calculate_score_gewinnmarge(ergebnis_21, sollwert_21, threshold_high_21, threshold_low_21) + calculate_score_kapitaleffizienz(ergebnis_3, sollwert_3, threshold_high_3, threshold_low_3) + calculate_score_entwicklung_stock_ipo(ergebnis_4, sollwert_4, threshold_high_4, threshold_low_4) + calculate_score_umsatzwachstum(ergebnis_50, sollwert_50, threshold_high_50, threshold_low_50) + calculate_score_eps_wachstum(ergebnis_51, sollwert_51, threshold_high_51, threshold_low_51) + calculate_score_eps_vergleich(ergebnis_52, sollwert_52, threshold_high_52, threshold_low_52) + calculate_score_peter_lynch_fair_value(ergebnis_6, sollwert_6, threshold_high_6, threshold_low_6)
total_sum_quality_scoring_weighted = "n/a"

st.write(quality_scoring)
st.write(f"Gesamtergesamtergebnis (einfach summiert): {total_sum_quality_scoring_simple}")
st.write(f"Gesamtergesamtergebnis (gewichtet): {total_sum_quality_scoring_weighted}")
st.write("\n")
st.write("\n")
####################################################################################


####################################################################################
## STEP 8 - Ausblick, mittels LSTM Modell vorhersagen des close Preises
#Outsourcing?
st.write("\nSCHRITT 8 - Ausblick, mittels LSTM Modell vorhersagen des close Preises:")
st.write("Das LSTM Modell ist ausgelagert.")
#LSTM Modell entwickeln


####################################################################################






############################## Zusammenführen und Anzeigen der Tabelle von Kaufsignalen
st.write()
st.write("\n")
st.write("Kaufsignale:")
st.write(matched_signals_df)
st.write("\n")
#st.write("Verkaufsignale:")
#st.write(matched_verkauf_signals_df)
#st.write("\n")
####################################################################

# show diagram Figure 1
plt.tight_layout()
plt.show()
st.write()
st.write()

## Jetzt kommt der Berichtskram
## TABELLE 1 - Quartalberichte
available_quarters = quarterly_financials.shape[1]
st.write(f"Es stehen {available_quarters} Quartale zur Verfügung:")
if available_quarters > 0:
    st.write(kennzahlen)
st.write()
st.write()

## TABELLE 2 - Jahresberichte
available_years = annual_financials.shape[1]
st.write(f"Es stehen {available_years} Jahre zur Verfügung:")
#st.write(f"Anzahl Überschneidungen: {common_years}")
if available_years > 0:
    st.write(kennzahlen_jahresberichte)
st.write()
st.write()

## News zum aktuellen Tickersymbol________________________________________________
# Die Titel der letzten n Artikel ins Deutsche übersetzen
for article in last_n_articles:
    title = article['title']
    translated_title = translate_to_german(title)
    
    # Unix-Zeitstempel in ein Datum umwandeln
    timestamp = article['providerPublishTime']
    date = datetime.fromtimestamp(timestamp)
    formatted_date = date.strftime("%d.%m.%Y")    
    
    st.write(f"Übersetzt: {translated_title}")
    #st.write(f"Original: {title}")
    st.write(f"Quelle: {article['link']}")
    st.write(f"Veröffentlicher: {article['publisher']}")
    st.write(f"Datum: {formatted_date}")    
    st.write()
######################################################################################