# Verkaufsanalyse-App 🚀

Eine **Streamlit**-Anwendung zur **deskriptiven Analyse**, **Verkaufsvorhersage** und **Promotions-Optimierung** auf
Basis historischer Verkaufsdaten.

---

## 🛠️ Features

1. **Deskriptive Analyse**

    * Historische Verkaufsdaten untersuchen
    * Trends, Saisonalitäten und Ausreißer visualisieren
    * Filter nach Store, Department, Produkt und Lager

2. **Verkaufsvorhersage**

    * Prognosen mit **Prophet**, **ARIMA** und **Holt-Winters**
    * KPI-Berechnung (Gesamtnachfrage, Durchschnitt, Volatilität, Wachstum)
    * Automatisches Speichern der Vorhersagen in SQLite

3. **Promotions-Optimierung**

    * Aktionen basierend auf Prognosedaten planen
    * Budget- und Zeitraumoptimierung zur Umsatzsteigerung

---

## 🚀 Installation

**Voraussetzungen:**

   * Python 3.12

**Anleitung:**

1. Klone das Repository:

   ```bash
   git clone <REPO_URL>
   cd <REPO_FOLDER>
   ```

2. Erstelle ein virtuelles Environment und installiere Abhängigkeiten:

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows PowerShell

   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚡ Start der App

Starte die Streamlit-App mit:

```bash
streamlit run main.py
```

Die **Willkommensseite** wird beim Start angezeigt und gibt einen Überblick über die Funktionen.

---

## 📁 Repository-Struktur

```text
.
├── main.py                  # Entry-Point der Streamlit-App
├── pages/                   # Seiten der App (Willkommensseite, Analysen, Forecasts, Optimierung, ...)
├── database/                # Datenbankverwaltung
│   ├── import_product_db.py # Datenbank-Import
│   ├── data_loader.py       # Daten-Lade-Logik
│   ├── data_writer.py       # Daten-Speichern-Logik
│   └── ...
├── layout.py                # Layout-Wrapper (Header/Footer)
├── logic/                   # Logik-Implementierung
│   ├── forecasting/         # Forecast-Modelle und Funktionen
│   ├── optimization/        # Optimierungs-Modelle und Funktionen
│   └── ...
└── requirements.txt         # Python-Abhängigkeiten
```

---

## 📦 Abhängigkeiten

Die App nutzt folgende Pakete:

```text
pandas~=2.3.0
numpy~=1.26.4
scikit-learn~=1.7.0
statsmodels~=0.14.4
pmdarima~=2.0.4
matplotlib~=3.10.3
PuLP~=3.2.1
seaborn~=0.13.2
streamlit~=1.45.1
prophet~=1.1.7
plotly~=6.1.2
pathspec~=0.12.1
```

---
<div style='text-align: center; color: grey; font-size: 0.9em;'>
<p>Verkaufsanalyse-App | Projekt für das Modul KPL an der OTH Regensburg im SoSe 2025 | Gruppe 10</p>
<p>Nikolai Mallet, Julian Malovanij, Bastian Rapps, Sebastian Schall | Erstellt mit ❤️ und Streamlit</p>
</div>
