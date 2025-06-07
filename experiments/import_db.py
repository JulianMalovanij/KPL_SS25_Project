import sqlite3

import pandas as pd

# ----------------------------------------
# 1) Verbindung zur SQLite-Datenbank
# ----------------------------------------
# Wenn "walmart.db" noch nicht existiert, wird es neu angelegt.
conn = sqlite3.connect("walmart_old.db")

# ----------------------------------------
# 2) CSVs mit pandas einlesen
# ----------------------------------------
# Passe den Pfad an, falls deine CSVs an anderer Stelle liegen.
df_stores = pd.read_csv("../data/stores.csv")  # Spalten: Store, Type, Size
df_features = pd.read_csv("../data/features.csv")  # Spalten: Store, Date, Temperature, …, IsHoliday
df_train = pd.read_csv("../data/train.csv")  # Spalten: Store, Dept, Date, Weekly_Sales, IsHoliday

# Datumsspalten in pandas-Datetime umwandeln
df_features["Date"] = pd.to_datetime(df_features["Date"], format="%Y-%m-%d")
df_train["Date"] = pd.to_datetime(df_train["Date"], format="%Y-%m-%d")

# ----------------------------------------
# 3) Tabellen automatisch mit to_sql erstellen
# ----------------------------------------
# 3.1) Stores-Tabelle
df_stores = df_stores.rename(
    columns={"Store": "StoreID", "Type": "StoreType", "Size": "StoreSize"}
)
# Wenn die Tabelle bereits existiert, wird sie überschrieben (replace).
df_stores.to_sql("Store", conn, if_exists="replace", index=False)

# 3.2) Features-Tabelle
# Spalten umbenennen, damit sie ohne Leerzeichen in SQLite landen
df_feat = df_features.rename(
    columns={
        "Store": "StoreID",
        "Fuel_Price": "FuelPrice",
        "IsHoliday": "IsHoliday",
        "MarkDown1": "MarkDown1",
        "MarkDown2": "MarkDown2",
        "MarkDown3": "MarkDown3",
        "MarkDown4": "MarkDown4",
        "MarkDown5": "MarkDown5",
        "CPI": "CPI",
        "Unemployment": "Unemployment",
    }
)
df_feat.to_sql("StoreFeature", conn, if_exists="replace", index=False)

# 3.3) WeeklySales-Tabelle (Train-Daten)
df_sales = df_train.rename(
    columns={
        "Store": "StoreID",
        "Dept": "DeptID",
        "Weekly_Sales": "WeeklySales",
        "IsHoliday": "IsHoliday",
    }
)
df_sales.to_sql("WeeklySales", conn, if_exists="replace", index=False)

# ----------------------------------------
# 4) Indizes anlegen (optional, aber empfohlen)
# ----------------------------------------
cursor = conn.cursor()
cursor.execute("CREATE INDEX IF NOT EXISTS idx_feat_store_date ON StoreFeature(StoreID, Date);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_store_date ON WeeklySales(StoreID, Date);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_dept      ON WeeklySales(DeptID);")
conn.commit()

# ----------------------------------------
# 5) Verbindung schließen
# ----------------------------------------
conn.close()

print("Fertig! Die SQLite-Datei 'walmart.db' wurde erstellt und enthält drei Tabellen:")
print("  • Store       (Spalten: StoreID, StoreType, StoreSize)")
print("  • StoreFeature(Spalten: StoreID, Date, Temperature, FuelPrice, …, IsHoliday)")
print("  • WeeklySales (Spalten: StoreID, DeptID, Date, WeeklySales, IsHoliday)")
