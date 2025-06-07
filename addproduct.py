import pandas as pd
import sqlite3

# ----------------------------------------
# 1) Verbindung zur SQLite-Datenbank öffnen
# ----------------------------------------
# Wenn "walmart.db" noch nicht existiert, wird es neu angelegt.
DB_PATH = "walmart.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ----------------------------------------
# 1a) Prüfen, ob in Tabelle Store die Spalte WarehouseCode existiert
# ----------------------------------------
cursor.execute("PRAGMA table_info(Store);")
store_columns = [row[1] for row in cursor.fetchall()]

if "WarehouseCode" not in store_columns:
    cursor.execute("ALTER TABLE Store ADD COLUMN WarehouseCode TEXT;")
    conn.commit()

# ----------------------------------------
# 2) CSVs mit pandas einlesen
# ----------------------------------------
# Stelle sicher, dass du aus dem Projekt-Root arbeitest,
# damit "data/..." korrekt gefunden wird.
# Überprüfe kurz das aktuelle Arbeitsverzeichnis:
# print("Arbeitsverzeichnis:", os.getcwd())

# 2.1) Ursprüngliche Walmart-Dateien
df_stores   = pd.read_csv("data/stores.csv")    # Spalten: Store, Type, Size
df_features = pd.read_csv("data/features.csv")  # Spalten: Store, Date, Temperature, …, IsHoliday
df_train    = pd.read_csv("data/train.csv")     # Spalten: Store, Dept, Date, Weekly_Sales, IsHoliday

# 2.2) Neue Historical Demand-Datei
df_hist = pd.read_csv("data/Historical Product Demand.csv")
# Spalten: Product_Code, Warehouse, Product_Category, Date, Order_Demand

# Datumsspalten in pandas-Datetime umwandeln
df_features["Date"] = pd.to_datetime(df_features["Date"], format="%Y-%m-%d")
df_train   ["Date"] = pd.to_datetime(df_train["Date"],   format="%Y-%m-%d")

# Bei Historic-Datei: Datum im Format "YYYY/M/D" oder "YYYY/MM/DD"
df_hist["Date"] = pd.to_datetime(df_hist["Date"], format="%Y/%m/%d")

# ----------------------------------------
# 3) Tabellen automatisch mit to_sql erstellen / befüllen
# ----------------------------------------

# 3.1) Stores-Tabelle (vorhanden, aber wir schreiben neu, falls sich etwas geändert hat)
df_stores = df_stores.rename(
    columns={"Store": "StoreID", "Type": "StoreType", "Size": "StoreSize"}
)
df_stores.to_sql("Store", conn, if_exists="replace", index=False)

# Nach dem Überschreiben von Store existiert WarehouseCode möglicherweise nicht mehr.
# Führe den Check erneut durch, damit wir WarehouseCode hinzufügen können.
cursor.execute("PRAGMA table_info(Store);")
store_columns = [row[1] for row in cursor.fetchall()]
if "WarehouseCode" not in store_columns:
    cursor.execute("ALTER TABLE Store ADD COLUMN WarehouseCode TEXT;")
    conn.commit()

# 3.2) Features-Tabelle
df_feat = df_features.rename(
    columns={
        "Store":        "StoreID",
        "Fuel_Price":   "FuelPrice",
        "IsHoliday":    "IsHoliday",
        "MarkDown1":    "MarkDown1",
        "MarkDown2":    "MarkDown2",
        "MarkDown3":    "MarkDown3",
        "MarkDown4":    "MarkDown4",
        "MarkDown5":    "MarkDown5",
        "CPI":          "CPI",
        "Unemployment": "Unemployment",
    }
)
df_feat.to_sql("StoreFeature", conn, if_exists="replace", index=False)

# 3.3) WeeklySales-Tabelle (Train-Daten)
df_sales = df_train.rename(
    columns={
        "Store":        "StoreID",
        "Dept":         "DeptID",
        "Weekly_Sales": "WeeklySales",
        "IsHoliday":    "IsHoliday",
    }
)
df_sales.to_sql("WeeklySales", conn, if_exists="replace", index=False)

# 3.4) HistoricalDemand-Tabelle (neu)
# Spalten: ProductCode, Warehouse, ProductCategory, Date, OrderDemand
df_hist_renamed = df_hist.rename(
    columns={
        "Product_Code":     "ProductCode",
        "Warehouse":        "WarehouseCode",
        "Product_Category": "ProductCategory",
        "Order_Demand":     "OrderDemand"
    }
)
df_hist_renamed["Date"] = df_hist_renamed["Date"].dt.strftime("%Y-%m-%d")
df_hist_renamed.to_sql("HistoricalDemand", conn, if_exists="replace", index=False)

# ----------------------------------------
# 4) Neue Tabellen für ProductCategory und Product (falls nicht schon vorhanden)
# ----------------------------------------
cursor.executescript("""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ProductCategory (
  CategoryID   INTEGER PRIMARY KEY AUTOINCREMENT,
  Name         TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS Product (
  ProductID    INTEGER PRIMARY KEY AUTOINCREMENT,
  CategoryID   INTEGER   NOT NULL,
  ProductCode  TEXT      UNIQUE,
  FOREIGN KEY (CategoryID) REFERENCES ProductCategory(CategoryID)
);
""")
conn.commit()

# 4.1) Befüllen von ProductCategory
unique_cats = df_hist["Product_Category"].unique()
for cat in unique_cats:
    cursor.execute(
        "INSERT OR IGNORE INTO ProductCategory(Name) VALUES (?);",
        (cat,)
    )
conn.commit()

# 4.2) Befüllen von Product
for prod, cat in df_hist[["Product_Code", "Product_Category"]].drop_duplicates().values:
    cursor.execute("SELECT CategoryID FROM ProductCategory WHERE Name = ?;", (cat,))
    cat_id = cursor.fetchone()[0]
    cursor.execute(
        "INSERT OR IGNORE INTO Product(ProductCode, CategoryID) VALUES (?, ?);",
        (prod, cat_id)
    )
conn.commit()

# ----------------------------------------
# 5) (Optional) Mapping WarehouseCode → StoreID
# ----------------------------------------
# Falls du Retail-Store-Einträge in "Store" um das Feld WarehouseCode erweitert hast,
# musst du nun jedem Store den passenden WarehouseCode zuweisen.
# Beispiel: Wenn StoreType 'A' zu WarehouseCode 'Whse_A' gehört, usw.
# Du kannst hier beliebige Logik einsetzen, z. B.:
#
# mapping = {
#   "A": "Whse_A",
#   "B": "Whse_B",
#   "C": "Whse_C",
#   "J": "Whse_J",
#   ...
# }
#
# In df_hist findest du alle WarehouseCodes:
#
for wh_code in df_hist_renamed["WarehouseCode"].unique():
    # Beispiel, wir setzen WarehouseCode nur in Stores, in denen StoreType gleich dem Buchstaben nach 'Whse_'
    # (das kannst du anpassen, je nachdem wie deine Stores benannt sind!)
    inferred_type = wh_code.split("_")[-1]  # z.B. 'Whse_J' → 'J'
    cursor.execute("""
        UPDATE Store
        SET WarehouseCode = ?
        WHERE WarehouseCode IS NULL AND StoreType = ?;
    """, (wh_code, inferred_type))
conn.commit()

# ----------------------------------------
# 6) Indexe anlegen (Performance-Tuning)
# ----------------------------------------

# Walmart-Tabellen
cursor.execute("CREATE INDEX IF NOT EXISTS idx_feat_store_date ON StoreFeature(StoreID, Date);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_store_date ON WeeklySales(StoreID, Date);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_dept      ON WeeklySales(DeptID);")

# Neue HistoricalDemand-Indexe
cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_prod       ON HistoricalDemand(ProductCode);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_wh         ON HistoricalDemand(WarehouseCode);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_date       ON HistoricalDemand(Date);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_prod_wh    ON HistoricalDemand(ProductCode, WarehouseCode);")

# Produkt-Tabellen
cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_cat     ON Product(CategoryID);")
conn.commit()

# ----------------------------------------
# 7) Verbindung schließen
# ----------------------------------------
conn.close()

print("Fertig! Die SQLite-Datei 'walmart.db' enthält nun:")
print("  • Store (mit optionaler Spalte WarehouseCode)")
print("  • StoreFeature")
print("  • WeeklySales")
print("  • HistoricalDemand")
print("  • ProductCategory")
print("  • Product")