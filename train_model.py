import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# --- 1. Baca data ---
df = pd.read_csv("merged_from_json.csv")

# ttrcheck_description -> dari teknisi (diagnosis)
# ttcheck_remark_service -> dari customer (keluhan)
# ttcheck_device_service, ttcheck_brand_service, ttcheck_series_service -> info perangkat

def map_kategori(desc: str) -> str:
    if not isinstance(desc, str):
        return "LAINNYA"
    d = desc.lower()

    # Dummy / non-kerusakan jelas
    if "dummy" in d or "tes" in d or "test" in d or "done" in d or "terimakasih" in d:
        return "DUMMY"

    # Battery / power
    if "baterai" in d or "battery" in d:
        return "BATTERY"
    if "psu" in d or "kabel power" in d or "adaptor" in d or "charger" in d:
        return "POWER"

    # Motherboard / mainboard
    if "motherboard" in d or "mainboard" in d or "mobo" in d or "mainboard" in d:
        return "MOTHERBOARD"

    # Overheat / cleaning + pasta
    if "overheat" in d or "over heat" in d or "panas" in d:
        return "OVERHEAT"
    if "repasta" in d or "ganti pasta" in d or ("pasta" in d and "clean" in d):
        return "OVERHEAT"
    if "cleaning dan ganti pasta" in d or "cleaning & repasta" in d:
        return "OVERHEAT"

    # Keyboard / touchpad
    if "keyboard" in d or "tuts" in d:
        return "KEYBOARD"
    if "touch pad" in d or "touchpad" in d:
        return "TOUCHPAD"

    # Engsel
    if "engsel" in d:
        return "ENGSEL"

    # Display / LCD / monitor
    if "lcd" in d or "layar" in d or "tampilan" in d or "display" in d or "monitor" in d:
        return "DISPLAY"

    # Printer / tinta / paperjam / head
    if "printer" in d or "hasil print" in d or "paperjam" in d or "paper jam" in d or "tray" in d:
        return "PRINTER"
    if "tinta" in d or "ink" in d or "cartridge" in d or "catridge" in d:
        return "PRINTER"

    # Storage / OS / software
    if "ssd" in d or "hdd" in d or "hardisk" in d or "hard disk" in d:
        return "STORAGE"
    if "install ulang" in d or "install ulang windows" in d or "windows 10" in d or "antivirus" in d or "anti virus" in d or "driver" in d or "firmware" in d:
        return "SOFTWARE"

    return "LAINNYA"


# buat kolom label kategori_kerusakan dari deskripsi teknisi
df["kategori_kerusakan"] = df["ttrcheck_description"].apply(map_kategori)

required_cols = [
    "kategori_kerusakan",
    "ttcheck_remark_service",
    "ttcheck_device_service",
    "ttcheck_brand_service",
]

# dropna hanya untuk kolom wajib
df_model = df[required_cols].dropna().copy()

# tambahkan kolom series (boleh NaN, nanti di-fill "")
df_model["ttcheck_series_service"] = df.loc[df_model.index, "ttcheck_series_service"]

# --- 2. Filter label yang muncul cukup sering ---
label_counts = df_model["kategori_kerusakan"].value_counts()
valid_labels = label_counts[label_counts >= 2].index   # bisa turunin jadi 3 kalau data sedikit
df_model = df_model[df_model["kategori_kerusakan"].isin(valid_labels)]

print("Jumlah data setelah filter:", len(df_model))
print("Jumlah kelas setelah filter:", df_model["kategori_kerusakan"].nunique())
print("Distribusi label:\n", df_model["kategori_kerusakan"].value_counts())

# --- 3. Gabungkan fitur jadi text_input ---
df_model["text_input"] = (
    df_model["ttcheck_device_service"].fillna("") + " " +
    df_model["ttcheck_brand_service"].fillna("") + " " +
    df_model["ttcheck_series_service"].fillna("") + " " +
    df_model["ttcheck_remark_service"].fillna("")
)

X = df_model["text_input"]
y = df_model["kategori_kerusakan"]

# --- 4. Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- 5. Pipeline TF-IDF + Logistic Regression ---
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True
    )),
    ("clf", LogisticRegression(
        max_iter=300,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

print("Training model..")
model.fit(X_train, y_train)

# --- 6. Evaluasi ---
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

# --- 7. Simpan model ---
joblib.dump(model, "model_keluhan.pkl")
print("\nModel disimpan sebagai model_keluhan.pkl")
