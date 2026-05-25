# BorsaBot - Entegrasyon Rehberi
## GMSTR & Halka Arz Tahmin Motoru

---

## 1. PROJE YAPISI

```
otonomBorsa/
│
├── main.py                          # Yeni: Kivy Mobil Uygulama (Giriş noktası)
├── main_trading_bot.py              # Eski: Trading bot (yedeklendi)
│
├── halka_arz_system/                # YENI MODUL
│   ├── __init__.py
│   ├── data_model.py                # Veri modelleri (Enum, Dataclass)
│   ├── predictor.py                 # Ana tahmin motoru (Tavan Gucu Skoru)
│   ├── backtest.py                  # Backtest & optimizasyon
│   ├── historical_data.py           # Gecmis halka arz verileri
│   └── main.py                      # CLI arayuzu
│
├── gmstr_system/                    # MEVCUT MODUL
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── training.py
│   ├── predictor.py
│   ├── evaluation.py
│   ├── live_monitor.py
│   └── main.py
│
├── buildozer.spec                   # APK build konfigurasyonu (guncellendi)
└── requirements.txt                 # Bagimliliklar (guncellendi)
```

---

## 2. MODULLER ARASI ILISKI

```
┌─────────────────────────────────────────────────────────────┐
│                    KIVY MOBIL UYGULAMA                        │
│                      (main.py)                                │
├──────────────────────────┬──────────────────────────────────┤
│    Halka Arz Sekmesi     │         GMSTR Sekmesi            │
│  (halka_arz_system/)     │      (gmstr_system/)             │
├──────────────────────────┼──────────────────────────────────┤
│  • Tavan Tahmini         │  • Model Egitimi                 │
│  • Backtest              │  • Tahmin                        │
│  • Parametrik Skorlama   │  • Canli Monitor                 │
│  • Tarihsel Veriler      │  • Ensemble ML                   │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 3. HALKA ARZ TAHMIN MOTORU - KULLANIM

### 3.1 Python API (Programatik Kullanim)

```python
from halka_arz_system.data_model import (
    HalkaArzGirdileri, DagitimYontemi, Sektorm, PiyasaDuyarliligi
)
from halka_arz_system.predictor import HalkaArzPredictor

# Tahmin motoru olustur
predictor = HalkaArzPredictor()

# Girdileri hazirla
girdi = HalkaArzGirdileri(
    sirket_adi="Yeni Sirket A.S.",
    dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
    halka_arz_boyutu_tl=85_000_000,
    katilimci_beklentisi=1400,
    kurumsal_oran=0.25,
    kurumsal_taahhut=False,
    sektor=Sektorm.ENERJI,
    borcluluk_orani=0.35,
    net_kar_buyumesi=0.25,
    piyasa_duyarliligi=PiyasaDuyarliligi.BOGA,
    lot_basi_dusen_maliyet=1200,
)

# Tahmin yap
tahmin = predictor.tahmin_yap(girdi)

print(tahmin.ozet())
# Cikti: Yeni Sirket A.S.: Skor=92.2/100 | Kategori=guclu_tavan | Tahmin=7-10+ Gun Tavan | Guven=%88

# Detayli skorlara erisim
for skor in tahmin.detayli_skorlar:
    print(f"{skor.parametre_adi}: {skor.normalize_skor:.1f} (Katkı: {skor.agirlikli_skor:.2f})")
```

### 3.2 CLI Kullanimi

```bash
# Demo tahminler
python -m halka_arz_system.main --mode demo

# Tek tahmin
python -m halka_arz_system.main --mode predict \
  --sirket "Ornek A.S." \
  --dagitim bireysele_esit \
  --boyut 100000000 \
  --katilimci 1000 \
  --kurumsal 0.25 \
  --sektor enerji \
  --borcluluk 0.35 \
  --kar-buyume 0.20 \
  --piyasa boga

# Backtest
python -m halka_arz_system.main --mode backtest
```

### 3.3 Skorlama Sistemi

| Parametre | Agirlik | Aciklama |
|-----------|---------|----------|
| Dagitim Yontemi | %25 | Esit dagitim = uzun tavan |
| Halka Arz Boyutu | %20 | Kucuk tahta = kolay manipulasyon |
| Katilimci Sayisi | %15 | Yuksek katilim = talep patlamasi |
| Kurumsal Oran | %15 | Dengeli + taahhut = guven |
| Sektör & Finansal | %15 | Populer sektor + iyi finansal |
| Piyasa Duyarliligi | %10 | BIST 100 trendi |

### 3.4 Kategoriler

| Skor | Kategori | Tahmin |
|------|----------|--------|
| 80-100 | Guclu Tavan | 7-10+ Gun |
| 60-80 | Orta Tavan | 4-6 Gun |
| 40-60 | Zayif Tavan | 2-3 Gun |
| <40 | Riskli | 1-2 Gun / Erken Bozma |

---

## 4. GMSTR MODULU - KULLANIM (Mevcut)

```bash
# Model egitimi
python -m gmstr_system.main --mode train

# Tahmin
python -m gmstr_system.main --mode predict

# Full pipeline (egitim + tahmin)
python -m gmstr_system.main --mode full

# Canli monitor
python -m gmstr_system.main --mode live --interval 300
```

---

## 5. MOBIL UYGULAMA (Kivy)

### 5.1 Bilgisayarda Calistirma

```bash
# Gereksinimleri yukle
pip install -r requirements.txt

# Uygulamayi baslat
python main.py
```

### 5.2 Android APK Olusturma (Buildozer)

```bash
# Buildozer yuklu degilse
pip install buildozer

# Debug APK olustur
buildozer android debug

# Cihaza yukle ve calistir
buildozer android debug deploy run

# Log takibi
buildozer android logcat
```

### 5.3 Buildozer Konfigurasyonu

`buildozer.spec` dosyasi otomatik guncellendi:
- `title = BorsaBot - Halka Arz & GMSTR`
- `package.name = borsabot`
- `requirements = python3,kivy,numpy,pandas,requests,scikit-learn,xgboost,lightgbm`
- `android.permissions = INTERNET, VIBRATE`

---

## 6. IKI MODULU AYRI AYRI CALISTIRMA

Eski trading botu hala kullanilabilir:

```bash
# Eski trading bot
python main_trading_bot.py

# GMSTR sistem
python -m gmstr_system.main --mode full

# Halka Arz sistemi
python -m halka_arz_system.main --mode demo
```

---

## 7. GELISTIRME ve BACKTEST

### 7.1 Yeni Veri Ekleme

```python
from halka_arz_system.historical_data import veri_ekle, GecmisHalkaArz
from halka_arz_system.data_model import HalkaArzGirdileri, DagitimYontemi, Sektorm, PiyasaDuyarliligi

yeni_veri = GecmisHalkaArz(
    sirket_adi="Yeni Halka Arz (YENI)",
    halka_arz_tarihi="2025-05-01",
    girdiler=HalkaArzGirdileri(...),
    gerceklesen_tavan_gunu=5,
)
veri_ekle(yeni_veri)
```

### 7.2 Agirlik Optimizasyonu

```python
from halka_arz_system.backtest import agirlik_optimizasyonu
from halka_arz_system.historical_data import get_gecmis_veriler

veriler = get_gecmis_veriler()
en_iyi_agirliklar, rapor = agirlik_optimizasyonu(veriler, iterasyon=500)
print(en_iyi_agirliklar)
```

---

## 8. ONEMLI NOTLAR

1. **Tahmin motoru deterministiktir** - Ayni girdiler her zaman ayni skoru uretir.
2. **Backtest icin ornek veriler mevcuttur** - Gercek verilerle KAP/BIST verilerinden genisletilmelidir.
3. **Mobil uygulama Kivy tabanlidir** - Python 3.8+ ve Kivy 2.2+ gerektirir.
4. **GMSTR modulu bagimsiz calisir** - Halka Arz modulu ile cakismaz.
5. **JSON ciktisi alinabilir** - Tahmin ve backtest sonuclari JSON olarak kaydedilir.

---

## 9. HATA AYIKLAMA

```bash
# Kivy loglari
~/.kivy/logs/

# Buildozer loglari
buildozer android logcat

# Modul testi
python -m halka_arz_system.main --mode demo
python -m gmstr_system.main --mode predict
```
