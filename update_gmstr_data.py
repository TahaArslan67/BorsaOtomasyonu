"""
GMSTR CSV Otomatik Güncelleme
Yahoo Finance'den günlük veri çekip areaxdatetime.csv'ye ekler.
Çalıştırma: python update_gmstr_data.py
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
CSV_PATH = ROOT / 'claude' / 'areaxdatetime.csv'


def update_csv():
    print("=" * 60)
    print("  GMSTR VERİ GÜNCELLEME")
    print("=" * 60)

    # Mevcut CSV'yi yükle
    try:
        df_existing = pd.read_csv(CSV_PATH, encoding='utf-8', index_col=0, parse_dates=True)
        last_date = df_existing.index[-1]
        print(f"  Mevcut veri: {len(df_existing)} satır | Son tarih: {last_date.date()}")
    except Exception as e:
        print(f"  CSV yükleme hatası: {e}")
        return False

    today = datetime.now().date()
    if last_date.date() >= today:
        print(f"  Veri zaten güncel ({last_date.date()})")
        return True

    # Yahoo Finance'den yeni veri çek
    try:
        import yfinance as yf
        ticker = yf.Ticker("GMSTR.IS")

        # Son tarihten bugüne kadar çek
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"  Yahoo Finance'den çekiliyor: {start_date} -> {end_date}")
        df_new = ticker.history(start=start_date, end=end_date, interval='1d')

        if len(df_new) == 0:
            print("  Yeni veri yok (piyasa kapalı olabilir)")
            return True

        # Timezone temizle
        if df_new.index.tz is not None:
            df_new.index = df_new.index.tz_localize(None)

        print(f"  {len(df_new)} yeni satır bulundu")

        # Mevcut CSV kolonlarıyla eşleştir
        # CSV'deki kolonlar: Open, High, Low, Close, Volume (ve türetilmiş kolonlar)
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_new_clean = df_new[cols_to_keep].copy()

        # Mevcut CSV'nin ek kolonlarını NaN ile doldur
        for col in df_existing.columns:
            if col not in df_new_clean.columns:
                df_new_clean[col] = np.nan

        # Birleştir
        df_combined = pd.concat([df_existing, df_new_clean])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()

        # Kaydet
        df_combined.to_csv(CSV_PATH, encoding='utf-8')
        print(f"  Güncellendi: {len(df_existing)} -> {len(df_combined)} satır")
        print(f"  Son tarih: {df_combined.index[-1].date()}")
        return True

    except Exception as e:
        print(f"  Yahoo Finance hatası: {e}")
        import traceback
        traceback.print_exc()
        return False


def fetch_macro_data():
    """
    Makro veri çek: Dolar/TL, Altın, Gümüş vadeli işlemler
    ve GMSTR CSV'sine ekle.
    """
    print("\n[Makro Veri Güncelleme]")
    try:
        import yfinance as yf

        # Mevcut CSV'yi yükle
        df = pd.read_csv(CSV_PATH, encoding='utf-8', index_col=0, parse_dates=True)
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        macro_tickers = {
            'USDTRY=X': 'usd_try',      # Dolar/TL kuru
            'GC=F': 'gold_usd',          # Altın vadeli (USD/oz)
            'SI=F': 'silver_usd',        # Gümüş vadeli (USD/oz)
            'XU100.IS': 'bist100',       # BIST 100
        }

        print(f"  {len(macro_tickers)} makro gösterge çekiliyor...")

        for ticker_sym, col_name in macro_tickers.items():
            try:
                t = yf.Ticker(ticker_sym)
                hist = t.history(start=start_date, end=end_date, interval='1d')
                if len(hist) == 0:
                    print(f"  [{ticker_sym}] Veri yok")
                    continue

                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)

                # Kapanış fiyatını ekle
                df[col_name] = hist['Close'].reindex(df.index)

                # Günlük değişim
                df[f'{col_name}_ret'] = df[col_name].pct_change()

                print(f"  [{ticker_sym}] -> {col_name}: {len(hist)} satır")
            except Exception as e:
                print(f"  [{ticker_sym}] Hata: {e}")

        # Gümüş/Altın oranı
        if 'silver_usd' in df.columns and 'gold_usd' in df.columns:
            df['gold_silver_ratio'] = df['gold_usd'] / (df['silver_usd'] + 1e-8)
            df['gold_silver_ratio_ret'] = df['gold_silver_ratio'].pct_change()

        # Kaydet
        df.to_csv(CSV_PATH, encoding='utf-8')
        print(f"  Makro veriler eklendi. Toplam kolon: {len(df.columns)}")
        return True

    except Exception as e:
        print(f"  Makro veri hatası: {e}")
        return False


if __name__ == '__main__':
    # 1. CSV güncelle
    success = update_csv()

    # 2. Makro veri ekle
    if success:
        fetch_macro_data()

    print("\nTamamlandı!")
