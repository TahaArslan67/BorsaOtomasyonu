"""
GMSTR Veri Yükleme ve Temizleme Modülü
- 5 yıllık areaxdatetime.csv (günlük, Net Getiri + Benchmark)
- Saatlik gercek_data.csv (2024-2026, OHLCV)
Sentetik OHLCV türetimi, veri kalitesi kontrolleri.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Literal
import warnings
warnings.filterwarnings('ignore')


class GMSTRDataLoader:
    """GMSTR.IS veri yükleme, doğrulama ve temizleme sınıfı."""

    REQUIRED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume'}

    def __init__(self, csv_path: Optional[str] = None,
                 source: Literal['auto', '5y', 'hourly'] = 'auto'):
        self.source = source
        self.csv_path = csv_path
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.benchmark_df: Optional[pd.Series] = None

        if csv_path is None:
            self.csv_path = self._auto_detect_source()

    def _auto_detect_source(self) -> str:
        """En uygun veri kaynağını otomatik seç."""
        candidates = [
            Path(__file__).parent.parent / 'claude' / 'areaxdatetime.csv',
            Path(__file__).parent.parent / 'claude' / 'gercek_data.csv',
            Path(__file__).parent.parent / 'claude' / 'gercek_data_5y_1d.csv',
            Path(__file__).parent.parent / 'claude' / 'gmstr_gunluk.csv',
        ]
        for c in candidates:
            if c.exists():
                print(f"[DataLoader] Otomatik veri kaynağı seçildi: {c.name}")
                return str(c)
        raise FileNotFoundError("Hiçbir veri dosyası bulunamadı!")

    def load(self) -> pd.DataFrame:
        """Ham CSV'yi yükler, temel yapısal düzeltmeleri uygular."""
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")

        if 'areaxdatetime' in path.name.lower():
            df = self._load_areaxdatetime(path)
        elif 'gercek_data' in path.name.lower():
            df = self._load_gercek_data(path)
        else:
            df = self._load_generic(path)

        self.raw_df = df.copy()
        print(f"[DataLoader] Ham veri yüklendi: {len(df)} satır | "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        return df

    def _load_areaxdatetime(self, path: Path) -> pd.DataFrame:
        """5 yıllık areaxdatetime.csv'den gerçek fiyat ölçeğinde OHLCV + benchmark türet."""
        raw = pd.read_csv(path)
        raw.columns = [c.strip() for c in raw.columns]

        # Tarih kolonunu bul: 'category', 'Unnamed: 0', 'Date' veya index
        date_col = None
        for candidate in ['category', 'Unnamed: 0', 'Date', 'date']:
            if candidate in raw.columns:
                date_col = candidate
                break

        if date_col is not None:
            # 'category' formatı: 'Mon Jan 01 2021', diğerleri ISO format
            if date_col == 'category':
                raw['Date'] = pd.to_datetime(raw[date_col], format='%a %b %d %Y', errors='coerce')
            else:
                raw['Date'] = pd.to_datetime(raw[date_col], errors='coerce')
            raw = raw.dropna(subset=['Date']).set_index('Date').sort_index()
        else:
            # Index zaten tarih olabilir
            raw.index = pd.to_datetime(raw.index, errors='coerce')
            raw = raw[raw.index.notna()].sort_index()

        fund_ret = pd.to_numeric(raw['Net Getiri'], errors='coerce')
        benchmark_ret = pd.to_numeric(raw['Karşılaştırma Ölçütü'], errors='coerce')

        # --- GERÇEK FİYAT ÖLÇEKLEMESİ ---
        # gercek_data.csv varsa, lineer regresyon ile ölçekle
        real_csv = Path(__file__).parent.parent / 'claude' / 'gercek_data.csv'
        if real_csv.exists():
            try:
                real_daily = self._load_gercek_data_daily(real_csv)
                # Timezone temizle
                if real_daily.index.tz is not None:
                    real_daily.index = real_daily.index.tz_localize(None)
                # Suffix ekle - raw'da zaten Open/High/Low/Close/Volume olabilir
                real_daily_renamed = real_daily[['Close']].rename(columns={'Close': 'Close_real'})
                common = raw.join(real_daily_renamed, how='inner')
                if len(common) >= 30:
                    x = common['Net Getiri'].astype(float).fillna(0).values
                    y = common['Close_real'].fillna(method='ffill').values
                    # NaN kontrolü
                    valid = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[valid], y[valid]
                    if len(x) >= 30 and np.var(x) > 0:
                        b_slope = np.cov(x, y)[0, 1] / np.var(x)
                        a_intercept = np.mean(y) - b_slope * np.mean(x)
                        close_price = a_intercept + b_slope * fund_ret
                        print(f"[DataLoader] Gerçek fiyat ölçeklemesi uygulandı: "
                              f"Close = {a_intercept:.2f} + {b_slope:.6f} * fund_ret")
                    else:
                        close_price = 100.0 * (1.0 + fund_ret / 100.0)
                else:
                    close_price = 100.0 * (1.0 + fund_ret / 100.0)
            except Exception as e:
                print(f"[DataLoader] Ölçekleme hatası ({e}), varsayılan sentetik fiyat kullanılıyor.")
                close_price = 100.0 * (1.0 + fund_ret / 100.0)
        else:
            close_price = 100.0 * (1.0 + fund_ret / 100.0)

        # Günlük log getiri
        log_price = np.log(close_price.replace(0, np.nan))
        daily_log_ret = log_price.diff()
        vol = daily_log_ret.rolling(window=20, min_periods=5).std().fillna(daily_log_ret.std())

        # Sentetik OHLCV (fiyat hareketini gerçekçi tutmak için volatilite bazlı)
        open_price = close_price.shift(1).fillna(close_price)
        high_price = close_price * np.exp(np.abs(daily_log_ret) * 0.5 + vol * 0.3)
        low_price = close_price * np.exp(-np.abs(daily_log_ret) * 0.5 - vol * 0.3)
        high_price = np.maximum(high_price, np.maximum(open_price, close_price))
        low_price = np.minimum(low_price, np.minimum(open_price, close_price))

        volume = (np.abs(daily_log_ret) * 1e6).fillna(0).astype(int)

        df = pd.DataFrame({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
            'Fund_Return': fund_ret,
            'Benchmark_Return': benchmark_ret,
        }).dropna()

        # Makro kolonları varsa ekle (update_gmstr_data.py tarafından eklendi)
        macro_cols = ['usd_try', 'usd_try_ret', 'gold_usd', 'gold_usd_ret',
                      'silver_usd', 'silver_usd_ret', 'bist100', 'bist100_ret',
                      'gold_silver_ratio', 'gold_silver_ratio_ret']
        for col in macro_cols:
            if col in raw.columns:
                df[col] = pd.to_numeric(raw[col], errors='coerce').reindex(df.index)

        self.benchmark_df = df['Benchmark_Return'].copy()
        return df

    def _load_gercek_data(self, path: Path) -> pd.DataFrame:
        """YFinance saatlik CSV formatını parse et, günlük resample yap."""
        raw = pd.read_csv(path, header=None)
        first_row = raw.iloc[0].astype(str).tolist()
        if 'Price' in first_row or 'Date' in first_row or 'Datetime' in first_row:
            header_row = raw.iloc[0].tolist()
            data = raw.iloc[3:].copy()
            data.columns = header_row
            date_col = None
            for c in data.columns:
                if str(c).lower() in ['price', 'date', 'datetime']:
                    date_col = c
                    break
            if date_col:
                data = data.rename(columns={date_col: 'Date'})
                data = data.set_index('Date')
        else:
            data = raw.copy()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data.set_index('Date')

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
        data = data[data.index.notna()]
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        data = data.sort_index()

        avg_bars_per_day = len(data) / data.index.normalize().nunique()
        if avg_bars_per_day > 1.5:
            print(f"[DataLoader] Saatlik veri tespit edildi ({avg_bars_per_day:.1f} bar/gün). "
                  f"Günlük OHLCV'ye dönüştürülüyor...")
            data = data.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        return data

    def _load_gercek_data_daily(self, path: Path) -> pd.DataFrame:
        """YFinance saatlik CSV'yi günlük OHLCV'ye dönüştür ve döndür."""
        raw = pd.read_csv(path, header=None)
        first_row = raw.iloc[0].astype(str).tolist()
        if 'Price' in first_row or 'Date' in first_row or 'Datetime' in first_row:
            header_row = raw.iloc[0].tolist()
            data = raw.iloc[3:].copy()
            data.columns = header_row
            date_col = None
            for c in data.columns:
                if str(c).lower() in ['price', 'date', 'datetime']:
                    date_col = c
                    break
            if date_col:
                data = data.rename(columns={date_col: 'Date'})
                data = data.set_index('Date')
        else:
            data = raw.copy()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data.set_index('Date')

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
        data = data[data.index.notna()]
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().sort_index()
        daily = data.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return daily

    def _load_generic(self, path: Path) -> pd.DataFrame:
        """Standart CSV yükleme."""
        data = pd.read_csv(path)
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.set_index('Date')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().sort_index()
        return data

    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Veri kalitesi kontrolleri ve temizleme."""
        df = df.copy() if df is not None else self.raw_df.copy()
        if df is None:
            df = self.load()

        initial_len = len(df)
        issues = []

        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            bad = (df[col] <= 0) | df[col].isna()
            if bad.any():
                issues.append(f"{col}: {bad.sum()} geçersiz değer")
                df = df[~bad]

        hl_invalid = df['High'] < df['Low']
        if hl_invalid.any():
            issues.append(f"High<Low: {hl_invalid.sum()} satır")
            df = df[~hl_invalid]

        ohlc_invalid = (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        if ohlc_invalid.any():
            issues.append(f"Close High/Low dışı: {ohlc_invalid.sum()} satır")
            df.loc[ohlc_invalid, 'Close'] = df.loc[ohlc_invalid, price_cols].median(axis=1)

        vol_invalid = (df['Volume'] < 0)
        if vol_invalid.any():
            issues.append(f"Negatif volume: {vol_invalid.sum()} satır")
            df = df[~vol_invalid]

        if 'Volume' in df.columns and df['Volume'].max() > 0:
            q1, q3 = df['Volume'].quantile([0.01, 0.99])
            iqr = q3 - q1
            vol_outliers = (df['Volume'] > q3 + 5 * iqr) | (df['Volume'] < q1 - 5 * iqr)
            if vol_outliers.any():
                issues.append(f"Volume outlier: {vol_outliers.sum()} satır")
                df.loc[vol_outliers, 'Volume'] = df['Volume'].median()

        dups = df.index.duplicated().sum()
        if dups:
            issues.append(f"Tekrarlayan tarih: {dups} satır")
            df = df[~df.index.duplicated(keep='first')]

        dup_prices = df['Close'].eq(df['Close'].shift())
        dup_runs = dup_prices.groupby((~dup_prices).cumsum()).transform('sum')
        flat_mask = dup_runs >= 20
        if flat_mask.any():
            issues.append(f"Sabit fiyat bölgesi: {flat_mask.sum()} satır (>=20 gün aynı)")
            df = df[~flat_mask]

        returns = df['Close'].pct_change().abs()
        jumps = returns > 0.20
        if jumps.any():
            issues.append(f"Aşırı fiyat sıçraması (>%20): {jumps.sum()} satır")
            for idx in df[jumps].index:
                window = df.loc[:idx].tail(5)
                if len(window) >= 3:
                    median_price = window['Close'].median()
                    df.loc[idx, price_cols] = median_price

        all_bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())
        missing = all_bdays.difference(df.index)
        if len(missing) > 0:
            issues.append(f"Eksik işlem günü (tatil): {len(missing)} gün")

        # Sadece OHLCV kolonlarında NaN varsa sil (makro kolonlar NaN olabilir)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.clean_df = df.copy()

        print(f"[DataLoader] Temizlik tamamlandı: {initial_len} → {len(df)} satır")
        if issues:
            for issue in issues:
                print(f"  ⚠ {issue}")
        else:
            print("  ✓ Veri kalitesi sorunu tespit edilmedi")

        return df

    def validate_quality(self, df: pd.DataFrame) -> dict:
        """Veri kalitesi metrikleri."""
        metrics = {
            'total_rows': len(df),
            'date_range': (df.index.min().date(), df.index.max().date()),
            'missing_values': df.isna().sum().sum(),
            'zero_volume_days': (df['Volume'] == 0).sum() if 'Volume' in df.columns else 0,
            'price_range': (df['Close'].min(), df['Close'].max()),
            'avg_daily_return': df['Close'].pct_change().mean(),
            'daily_volatility': df['Close'].pct_change().std(),
            'max_single_day_move': df['Close'].pct_change().abs().max(),
        }
        return metrics

    def get_train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Zaman serisi uyumlu train/test bölme."""
        split_idx = int(len(df) * (1 - test_ratio))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        print(f"[DataLoader] Train/Test: {len(train)} / {len(test)} "
              f"({(1 - test_ratio) * 100:.0f}%/{test_ratio * 100:.0f}%)")
        return train, test


def load_and_prepare(csv_path: Optional[str] = None,
                     source: Literal['auto', '5y', 'hourly'] = 'auto') -> pd.DataFrame:
    """Kolay kullanım fonksiyonu: yükle, temizle, döndür."""
    loader = GMSTRDataLoader(csv_path, source=source)
    loader.load()
    df = loader.clean()
    quality = loader.validate_quality(df)
    print(f"[DataLoader] Kalite raporu:")
    for k, v in quality.items():
        print(f"  • {k}: {v}")
    return df
