"""
GMSTR Tahmin Motoru
Güncel veriden çok vadeli, çok frekanslı tahminler üretir.
"""
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json

from .data_loader import GMSTRDataLoader
from .features import FeatureEngineer
from .models import SimpleEnsemble
from .price_fetcher import GMSTRPriceFetcher


class GMSTRPredictor:
    """GMSTR için gerçek zamanlı tahmin motoru."""

    def __init__(self, model_dir: str = 'gmstr_models',
                 daily_csv: str = None, hourly_csv: str = None):
        self.model_dir = Path(model_dir)
        # PyInstaller desteği
        base = Path.cwd()
        if hasattr(sys, '_MEIPASS'):
            base = Path(sys._MEIPASS)

        self.daily_csv = daily_csv or str(base / 'claude' / 'areaxdatetime.csv')
        self.hourly_csv = hourly_csv or str(base / 'claude' / 'gercek_data.csv')
        self.engineer = FeatureEngineer()
        self.models: Dict[str, SimpleEnsemble] = {}
        self.feature_cols: List[str] = []
        self.horizons: Dict[str, int] = {}
        self.is_ready = False

    def load_system(self):
        """Tüm modelleri ve metadata'yı yükle."""
        print("[Predictor] Sistem yükleniyor...")

        feat_path = self.model_dir / 'feature_columns.json'
        if feat_path.exists():
            with open(feat_path) as f:
                self.feature_cols = json.load(f)
        else:
            self.feature_cols = []

        results_path = self.model_dir / 'training_results.json'
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            self.horizons = {h: r['bars'] for h, r in results.items()}
        else:
            self.horizons = {
                '1d_daily': 1, '3d_daily': 3, '5d_daily': 5, '10d_daily': 10,
                '1h_hourly': 1, '4h_hourly': 4,
            }

        for h_name in self.horizons.keys():
            model_path = self.model_dir / f'simple_{h_name}.pkl'
            if model_path.exists():
                self.models[h_name] = SimpleEnsemble.load(str(model_path))
                print(f"  ✓ Model yüklendi: {h_name}")
            else:
                print(f"  ⚠ Model bulunamadı: {h_name}")

        self.is_ready = len(self.models) > 0
        if self.is_ready:
            print(f"[Predictor] Sistem hazır. {len(self.models)} model yüklendi.")
        else:
            print(f"[Predictor] ⚠ Hiç model yüklenemedi!")
        return self.is_ready

    def predict(self, df_raw: Optional[pd.DataFrame] = None,
                confidence_threshold: float = 0.60,
                is_hourly: bool = False) -> Dict[str, Dict]:
        """Son veriden tahmin üret."""
        if not self.is_ready:
            raise RuntimeError("Sistem hazır değil. load_system() çağırın.")

        # Veri yükle
        if df_raw is None:
            if is_hourly:
                csv = self.hourly_csv or str(
                    Path(__file__).parent.parent / 'claude' / 'gercek_data.csv'
                )
                df_raw = self._load_hourly_raw(csv)
            else:
                csv = self.daily_csv or str(
                    Path(__file__).parent.parent / 'claude' / 'areaxdatetime.csv'
                )
                loader = GMSTRDataLoader(csv)
                loader.load()
                df_raw = loader.clean()

        # Günlük tahminlerde de gerçek son fiyatı gercek_data.csv'den al
        real_current_price = None
        if not is_hourly:
            try:
                real_csv = Path(__file__).parent.parent / 'claude' / 'gercek_data.csv'
                if real_csv.exists():
                    real_df = self._load_hourly_raw(str(real_csv))
                    if len(real_df) > 0:
                        # Saatlikten günlük son kapanışı al
                        real_daily = real_df.resample('D').agg({
                            'Open': 'first', 'High': 'max', 'Low': 'min',
                            'Close': 'last', 'Volume': 'sum'
                        }).dropna()
                        if len(real_daily) > 0:
                            real_current_price = float(real_daily['Close'].iloc[-1])
            except Exception:
                pass

        df = self.engineer.transform(df_raw.tail(300))

        # Sadece veride var olan feature kolonlarını kullan
        available_features = [c for c in df.columns if c not in
                              {'Open', 'High', 'Low', 'Close', 'Volume',
                               'Fund_Return', 'Benchmark_Return',
                               'future_ret_1d', 'future_ret_3d', 'future_ret_5d',
                               'future_ret_10d', 'future_ret_1h', 'future_ret_4h'}]
        clean = df.dropna(subset=available_features)
        if len(clean) == 0:
            raise ValueError("Yeterli veri yok (NaN içeren özellikler).")

        last = clean.iloc[[-1]]
        current_price = real_current_price if real_current_price is not None else float(df_raw['Close'].iloc[-1])
        current_date = df_raw.index[-1]

        # CANLI FİYAT - API'den çek, her zaman güncel olsun
        try:
            live = GMSTRPriceFetcher.get_price_with_fallback()
            if live['price'] is not None:
                current_price = float(live['price'])
                current_date = live['timestamp']
        except Exception:
            pass

        predictions = {}
        for h_name, bars in self.horizons.items():
            if h_name not in self.models:
                continue
            if is_hourly and 'hourly' not in h_name:
                continue
            if not is_hourly and 'hourly' in h_name:
                continue

            model = self.models[h_name]

            # Modelin eğitim sırasında gördüğü feature isimlerini al
            if hasattr(model, 'feature_cols') and model.feature_cols:
                expected = model.feature_cols
            elif hasattr(model.scaler, 'feature_names_in_'):
                expected = list(model.scaler.feature_names_in_)
            else:
                expected = available_features

            # Eksik/feature hizalaması: modelin beklediği isimlere göre DataFrame oluştur
            X_model = pd.DataFrame(0.0, index=last.index, columns=expected)
            for col in expected:
                if col in available_features:
                    X_model[col] = last[col].values
                else:
                    X_model[col] = 0.0

            proba = float(model.predict_proba(X_model)[0])
            confidence = max(proba, 1 - proba)
            direction = 'YUKARI ↑' if proba > 0.5 else 'AŞAĞI ↓'

            base_votes = {}

            hist_vol = df['log_ret'].std() * np.sqrt(252 if not is_hourly else 252 * 6.5)
            expected_move = hist_vol * np.sqrt(bars / (252 if not is_hourly else 252 * 6.5)) * (proba - 0.5) * 2
            pred_price = current_price * (1 + expected_move)

            signal_strength = 'GÜÇLÜ' if confidence >= confidence_threshold else 'ZAYIF'
            if confidence >= 0.70:
                signal_strength = 'ÇOK GÜÇLÜ'

            predictions[h_name] = {
                'direction': direction,
                'prob_up': round(proba, 4),
                'prob_down': round(1 - proba, 4),
                'confidence': round(confidence, 4),
                'signal_strength': signal_strength,
                'current_price': round(current_price, 2),
                'predicted_price': round(pred_price, 2),
                'expected_change_pct': round(expected_move * 100, 2),
                'base_votes': base_votes,
                'date': str(current_date),
            }
        return predictions

    @staticmethod
    def _load_hourly_raw(csv_path: str) -> pd.DataFrame:
        """gercek_data.csv'yi saatlik olarak yükle (günlük resample olmadan)."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Saatlik veri bulunamadı: {path}")

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

        for col in ['Open', 'High', 'Low', 'Close']:
            data = data[data[col] > 0]
        data = data[data['High'] >= data['Low']]

        return data

    def _get_base_votes(self, model: SimpleEnsemble, X: pd.DataFrame) -> Dict[str, float]:
        votes = {}
        if not hasattr(model, 'models') or not model.models:
            return votes
        X_s = model.scaler.transform(X)
        for name, mdl in model.models.items():
            try:
                p = float(mdl.predict_proba(X_s)[0, 1])
                votes[name] = round(p, 3)
            except Exception:
                votes[name] = None
        return votes

    def get_recommendation(self, predictions: Dict[str, Dict],
                           min_confidence: float = 0.60) -> str:
        strong_up = []
        strong_down = []
        mixed = []

        for h_name, pred in predictions.items():
            conf = pred['confidence']
            if conf >= min_confidence:
                if 'YUKARI' in pred['direction']:
                    strong_up.append(h_name)
                else:
                    strong_down.append(h_name)
            else:
                mixed.append(h_name)

        lines = []
        lines.append("\n" + "="*60)
        lines.append("GMSTR KARAR DESTEK ÖZETİ")
        lines.append("="*60)

        if strong_up and not strong_down:
            lines.append(f"🟢 BÜYÜK ÇOĞUNLUK YUKARI ({', '.join(strong_up)})")
            lines.append(f"   Güven: {min(predictions[h]['confidence'] for h in strong_up):.0%}+")
        elif strong_down and not strong_up:
            lines.append(f"🔴 BÜYÜK ÇOĞUNLUK AŞAĞI ({', '.join(strong_down)})")
            lines.append(f"   Güven: {min(predictions[h]['confidence'] for h in strong_down):.0%}+")
        elif strong_up and strong_down:
            lines.append(f"⚠ ÇELİŞKILI SİNYALLER")
            lines.append(f"   Yukarı: {', '.join(strong_up)}")
            lines.append(f"   Aşağı: {', '.join(strong_down)}")
        else:
            lines.append(f"⚪ BELİRSİZ (Güven yetersiz)")

        if mixed:
            lines.append(f"   Belirsiz vadeler: {', '.join(mixed)}")

        lines.append("="*60)
        return "\n".join(lines)

    def print_predictions(self, predictions: Dict[str, Dict]):
        print("\n" + "="*70)
        print("GMSTR GÜNCEL TAHMİNLER")
        print("="*70)
        print(f"Tarih: {list(predictions.values())[0]['date']}")
        print(f"Son Fiyat: {list(predictions.values())[0]['current_price']:.2f} TRY")
        print("-"*70)

        for h_name, pred in predictions.items():
            emoji = "🟢" if 'YUKARI' in pred['direction'] else "🔴"
            bar = "█" * int(pred['confidence'] * 10)
            base_str = ", ".join([f"{k}={v:.2f}" for k, v in pred['base_votes'].items()])

            print(f"\n{emoji} {h_name:>12} | {pred['direction']:<10} | "
                  f"Güven: {pred['confidence']:.0%} {bar:<10}")
            print(f"       → Tahmini: {pred['predicted_price']:.2f} TRY "
                  f"({pred['expected_change_pct']:+.2f}%)")
            print(f"       Base Votes: {base_str}")
            print(f"       Sinyal: {pred['signal_strength']}")

        print("="*70)
