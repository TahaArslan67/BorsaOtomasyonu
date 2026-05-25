"""
GMSTR Model Eğitim Modülü
- Günlük vadeler: 1d, 3d, 5d, 10d (areaxdatetime.csv - 5 yıl)
- Saatlik vadeler: 1h, 4h (gercek_data.csv - 2 yıl saatlik)
- Target redesign: binary, dynamic, 3-class
- Walk-forward validasyon, feature importance, model kaydetme.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Literal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from .models import SimpleEnsemble
from .features import FeatureEngineer, create_targets, create_targets_dynamic, create_targets_3class
from .data_loader import GMSTRDataLoader


class ModelTrainer:
    """GMSTR için çok vadeli, çok frekanslı model eğitim koordinatörü."""

    HORIZONS_DAILY = {
        '1d': 1,
        '3d': 3,
        '5d': 5,
        '10d': 10,
    }

    HORIZONS_HOURLY = {
        '1h': 1,
        '4h': 4,
    }

    def __init__(self, model_dir: str = 'gmstr_models',
                 target_mode: Literal['binary', 'dynamic', '3class'] = 'dynamic',
                 use_mlp: bool = True):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.engineer = FeatureEngineer()
        self.results: Dict[str, Dict] = {}
        self.models: Dict[str, SimpleEnsemble] = {}
        self.feature_cols: List[str] = []
        self.target_mode = target_mode
        self.use_mlp = use_mlp

    def prepare_data(self, df: pd.DataFrame, horizons: Dict[str, int],
                     is_hourly: bool = False) -> pd.DataFrame:
        print(f"\n[Trainer] Özellikler hesaplanıyor ({'saatlik' if is_hourly else 'günlük'})...")
        df = self.engineer.transform(df)

        if self.target_mode == 'dynamic':
            df = create_targets_dynamic(df, horizons, vol_window=20, multiplier=0.5)
            # Dinamik hedef kolonlarını 'target_' prefixiyle standart hale getir
            for h in horizons.keys():
                df[f'target_{h}'] = df[f'target_dyn_{h}']
        elif self.target_mode == '3class':
            df = create_targets_3class(df, horizons, z_thresh=0.5)
            for h in horizons.keys():
                df[f'target_{h}'] = df[f'target_3c_{h}']
        else:
            df = create_targets(df, horizons, threshold=0.0)

        self.feature_cols = self.engineer.get_feature_columns(df)
        print(f"  • Toplam özellik: {len(self.feature_cols)}")
        return df

    def train_daily_models(self, csv_path: str = None) -> Dict[str, Dict]:
        """5 yıllık günlük veriyle (areaxdatetime) günlük modeller eğit."""
        loader = GMSTRDataLoader(csv_path)
        if csv_path is None:
            loader.csv_path = str(
                Path(__file__).parent.parent / 'claude' / 'areaxdatetime.csv'
            )
        loader.load()
        df = loader.clean()

        print(f"\n{'='*70}")
        print("GMSTR GÜNLÜK MODEL EĞİTİMİ (5 Yıllık Veri)")
        print(f"{'='*70}")

        return self._train_horizons(df, self.HORIZONS_DAILY, is_hourly=False)

    def train_hourly_models(self, csv_path: str = None) -> Dict[str, Dict]:
        """Saatlik veriyle (gercek_data) 1h/4h modeller eğit."""
        if csv_path is None:
            csv_path = str(Path(__file__).parent.parent / 'claude' / 'gercek_data.csv')

        # Saatlik veriyi özel yükle (günlük resample yapmadan)
        df = self._load_hourly_raw(csv_path)
        if df is None or len(df) < 500:
            print("[Trainer] ⚠ Saatlik veri yetersiz, saatlik modeller atlanıyor.")
            return {}

        print(f"\n{'='*70}")
        print("GMSTR SAATLİK MODEL EĞİTİMİ")
        print(f"{'='*70}")

        return self._train_horizons(df, self.HORIZONS_HOURLY, is_hourly=True)

    def _load_hourly_raw(self, csv_path: str) -> pd.DataFrame:
        """gercek_data.csv'yi saatlik olarak yükle (günlük resample olmadan)."""
        path = Path(csv_path)
        if not path.exists():
            return None

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

        # Temel temizlik
        for col in ['Open', 'High', 'Low', 'Close']:
            data = data[data[col] > 0]
        data = data[data['High'] >= data['Low']]

        print(f"[Trainer] Saatlik veri yüklendi: {len(data)} satır | "
              f"{data.index[0]} → {data.index[-1]}")
        return data

    def _train_horizons(self, df: pd.DataFrame, horizons: Dict[str, int],
                        is_hourly: bool = False) -> Dict[str, Dict]:
        """Belirtilen horizonlar için model eğitimi."""
        df = self.prepare_data(df, horizons, is_hourly)
        freq_label = 'hourly' if is_hourly else 'daily'

        for h_name, bars in horizons.items():
            print(f"\n{'─'*70}")
            print(f"[VADE: {h_name}] ({bars} bar sonrası)")
            print(f"{'─'*70}")

            target_col = f'target_{h_name}'
            clean = df.dropna(subset=self.feature_cols + [target_col]).copy()

            if len(clean) < 200:
                print(f"  ⚠ Yetersiz veri: {len(clean)} satır, atlanıyor.")
                continue

            X = clean[self.feature_cols]
            y = clean[target_col]

            # Zaman serisi split (son %20 test)
            split_idx = int(len(X) * 0.80)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            print(f"  • Eğitim: {len(X_train)} | Test: {len(X_test)} | "
                  f"Pozitif: {y_train.mean():.1%}")

            # 3-class ise class dağılımını raporla
            if self.target_mode == '3class':
                vc = y_train.value_counts().sort_index()
                print(f"  • Sınıf dağılımı: {dict(vc)}")

            print(f"  • Walk-forward CV yapılıyor...")
            cv_acc, cv_std, cv_auc = self._quick_cv(X_train, y_train)
            print(f"    CV Sonuç: Acc={cv_acc:.2%} ± {cv_std:.2%} | AUC={cv_auc:.3f}")

            print(f"  • Feature selection (top 30)...")
            selected = FeatureEngineer.select_top_features(X_train, y_train, k=30)
            print(f"    Secilen feature: {len(selected)} / {len(self.feature_cols)}")
            X_train = X_train[selected]
            X_test = X_test[selected]
            self.feature_cols = selected

            print(f"  • Simple Ensemble eğitiliyor...")
            model = SimpleEnsemble(use_mlp=self.use_mlp)
            model.fit(X_train, y_train)

            # Test degerlendirmesi
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
            proba_test = model.predict_proba(X_test)
            preds_test = (proba_test > 0.5).astype(int)
            test_acc = accuracy_score(y_test, preds_test)
            test_auc = roc_auc_score(y_test, proba_test)
            test_precision = precision_score(y_test, preds_test, zero_division=0)
            test_recall = recall_score(y_test, preds_test, zero_division=0)
            metrics = {
                'test_accuracy': test_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
            }
            print(f"    Test Acc={test_acc:.2%} | AUC={test_auc:.3f}")

            suffix = f"_{freq_label}"
            model_path = self.model_dir / f'simple_{h_name}{suffix}.pkl'
            model.save(str(model_path))

            feat_imp = self._get_feature_importance(model, X_train)

            result = {
                'horizon': h_name,
                'bars': bars,
                'frequency': freq_label,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'positive_rate': round(y_train.mean(), 4),
                'cv_accuracy': round(cv_acc, 4),
                'cv_std': round(cv_std, 4),
                'cv_auc': round(cv_auc, 4),
                **{k: round(v, 4) if v is not None else None for k, v in metrics.items()},
                'top_features': feat_imp,
            }

            key = f'{h_name}{suffix}'
            self.results[key] = result
            self.models[key] = model

            print(f"\n  {'━'*50}")
            print(f"  SONUÇLAR [{h_name}]")
            print(f"  {'━'*50}")
            print(f"  CV Doğruluk:      {cv_acc:.2%} ± {cv_std:.2%}")
            if 'test_accuracy' in metrics:
                print(f"  Test Doğruluk:    {metrics['test_accuracy']:.2%}")
                print(f"  Test AUC:         {metrics['test_auc']:.3f}")
            if 'hc_60_acc' in metrics and metrics['hc_60_acc'] is not None:
                print(f"  Yüksek Güven 60%: {metrics['hc_60_acc']:.2%} "
                      f"({metrics['hc_60_ratio']:.1%} veri)")
            if 'hc_70_acc' in metrics and metrics['hc_70_acc'] is not None:
                print(f"  Yüksek Güven 70%: {metrics['hc_70_acc']:.2%} "
                      f"({metrics['hc_70_ratio']:.1%} veri)")

        # Sonuçları kaydet
        results_path = self.model_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n[Trainer] Sonuçlar kaydedildi: {results_path}")

        feat_path = self.model_dir / 'feature_columns.json'
        with open(feat_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_cols, f, indent=2)

        return self.results

    def _quick_cv(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[float, float, float]:
        """Hızlı walk-forward CV (XGB+LGB ortalaması)."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs, aucs = [], []

        for tr_idx, val_idx in tscv.split(X):
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X.iloc[tr_idx])
            X_val = scaler.transform(X.iloc[val_idx])

            import xgboost as xgb
            import lightgbm as lgb

            m1 = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08,
                                   subsample=0.8, colsample_bytree=0.7,
                                   eval_metric='logloss', use_label_encoder=False,
                                   random_state=42, n_jobs=-1)
            m2 = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.08,
                                    subsample=0.8, colsample_bytree=0.7,
                                    random_state=42, n_jobs=-1, verbose=-1)

            m1.fit(X_tr, y.iloc[tr_idx])
            m2.fit(X_tr, y.iloc[tr_idx])

            p1 = m1.predict_proba(X_val)[:, 1]
            p2 = m2.predict_proba(X_val)[:, 1]
            avg_p = (p1 + p2) / 2
            preds = (avg_p > 0.5).astype(int)

            accs.append(accuracy_score(y.iloc[val_idx], preds))
            try:
                aucs.append(roc_auc_score(y.iloc[val_idx], avg_p))
            except Exception:
                aucs.append(0.5)

        return np.mean(accs), np.std(accs), np.mean(aucs)

    def _get_feature_importance(self, model, X_train: pd.DataFrame, top_n: int = 10) -> Dict[str, float]:
        """XGB feature importance al."""
        try:
            if hasattr(model, 'models') and model.models:
                for name, mdl in model.models.items():
                    if name == 'xgb' and hasattr(mdl, 'feature_importances_'):
                        imp = pd.Series(mdl.feature_importances_, index=X_train.columns)
                        return imp.sort_values(ascending=False).head(top_n).to_dict()
        except Exception:
            pass
        return {}

    def load_models(self) -> Dict[str, SimpleEnsemble]:
        """Dizindeki tüm modelleri yükle."""
        for pkl_path in self.model_dir.glob('simple_*.pkl'):
            h_name = pkl_path.stem.replace('simple_', '')
            try:
                self.models[h_name] = SimpleEnsemble.load(str(pkl_path))
                print(f"[Trainer] Model yüklendi: {h_name}")
            except Exception as e:
                print(f"[Trainer] ⚠ Model yüklenemedi {h_name}: {e}")

        feat_path = self.model_dir / 'feature_columns.json'
        if feat_path.exists():
            with open(feat_path) as f:
                self.feature_cols = json.load(f)

        return self.models

    def train_all(self, daily_csv: str = None, hourly_csv: str = None) -> Dict[str, Dict]:
        """Hem günlük hem saatlik modelleri eğit."""
        self.train_daily_models(daily_csv)
        self.train_hourly_models(hourly_csv)
        return self.results
