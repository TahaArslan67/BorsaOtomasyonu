"""
GMSTR Ensemble Model Mimarisi
XGBoost + LightGBM + RandomForest + GradientBoosting + ExtraTrees + MLP
-> Stacked Meta-Learner (LogisticRegression)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path


class BaseModelFactory:
    """Temel model fabrikası - her algoritma için optimize edilmiş parametreler."""

    @staticmethod
    def build_xgb(n_estimators: int = 300, max_depth: int = 5) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.15,
            reg_lambda=1.2,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=2,
            tree_method='hist',
        )

    @staticmethod
    def build_lgb(n_estimators: int = 300, max_depth: int = 5) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=10,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )

    @staticmethod
    def build_rf(n_estimators: int = 200, max_depth: int = 8) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=2,
        )

    @staticmethod
    def build_gb(n_estimators: int = 150, max_depth: int = 4) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )

    @staticmethod
    def build_et(n_estimators: int = 150, max_depth: int = 8) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=43,
            n_jobs=2,
        )

    @staticmethod
    def build_mlp(hidden_layers: Tuple[int, ...] = (128, 64, 32),
                  alpha: float = 0.001,
                  max_iter: int = 500) -> MLPClassifier:
        """
        Çok katmanlı perceptron (MLP) sinir ağı.
        alpha = L2 regularizasyon katsayısı (overfitting kontrolü)
        early_stopping = validasyon seti üzerinde durma
        """
        return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=max_iter,
            shuffle=False,  # Zaman serisi -> shuffle kapalı
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4,
        )


class StackedEnsemble:
    """
    2-seviyeli stacked ensemble:
    Seviye 1: XGB, LGB, RF, GB, ET, MLP (base modeller)
    Seviye 2: LogisticRegression (meta-learner)
    """

    def __init__(self, calibrate: bool = True, n_splits: int = 4,
                 use_mlp: bool = True):
        self.calibrate = calibrate
        self.n_splits = n_splits
        self.use_mlp = use_mlp
        self.scaler = RobustScaler()
        self.base_models: List[Tuple[str, Any]] = []
        self.meta_learner = LogisticRegression(C=0.5, random_state=42, max_iter=1000, class_weight='balanced')
        self.base_names: List[str] = []
        self.is_fitted = False

    def _get_base_models(self, y_train: pd.Series = None) -> List[Tuple[str, Any]]:
        factory = BaseModelFactory()
        # Sınıf dengesizliği varsa scale_pos_weight hesapla
        spw = None
        if y_train is not None:
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            if n_pos > 0:
                spw = n_neg / n_pos

        # Sınıf dengesizliği düzeltmesi KALDIRILDI
        # Model verinin kendi dağılımından öğrensin
        spw_clipped = None

        xgb_model = factory.build_xgb(n_estimators=300, max_depth=4)

        lgb_model = factory.build_lgb(n_estimators=300, max_depth=4)

        rf_model = factory.build_rf(n_estimators=200, max_depth=10)
        # class_weight kaldırıldı - model kendi öğrensin

        gb_model = factory.build_gb(n_estimators=150, max_depth=3)

        et_model = factory.build_et(n_estimators=200, max_depth=10)
        # class_weight kaldırıldı

        models = [
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('gb', gb_model),
            ('et', et_model),
        ]

        if self.use_mlp:
            mlp_model = factory.build_mlp(hidden_layers=(128, 64, 32), alpha=0.001)
            models.append(('mlp', mlp_model))

        return models

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Ensemble'ı eğit. TimeSeriesSplit ile OOF (out-of-fold) meta-feature üret.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_s = self.scaler.transform(X_test)
        else:
            X_test_s = None

        self.base_models = self._get_base_models(y_train)
        self.base_names = [name for name, _ in self.base_models]

        n_train = len(X_train_s)
        n_models = len(self.base_models)

        # OOF predictions (meta-learner için eğitim verisi)
        oof_preds = np.zeros((n_train, n_models))

        print(f"    [StackedEnsemble] {n_models} base model eğitiliyor ({self.n_splits}-fold TS-CV)...")

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        fitted_base_models = []
        for j, (name, model) in enumerate(self.base_models):
            oof = np.zeros(n_train)
            fold_aucs = []

            for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_s)):
                # Modeli kopyala
                clone_model = self._clone_model(model)
                clone_model.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])

                probas = clone_model.predict_proba(X_train_s[val_idx])[:, 1]
                oof[val_idx] = probas

                try:
                    fold_auc = roc_auc_score(y_train.iloc[val_idx], probas)
                    fold_aucs.append(fold_auc)
                except Exception:
                    fold_auc = 0.5

            oof_preds[:, j] = oof

            # Final: tüm train üzerinde tekrar eğit
            final_model = self._clone_model(model)
            final_model.fit(X_train_s, y_train)

            # Kalibrasyon
            if self.calibrate and hasattr(final_model, 'predict_proba'):
                try:
                    final_model = CalibratedClassifierCV(final_model, cv=3, method='sigmoid')
                    final_model.fit(X_train_s, y_train)
                except Exception as e:
                    print(f"      Kalibrasyon atlandı ({name}): {e}")

            fitted_base_models.append(final_model)

            if fold_aucs:
                print(f"      {name.upper()}: AUC={np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

        self.base_models = [(name, m) for name, m in zip(self.base_names, fitted_base_models)]

        # Meta-learner eğitimi (OOF üzerinden)
        print(f"    [StackedEnsemble] Meta-learner eğitiliyor...")
        self.meta_learner.fit(oof_preds, y_train)
        self.is_fitted = True

        # Eğitim performansı
        train_meta_p = self.meta_learner.predict_proba(oof_preds)[:, 1]
        train_acc = accuracy_score(y_train, (train_meta_p > 0.5).astype(int))
        train_auc = roc_auc_score(y_train, train_meta_p)

        results = {
            'train_accuracy': train_acc,
            'train_auc': train_auc,
        }

        # Test değerlendirmesi
        if X_test_s is not None and y_test is not None:
            test_preds = self.predict_proba(X_test, use_base_only=False)
            test_acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
            test_auc = roc_auc_score(y_test, test_preds)
            test_precision = precision_score(y_test, (test_preds > 0.5).astype(int), zero_division=0)
            test_recall = recall_score(y_test, (test_preds > 0.5).astype(int), zero_division=0)

            # High confidence analizi
            hc_results = self._analyze_high_confidence(y_test, test_preds)

            results.update({
                'test_accuracy': test_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                **hc_results,
            })

            hc_acc = hc_results.get('hc_60_acc') or 0
            hc_ratio = hc_results.get('hc_60_ratio') or 0
            print(f"    [StackedEnsemble] Test Acc={test_acc:.2%} | AUC={test_auc:.3f} | "
                  f"HC_Acc={hc_acc:.2%}@{hc_ratio:.1%}")

        self.is_fitted = True
        return results

    def predict_proba(self, X: pd.DataFrame, use_base_only: bool = False) -> np.ndarray:
        """Tahmin olasılıkları (yukarı sınıfı)."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmedi. fit() çağırın.")

        X_s = self.scaler.transform(X)
        base_probas = np.zeros((len(X_s), len(self.base_models)))

        for j, (name, model) in enumerate(self.base_models):
            base_probas[:, j] = model.predict_proba(X_s)[:, 1]

        if use_base_only:
            return base_probas.mean(axis=1)

        return self.meta_learner.predict_proba(base_probas)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Sınıf tahmini (0 veya 1)."""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

    def _analyze_high_confidence(self, y_true: pd.Series, probas: np.ndarray) -> Dict[str, float]:
        """Farklı güven eşiklerinde doğruluk analizi."""
        from sklearn.metrics import accuracy_score
        results = {}
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]

        for thresh in thresholds:
            mask = (probas > thresh) | (probas < (1 - thresh))
            if mask.sum() > 10:
                preds = (probas[mask] > 0.5).astype(int)
                acc = accuracy_score(y_true[mask], preds)
                results[f'hc_{int(thresh*100)}_acc'] = acc
                results[f'hc_{int(thresh*100)}_ratio'] = mask.mean()
            else:
                results[f'hc_{int(thresh*100)}_acc'] = None
                results[f'hc_{int(thresh*100)}_ratio'] = mask.mean()

        return results

    def _clone_model(self, model):
        """Modelin aynı parametrelerle yeni bir kopyasını oluştur."""
        return type(model)(**model.get_params())

    def save(self, path: str):
        """Modeli kaydet."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'base_names': self.base_names,
            'is_fitted': self.is_fitted,
            'use_mlp': self.use_mlp,
        }, path)
        print(f"    [StackedEnsemble] Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str):
        """Modeli yükle."""
        data = joblib.load(path)
        ensemble = cls(calibrate=False, use_mlp=data.get('use_mlp', True))
        ensemble.scaler = data['scaler']
        ensemble.base_models = data['base_models']
        ensemble.meta_learner = data['meta_learner']
        ensemble.base_names = data['base_names']
        ensemble.is_fitted = data['is_fitted']
        return ensemble


class SimpleEnsemble:
    """Hızlı ensemble (stacking olmadan, ağırlıklı oylama)."""

    def __init__(self, weights: Dict[str, float] = None, use_mlp: bool = True):
        self.weights = weights or {'xgb': 3.0, 'lgb': 3.0, 'rf': 2.0, 'gb': 2.0, 'mlp': 2.0}
        self.use_mlp = use_mlp
        self.scaler = RobustScaler()
        self.models: Dict[str, Any] = {}
        self.is_fitted = False
        self.feature_cols: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        factory = BaseModelFactory()
        self.feature_cols = list(X_train.columns)
        X_train_s = self.scaler.fit_transform(X_train)

        model_map = {
            'xgb': factory.build_xgb(n_estimators=200),
            'lgb': factory.build_lgb(n_estimators=200),
            'rf': factory.build_rf(n_estimators=150),
            'gb': factory.build_gb(n_estimators=100),
        }
        if self.use_mlp:
            model_map['mlp'] = factory.build_mlp(hidden_layers=(64, 32), max_iter=300)

        for name in self.weights.keys():
            if name in model_map:
                print(f"      {name.upper()} eğitiliyor...")
                model = model_map[name]
                model.fit(X_train_s, y_train)
                self.models[name] = model

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model eğitilmedi.")

        X_s = self.scaler.transform(X)
        weighted_sum = np.zeros(len(X_s))
        total_weight = 0.0

        for name, model in self.models.items():
            w = self.weights.get(name, 1.0)
            probas = model.predict_proba(X_s)[:, 1]
            weighted_sum += w * probas
            total_weight += w

        return weighted_sum / total_weight

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) > threshold).astype(int)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'models': self.models,
            'weights': self.weights,
            'feature_cols': self.feature_cols,
            'is_fitted': self.is_fitted,
            'use_mlp': self.use_mlp,
        }, path)
        print(f"    [SimpleEnsemble] Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        ensemble = cls(weights=data.get('weights'), use_mlp=data.get('use_mlp', True))
        ensemble.scaler = data['scaler']
        ensemble.models = data['models']
        ensemble.feature_cols = data.get('feature_cols', [])
        ensemble.is_fitted = data['is_fitted']
        return ensemble
