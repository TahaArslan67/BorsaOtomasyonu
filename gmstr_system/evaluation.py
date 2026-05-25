"""
GMSTR Model Değerlendirme Modülü
Backtest, güven eşiği analizi, sinyal kalitesi raporlama.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import json


class ModelEvaluator:
    """Model performans ve güvenilirlik değerlendiricisi."""
    
    def __init__(self, model_dir: str = 'gmstr_models'):
        self.model_dir = Path(model_dir)
        self.results: Dict[str, Dict] = {}
    
    def evaluate_confidence_thresholds(self, y_true: np.ndarray, probas: np.ndarray, 
                                        thresholds: List[float] = None) -> pd.DataFrame:
        """
        Farklı güven eşiklerinde model performansını analiz et.
        
        Returns:
            DataFrame: Her eşik için accuracy, coverage, precision, recall
        """
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        
        records = []
        for thresh in thresholds:
            # İki taraflı eşik: >thresh (güçlü yukarı) veya <(1-thresh) (güçlü aşağı)
            mask = (probas > thresh) | (probas < (1 - thresh))
            coverage = mask.mean()
            
            if mask.sum() < 5:
                records.append({
                    'threshold': thresh,
                    'coverage_pct': coverage * 100,
                    'n_samples': mask.sum(),
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1': np.nan,
                })
                continue
            
            preds = (probas[mask] > 0.5).astype(int)
            yt = y_true[mask]
            
            acc = accuracy_score(yt, preds)
            prec = precision_score(yt, preds, zero_division=0)
            rec = recall_score(yt, preds, zero_division=0)
            f1 = f1_score(yt, preds, zero_division=0)
            
            records.append({
                'threshold': thresh,
                'coverage_pct': coverage * 100,
                'n_samples': int(mask.sum()),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
            })
        
        return pd.DataFrame(records)
    
    def backtest_strategy(self, df: pd.DataFrame, predictions: pd.Series, 
                          probas: pd.Series, threshold: float = 0.60,
                          initial_capital: float = 10000.0) -> Dict:
        """
        Basit long-only backtest: Yüksek güvenli yukarı sinyallerinde al, 
        düşük güvenli aşağı sinyallerinde nakite geç.
        """
        capital = initial_capital
        position = 0.0  # 0 = nakit, 1 = tam pozisyon
        trades = []
        equity_curve = [capital]
        
        for i in range(len(df) - 1):
            current_price = df['Close'].iloc[i]
            next_price = df['Close'].iloc[i + 1]
            prob = probas.iloc[i]
            
            # Sadece yüksek güven sinyallerini değerlendir
            if prob > threshold:  # Güçlü yukarı
                if position == 0:
                    position = 1.0
                    trades.append({
                        'type': 'BUY',
                        'idx': i,
                        'price': current_price,
                        'prob': prob,
                    })
            elif prob < (1 - threshold):  # Güçlü aşağı
                if position == 1.0:
                    position = 0.0
                    trades.append({
                        'type': 'SELL',
                        'idx': i,
                        'price': current_price,
                        'prob': prob,
                    })
            
            # Basit getiri hesaplama (günlük)
            if position == 1.0:
                ret = (next_price - current_price) / current_price
            else:
                ret = 0.0
            
            capital *= (1 + ret)
            equity_curve.append(capital)
        
        # Metrikler
        total_return = (capital - initial_capital) / initial_capital
        n_trades = len([t for t in trades if t['type'] == 'BUY'])
        
        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            max_dd = ((equity / equity.cummax()) - 1).min()
        else:
            sharpe = 0.0
            max_dd = 0.0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return * 100,
            'n_trades': n_trades,
            'sharpe_annual': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'trades': trades,
            'equity_curve': equity_curve,
        }
    
    def generate_report(self, horizon: str, y_true: np.ndarray, probas: np.ndarray,
                        df_test: pd.DataFrame = None) -> Dict:
        """Kapsamlı değerlendirme raporu üret."""
        preds = (probas > 0.5).astype(int)
        
        # Temel metrikler
        acc = accuracy_score(y_true, preds)
        auc = roc_auc_score(y_true, probas)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds).tolist()
        
        # Güven eşiği analizi
        conf_df = self.evaluate_confidence_thresholds(y_true, probas)
        
        # En iyi eşik (coverage > %20 ve accuracy maksimum)
        valid = conf_df[(conf_df['coverage_pct'] > 15) & (conf_df['accuracy'].notna())]
        if not valid.empty:
            best_row = valid.loc[valid['accuracy'].idxmax()]
            best_threshold = best_row['threshold']
            best_acc = best_row['accuracy']
            best_cov = best_row['coverage_pct']
        else:
            best_threshold = 0.60
            best_acc = acc
            best_cov = 100.0
        
        report = {
            'horizon': horizon,
            'accuracy': round(acc, 4),
            'auc_roc': round(auc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'confusion_matrix': cm,
            'best_confidence_threshold': round(best_threshold, 2),
            'best_hc_accuracy': round(best_acc, 4),
            'best_hc_coverage': round(best_cov, 2),
            'confidence_analysis': conf_df.to_dict('records'),
        }
        
        # Backtest (eğer fiyat verisi varsa)
        if df_test is not None and len(df_test) == len(y_true):
            for thresh in [0.60, 0.65, 0.70]:
                bt = self.backtest_strategy(df_test, pd.Series(preds), pd.Series(probas), 
                                            threshold=thresh)
                report[f'backtest_{int(thresh*100)}'] = {
                    'total_return_pct': round(bt['total_return_pct'], 2),
                    'n_trades': bt['n_trades'],
                    'sharpe': round(bt['sharpe_annual'], 3),
                    'max_drawdown_pct': round(bt['max_drawdown_pct'], 2),
                }
        
        return report
    
    def print_report(self, report: Dict):
        """Konsola rapor yazdır."""
        h = report['horizon']
        print(f"\n{'='*60}")
        print(f"DEĞERLENDİRME RAPORU [{h}]")
        print(f"{'='*60}")
        print(f"  Doğruluk (Accuracy):     {report['accuracy']:.2%}")
        print(f"  AUC-ROC:                 {report['auc_roc']:.3f}")
        print(f"  Precision:               {report['precision']:.2%}")
        print(f"  Recall:                  {report['recall']:.2%}")
        print(f"  F1-Score:                {report['f1_score']:.2%}")
        print(f"\n  Confusion Matrix:        {report['confusion_matrix']}")
        print(f"\n  En İyi Güven Eşiği:      {report['best_confidence_threshold']:.0%}")
        print(f"  HC Doğruluk:             {report['best_hc_accuracy']:.2%}")
        print(f"  HC Kapsam:               {report['best_hc_coverage']:.1f}%")
        
        if 'backtest_60' in report:
            bt = report['backtest_60']
            print(f"\n  Backtest (60% eşik):")
            print(f"    Getiri:                {bt['total_return_pct']:.2f}%")
            print(f"    İşlem Sayısı:          {bt['n_trades']}")
            print(f"    Sharpe:                {bt['sharpe']:.3f}")
            print(f"    Max Drawdown:          {bt['max_drawdown_pct']:.2f}%")
        
        print(f"{'='*60}")
    
    def save_report(self, report: Dict, filename: str = None):
        """Raporu JSON olarak kaydet."""
        if filename is None:
            filename = f"evaluation_{report['horizon']}.json"
        path = self.model_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Rapor kaydedildi: {path}")


class SignalQualityAnalyzer:
    """Sinyal kalitesi ve piyasa gürültüsü analizi."""
    
    @staticmethod
    def noise_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Fiyat hareketinin ne kadarının gürültü olduğunu ölç."""
        returns = df['Close'].pct_change()
        trend = returns.rolling(window).mean()
        noise = returns - trend
        return (noise.abs() / returns.abs().replace(0, np.nan)).rolling(window).mean()
    
    @staticmethod
    def signal_to_noise(probas: np.ndarray, threshold: float = 0.60) -> float:
        """Güven skorlarının sinyal/gürültü oranı."""
        strong_signals = (probas > threshold) | (probas < (1 - threshold))
        return strong_signals.mean()
    
    @staticmethod
    def regime_consistency(df: pd.DataFrame, predictions: np.ndarray, 
                           window: int = 20) -> Dict:
        """Farklı volatilite rejimlerinde model tutarlılığı."""
        returns = df['Close'].pct_change()
        vol = returns.rolling(window).std()
        
        # Düşük, orta, yüksek volatilite
        low_vol = vol <= vol.quantile(0.33)
        mid_vol = (vol > vol.quantile(0.33)) & (vol <= vol.quantile(0.67))
        high_vol = vol > vol.quantile(0.67)
        
        # Tahmin doğruluğu her rejimde
        actual = (returns.shift(-1) > 0).astype(int).values
        
        results = {}
        for name, mask in [('low_vol', low_vol), ('mid_vol', mid_vol), ('high_vol', high_vol)]:
            mask = mask.values
            if mask.sum() > 10:
                acc = accuracy_score(actual[mask], predictions[mask])
                results[name] = round(acc, 4)
        
        return results
