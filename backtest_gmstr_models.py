#!/usr/bin/env python3
"""
GMSTR Model Backtest - Geçmiş veri üzerinde tahmin doğruluğu analizi
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from gmstr_system.predictor import GMSTRPredictor
from gmstr_system.data_loader import GMSTRDataLoader
from gmstr_system.features import FeatureEngineer
from gmstr_system.models import SimpleEnsemble


def backtest_models():
    print("="*70)
    print("GMSTR MODEL BACKTEST - Geçmiş Tahmin Doğruluk Analizi")
    print("="*70)

    # Modelleri yükle (sadece simple_*.pkl)
    predictor = GMSTRPredictor(model_dir='gmstr_models')
    predictor.models = {}
    from pathlib import Path
    for pkl_path in Path('gmstr_models').glob('simple_*.pkl'):
        h_name = pkl_path.stem.replace('simple_', '')
        try:
            predictor.models[h_name] = SimpleEnsemble.load(str(pkl_path))
            print(f"  ✓ Model yüklendi: {h_name}")
        except Exception as e:
            print(f"  ⚠ Model yüklenemedi {h_name}: {e}")
    predictor.is_ready = len(predictor.models) > 0
    if not predictor.is_ready:
        print("Modeller yüklenemedi!")
        return
    print(f"[Predictor] Sistem hazır. {len(predictor.models)} model yüklendi.")

    # Günlük veri yükle
    loader = GMSTRDataLoader('claude/areaxdatetime.csv')
    loader.load()
    df = loader.clean()

    # Özellikleri hesapla
    engineer = FeatureEngineer()
    df_feat = engineer.transform(df)

    exclude = {'Open', 'High', 'Low', 'Close', 'Volume',
               'Fund_Return', 'Benchmark_Return'}
    feature_cols = [c for c in df_feat.columns if c not in exclude]

    # Son 60 günü test et
    test_window = 60
    results = {}

    for h_name in ['1d_daily', '3d_daily', '5d_daily', '10d_daily']:
        if h_name not in predictor.models:
            continue

        model = predictor.models[h_name]
        bars = predictor.horizons.get(h_name, 1)

        # Modelin beklediği feature isimlerini al
        if hasattr(model, 'feature_cols') and model.feature_cols:
            expected = model.feature_cols
        elif hasattr(model.scaler, 'feature_names_in_'):
            expected = list(model.scaler.feature_names_in_)
        else:
            expected = feature_cols

        preds_up = 0
        preds_down = 0
        correct = 0
        total = 0
        actual_moves = []

        # Son 60 günü kaydırarak test et
        start_idx = max(len(df_feat) - test_window, 200)

        for i in range(start_idx, len(df_feat) - bars):
            # O günün verisi
            window = df_feat.iloc[:i+1]
            last = window.iloc[[-1]]

            # Feature hizalaması
            X_model = pd.DataFrame(0.0, index=last.index, columns=expected)
            for col in expected:
                if col in feature_cols:
                    X_model[col] = last[col].values
                else:
                    X_model[col] = 0.0

            # Tahmin
            try:
                proba = float(model.predict_proba(X_model)[0])
            except Exception:
                continue

            pred_up = proba > 0.5
            actual_price = df_feat['Close'].iloc[i]
            future_price = df_feat['Close'].iloc[i + bars]
            actual_up = future_price > actual_price
            move_pct = (future_price - actual_price) / actual_price * 100

            if pred_up:
                preds_up += 1
            else:
                preds_down += 1

            if pred_up == actual_up:
                correct += 1
            total += 1
            actual_moves.append(move_pct)

        if total == 0:
            continue

        accuracy = correct / total
        avg_move = np.mean(actual_moves)
        avg_abs_move = np.mean(np.abs(actual_moves))

        results[h_name] = {
            'total': total,
            'up_preds': preds_up,
            'down_preds': preds_down,
            'accuracy': accuracy,
            'avg_move': avg_move,
            'avg_abs_move': avg_abs_move,
        }

        print(f"\n{'─'*70}")
        print(f"[VADE: {h_name}] ({bars} gün sonrası)")
        print(f"{'─'*70}")
        print(f"  Test edilen gün:     {total}")
        print(f"  Yukarı tahmin:       {preds_up} ({preds_up/total:.1%})")
        print(f"  Aşağı tahmin:        {preds_down} ({preds_down/total:.1%})")
        print(f"  Doğruluk:            {accuracy:.2%}")
        print(f"  Ort. gerçek getiri:  {avg_move:+.3f}%")
        print(f"  Ort. mutlak hareket: {avg_abs_move:.3f}%")

    # Özet
    print(f"\n{'='*70}")
    print("ÖZET")
    print(f"{'='*70}")
    print(f"{'Vade':<12} {'Yukarı%':<10} {'Aşağı%':<10} {'Doğruluk':<10}")
    print(f"{'-'*42}")
    for h_name, r in results.items():
        up_pct = r['up_preds'] / r['total'] * 100
        down_pct = r['down_preds'] / r['total'] * 100
        print(f"{h_name:<12} {up_pct:>8.1f}% {down_pct:>8.1f}% {r['accuracy']:>8.1%}")

    print(f"\n{'='*70}")
    print("AÇIKLAMA:")
    print("  • Yukarı% = Modelin kaç kez 'yukarı' tahmin ettiği")
    print("  • Aşağı% = Modelin kaç kez 'aşağı' tahmin ettiği")
    print("  • Doğruluk = Tahminlerin gerçekleşme oranı")
    print(f"{'='*70}")


if __name__ == '__main__':
    backtest_models()
