#!/usr/bin/env python3
"""
GMSTR Advanced Prediction System - Ana Koordinasyon
=====================================================
Kullanım:
  python -m gmstr_system.main --mode train    # Modelleri eğit
  python -m gmstr_system.main --mode predict  # Tahmin üret
  python -m gmstr_system.main --mode full     # Eğit + Tahmin + Rapor
  python -m gmstr_system.main --mode live     # Canlı monitör

Seçenekler:
  --target-mode {binary,dynamic,3class}  # Hedef tipi (varsayılan: dynamic)
  --no-mlp                               # MLP sinir ağını devre dışı bırak
  --interval SEC                         # Live modu aralığı (varsayılan: 300)
  --daily-csv PATH                       # Günlük veri kaynağı
  --hourly-csv PATH                      # Saatlik veri kaynağı
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .data_loader import load_and_prepare
from .training import ModelTrainer
from .evaluation import ModelEvaluator
from .predictor import GMSTRPredictor
from .features import FeatureEngineer


def print_banner():
    print("\n" + "="*70)
    print("  GMSTR (Gümüş BYF) GELİŞMİŞ YAPAY ZEKA TAHMİN SİSTEMİ v2.0")
    print("  XGB + LGB + RF + GB + ET + MLP -> Stacked Ensemble")
    print("  5Y Günlük + 2Y Saatlik | Makro Özellikler | Dinamik Hedef")
    print("="*70)
    print()


def train_mode(daily_csv: Optional[str] = None,
               hourly_csv: Optional[str] = None,
               model_dir: str = 'gmstr_models',
               target_mode: str = 'dynamic',
               use_mlp: bool = True):
    print_banner()
    print(f"[MOD] EĞİTİM | Target={target_mode} | MLP={'Açık' if use_mlp else 'Kapalı'}\n")

    trainer = ModelTrainer(model_dir=model_dir, target_mode=target_mode, use_mlp=use_mlp)
    results = trainer.train_all(daily_csv=daily_csv, hourly_csv=hourly_csv)

    # Özet tablo
    print("\n" + "="*70)
    print("EĞİTİM ÖZET TABLOSU")
    print("="*70)
    print(f"{'Vade':<15} {'CV Acc':<12} {'Test Acc':<12} {'AUC':<8} {'HC60':<12} {'HC70':<12}")
    print("-"*70)
    for h_name, r in sorted(results.items()):
        cv = f"{r.get('cv_accuracy', 0):.2%}±{r.get('cv_std', 0):.2%}"
        test = f"{r.get('test_accuracy', 0):.2%}" if r.get('test_accuracy') else "N/A"
        auc = f"{r.get('test_auc', 0):.3f}" if r.get('test_auc') else "N/A"
        hc60 = f"{r.get('hc_60_acc', 0):.2%}" if r.get('hc_60_acc') else "N/A"
        hc70 = f"{r.get('hc_70_acc', 0):.2%}" if r.get('hc_70_acc') else "N/A"
        print(f"{h_name:<15} {cv:<12} {test:<12} {auc:<8} {hc60:<12} {hc70:<12}")
    print("="*70)

    print(f"\n✓ Modeller kaydedildi: {Path(model_dir).absolute()}/")
    return results


def predict_mode(model_dir: str = 'gmstr_models',
                 daily_csv: str = None,
                 hourly_csv: str = None):
    print_banner()
    print("[MOD] TAHMİN\n")

    predictor = GMSTRPredictor(model_dir=model_dir,
                               daily_csv=daily_csv,
                               hourly_csv=hourly_csv)

    if not predictor.load_system():
        print("\n⚠ Modeller bulunamadı! Önce eğitim yapın:")
        print("   python -m gmstr_system.main --mode train")
        return None

    # Günlük tahminler
    print("\n--- GÜNLÜK TAHMİNLER ---")
    predictions_daily = predictor.predict(is_hourly=False)
    predictor.print_predictions(predictions_daily)
    print(predictor.get_recommendation(predictions_daily, min_confidence=0.60))

    # Saatlik tahminler (eğer varsa)
    hourly_models = [k for k in predictor.models if 'hourly' in k]
    if hourly_models:
        print("\n--- SAATLİK TAHMİNLER ---")
        try:
            predictions_hourly = predictor.predict(is_hourly=True)
            predictor.print_predictions(predictions_hourly)
            print(predictor.get_recommendation(predictions_hourly, min_confidence=0.60))
        except Exception as e:
            print(f"⚠ Saatlik tahmin hatası: {e}")

    # JSON kaydet
    out_path = Path(model_dir) / 'latest_predictions.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_daily, f, indent=2, ensure_ascii=False)
    print(f"\nTahminler kaydedildi: {out_path}")

    return predictions_daily


def full_mode(daily_csv: Optional[str] = None,
              hourly_csv: Optional[str] = None,
              model_dir: str = 'gmstr_models',
              target_mode: str = 'dynamic',
              use_mlp: bool = True):
    train_mode(daily_csv, hourly_csv, model_dir, target_mode, use_mlp)
    print("\n\n")
    predict_mode(model_dir, daily_csv, hourly_csv)


def live_mode(model_dir: str = 'gmstr_models',
              daily_csv: str = None,
              hourly_csv: str = None,
              interval_sec: int = 300):
    from .live_monitor import LiveMonitor
    monitor = LiveMonitor(
        model_dir=model_dir,
        interval_sec=interval_sec,
        daily_csv=daily_csv,
        hourly_csv=hourly_csv,
    )
    monitor.start()


def main():
    parser = argparse.ArgumentParser(
        description='GMSTR Gelişmiş Yapay Zeka Tahmin Sistemi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python -m gmstr_system.main --mode train
  python -m gmstr_system.main --mode train --target-mode dynamic
  python -m gmstr_system.main --mode train --no-mlp
  python -m gmstr_system.main --mode predict
  python -m gmstr_system.main --mode full --target-mode 3class
  python -m gmstr_system.main --mode live --interval 300
        """
    )
    parser.add_argument('--mode', choices=['train', 'predict', 'full', 'live'],
                        default='full', help='Çalışma modu')
    parser.add_argument('--daily-csv', type=str, default=None,
                        help='Günlük veri CSV (varsayılan: claude/areaxdatetime.csv)')
    parser.add_argument('--hourly-csv', type=str, default=None,
                        help='Saatlik veri CSV (varsayılan: claude/gercek_data.csv)')
    parser.add_argument('--model-dir', type=str, default='gmstr_models',
                        help='Model kayıt dizini')
    parser.add_argument('--target-mode', choices=['binary', 'dynamic', '3class'],
                        default='dynamic', help='Hedef değişken tipi')
    parser.add_argument('--no-mlp', action='store_true',
                        help='MLP sinir ağını devre dışı bırak')
    parser.add_argument('--interval', type=int, default=300,
                        help='Live modu güncelleme aralığı (saniye, varsayılan: 300)')

    args = parser.parse_args()

    use_mlp = not args.no_mlp

    try:
        if args.mode == 'train':
            train_mode(args.daily_csv, args.hourly_csv, args.model_dir,
                       args.target_mode, use_mlp)
        elif args.mode == 'predict':
            predict_mode(args.model_dir, args.daily_csv, args.hourly_csv)
        elif args.mode == 'live':
            live_mode(args.model_dir, args.daily_csv, args.hourly_csv, args.interval)
        else:
            full_mode(args.daily_csv, args.hourly_csv, args.model_dir,
                      args.target_mode, use_mlp)
    except KeyboardInterrupt:
        print("\n\nKullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
