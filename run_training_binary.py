#!/usr/bin/env python3
"""GMSTR Gunluk Model Egitim Scripti - Binary target (threshold=0)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# UTF-8 encoding ayari ve dual logging
import io

class TeeLogger:
    def __init__(self, filename, stdout):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = stdout
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
        self.stdout.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()

sys.stdout = TeeLogger('train_log_binary.txt', sys.stdout)
sys.stderr = TeeLogger('train_log_binary.txt', sys.stderr)

from gmstr_system.training import ModelTrainer

print("="*70)
print("GMSTR Gunluk Model Egitimi - Binary Target")
print("="*70)

trainer = ModelTrainer(model_dir='gmstr_models', target_mode='binary', use_mlp=True)

try:
    results = trainer.train_daily_models()
    print("\n" + "="*70)
    print("EGITIM OZET TABLOSU")
    print("="*70)
    print(f"{'Vade':<15} {'CV Acc':<12} {'Test Acc':<12} {'AUC':<8}")
    print("-"*70)
    for h_name, r in sorted(results.items()):
        cv = f"{r.get('cv_accuracy', 0):.2%}+-{r.get('cv_std', 0):.2%}"
        test = f"{r.get('test_accuracy', 0):.2%}" if r.get('test_accuracy') else "N/A"
        auc = f"{r.get('test_auc', 0):.3f}" if r.get('test_auc') else "N/A"
        print(f"{h_name:<15} {cv:<12} {test:<12} {auc:<8}")
    print("="*70)
    print(f"\n[OK] Modeller kaydedildi: {Path('gmstr_models').absolute()}/")
except Exception as e:
    print(f"\n[HATA] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
