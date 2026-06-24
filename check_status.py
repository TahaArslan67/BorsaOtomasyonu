import json, os

# 1. Tahminler kontrol
preds = json.load(open('gmstr_models/latest_predictions.json', encoding='utf-8'))
print('=== TAHMİNLER ===')
for k, v in preds.items():
    print(f"  {k}: direction={v.get('direction')} | price={v.get('current_price')} | predicted={v.get('predicted_price')} | chg={v.get('expected_change_pct')}")

# 2. Model dosyaları kontrol
print()
print('=== MODEL DOSYALARI ===')
models_dir = 'gmstr_models'
for f in sorted(os.listdir(models_dir)):
    if f.endswith('.pkl'):
        size = os.path.getsize(os.path.join(models_dir, f))
        print(f"  {f}: {size//1024} KB")

# 3. Training results kontrol
print()
print('=== EGİTİM SONUÇLARI ===')
tr = json.load(open('gmstr_models/training_results.json', encoding='utf-8'))
for k, v in tr.items():
    print(f"  {k}: test_acc={v.get('test_accuracy')} | auc={v.get('test_auc')} | realistic={v.get('is_realistic')}")

# 4. Dosya varlık kontrolleri
print()
print('=== DOSYA KONTROL ===')
files_to_check = [
    'gmstr_enhanced/app.py',
    'gmstr_enhanced/gmstr_monitor.html',
    'gmstr_enhanced/trade_db.py',
    'gmstr_enhanced/news_analyzer.py',
    'generate_predictions.py',
    'retrain_hourly_models.py',
    'gmstr_enhanced/retrain_gmstr.py',
]
for f in files_to_check:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"  {'OK' if exists else 'EKSIK'}: {f} ({size} bytes)")
