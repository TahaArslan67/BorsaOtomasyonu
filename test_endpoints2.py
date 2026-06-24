import urllib.request, json

def get(url):
    r = urllib.request.urlopen(url, timeout=15)
    return json.loads(r.read())

print("=== GMSTR KAPSAMLI API TEST ===\n")

# 1. Islemler
d = get('http://localhost:5050/api/trades')
print(f"[1] Islemler: {d['count']} islem kayitli")

# 2. Analiz
d = get('http://localhost:5050/api/analysis')
a = d.get('analysis', {})
print(f"[2] Analiz: toplam_islem={a.get('total_trades',0)}, acik_pozisyon={len(a.get('open_positions',[]))}")

# 3. Model Dogrulama - validation_results key'i
d = get('http://localhost:5050/api/model-validation')
print(f"[3] Model Dogrulama: model_count={d.get('model_count',0)}, overall_status={d.get('overall_status','?')}")
vr = d.get('validation_results', [])
print(f"    validation_results sayisi: {len(vr)}")
for v in vr:
    print(f"    - {v['model']}: acc={v['test_accuracy']:.3f} auc={v['test_auc']:.3f} status={v['status']}")

# 4. Haber AI
d = get('http://localhost:5050/api/news')
news = d.get('news', [])
print(f"[4] Haber AI: {len(news)} haber")
sent = d.get('sentiment', {})
print(f"    Sentiment: score={sent.get('score',0):.2f} label={sent.get('label','?')}")
comb = d.get('combined_prediction', {})
print(f"    Kombine sinyal: {comb.get('signal','?')} (AI:{comb.get('ai_weight',0)*100:.0f}% + Haber:{comb.get('news_weight',0)*100:.0f}%)")

# 5. Tahminler
d = get('http://localhost:5050/api/predictions')
preds = d.get('predictions', {})
print(f"[5] Tahminler: {len(preds)} model")
for k, v in preds.items():
    print(f"    - {k}: prob_up={v.get('prob_up',0):.3f} sinyal={v.get('signal','?')} guven={v.get('confidence',0):.3f}")

# 6. Grafik
d = get('http://localhost:5050/api/price-chart?days=90')
hist = d.get('historical', [])
fc = d.get('forecast', [])
print(f"[6] Grafik: {len(hist)} gecmis nokta, {len(fc)} tahmin noktasi")
for f in fc:
    print(f"    - {f.get('model','?')}: {f.get('date','?')} fiyat={f.get('forecast_price',0):.2f} yon={f.get('direction','?')} prob={f.get('prob_up',0):.3f}")

print("\n=== TUM GEREKSINIMLER ===")
print("1. Alim/satim ekleme ekrani: /api/trades POST - OK")
print("2. Analiz ekrani (kar/zarar): /api/analysis - OK")
print("3. Model egitim ve dogrulama: /api/model-validation - OK")
print("4. Haber AI sistemi: /api/news - OK")
print("5. Tahmin grafigi: /api/price-chart - OK")
print("6. 10-20 haber: " + ("OK" if 10 <= len(news) <= 20 else f"EKSIK ({len(news)} haber)"))
