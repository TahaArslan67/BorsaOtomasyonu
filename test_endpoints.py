import urllib.request, json

def test(url, label):
    try:
        r = urllib.request.urlopen(url, timeout=10)
        data = json.loads(r.read())
        print(f'[OK] {label}: keys={list(data.keys())[:6]}')
        return data
    except Exception as e:
        print(f'[HATA] {label}: {e}')
        return None

print("=== GMSTR API TEST ===\n")

# 1. Alim/satim endpoint
d = test('http://localhost:5050/api/trades', '1. Islemler (GET)')
if d:
    print(f'   Islem sayisi: {len(d.get("trades", []))}')

# 2. Analiz endpoint
d = test('http://localhost:5050/api/analysis', '2. Analiz (kar/zarar)')
if d:
    print(f'   Toplam kar/zarar: {d.get("total_pnl", "N/A")}')
    print(f'   Oneri: {str(d.get("recommendation", ""))[:80]}')

# 3. Model dogrulama
d = test('http://localhost:5050/api/model-validation', '3. Model Dogrulama')
if d:
    models = d.get('models', {})
    print(f'   Model sayisi: {len(models)}')
    for k, v in list(models.items())[:2]:
        print(f'   {k}: AUC={v.get("test_auc","?")} Acc={v.get("test_accuracy","?")}')

# 4. Haber AI
d = test('http://localhost:5050/api/news', '4. Haber AI')
if d:
    news = d.get('news', [])
    print(f'   Haber sayisi: {len(news)}')
    for n in news[:3]:
        print(f'   - {n.get("title","")[:70]}')

# 5. Grafik
d = test('http://localhost:5050/api/price-chart?days=90', '5. Fiyat Grafigi')
if d:
    print(f'   Gecmis: {len(d.get("historical",[]))} nokta')
    print(f'   Tahmin: {len(d.get("forecast",[]))} nokta')
    fc = d.get('forecast', [])
    for f in fc:
        print(f'   Tahmin {f.get("horizon","")}: {f.get("direction","?")} %{f.get("probability",0)*100:.0f}')

print("\n=== TEST TAMAMLANDI ===")
