import urllib.request, json

def test(url, label):
    try:
        r = urllib.request.urlopen(url, timeout=10)
        d = json.loads(r.read())
        print(f"[OK] {label}")
        return d
    except Exception as e:
        print(f"[HATA] {label}: {e}")
        return {}

# Model doğrulama
d = test('http://localhost:5050/api/model-validation', 'Model Doğrulama')
print(f"  Genel Skor: {d.get('overall_score')} | Durum: {d.get('overall_status')}")
for v in d.get('validation_results', []):
    print(f"  {v['model']}: Test={v['test_accuracy']*100:.1f}% AUC={v['test_auc']:.3f} Skor={v['score']} {v['status']}")

print()

# İşlem ekleme testi
import urllib.request
req = urllib.request.Request(
    'http://localhost:5050/api/trades',
    data=json.dumps({
        'trade_type': 'BUY',
        'trade_date': '2026-05-01',
        'trade_time': '10:30',
        'price': 680.0,
        'quantity': 10,
        'commission': 5.0,
        'notes': 'Test alım',
        'bot_signal': 'YUKARI 62%'
    }).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    r = urllib.request.urlopen(req, timeout=5)
    d = json.loads(r.read())
    print(f"[OK] İşlem Ekleme: {d.get('message')}")
except Exception as e:
    print(f"[HATA] İşlem Ekleme: {e}")

# Satım ekle
req2 = urllib.request.Request(
    'http://localhost:5050/api/trades',
    data=json.dumps({
        'trade_type': 'SELL',
        'trade_date': '2026-05-15',
        'trade_time': '14:00',
        'price': 720.0,
        'quantity': 10,
        'commission': 5.0,
        'notes': 'Test satım',
        'bot_signal': 'AŞAĞI 58%'
    }).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    r = urllib.request.urlopen(req2, timeout=5)
    d = json.loads(r.read())
    print(f"[OK] Satım Ekleme: {d.get('message')}")
except Exception as e:
    print(f"[HATA] Satım Ekleme: {e}")

# Analiz
d = test('http://localhost:5050/api/analysis', 'Analiz')
an = d.get('analysis', {})
s = an.get('summary', {})
p = an.get('performance', {})
print(f"  Toplam İşlem: {s.get('total_trades')} | K/Z: {s.get('realized_pl')} TL")
print(f"  Kazanma: %{p.get('win_rate')} | Profit Factor: {p.get('profit_factor')}")
recs = an.get('recommendations', [])
for r in recs:
    print(f"  {r.get('icon')} {r.get('title')}: {r.get('text','')[:60]}...")

print()
print("Tum testler tamamlandi!")
