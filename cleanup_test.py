import urllib.request, json

# Test islemlerini sil (id 1-10 arasi)
r = urllib.request.urlopen('http://localhost:5050/api/trades', timeout=10)
d = json.loads(r.read())
trades = d.get('trades', [])
print(f"Toplam islem: {len(trades)}")
for t in trades:
    tid = t['id']
    req = urllib.request.Request(f'http://localhost:5050/api/trades/{tid}', method='DELETE')
    r2 = urllib.request.urlopen(req, timeout=10)
    d2 = json.loads(r2.read())
    print(f"Silindi: #{tid} - {d2.get('message','')}")

print("Temizlik tamamlandi.")
