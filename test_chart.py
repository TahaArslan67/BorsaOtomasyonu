import urllib.request, json
r = urllib.request.urlopen('http://localhost:5050/api/price-chart?days=30', timeout=30)
d = json.loads(r.read())
hist = d['historical']
print('Son 3 gecmis nokta:')
for h in hist[-3:]:
    live = ' (CANLI)' if h.get('is_live') else ''
    print(f"  {h['date']}: {h['price']} TL{live}")
print(f"Son fiyat: {d['last_price']} TL")
print('Tahminler:')
for f in d['forecast']:
    print(f"  {f['model']}: {f['forecast_price']} TL ({f['direction']}) prob={f['prob_up']}")
