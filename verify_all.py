"""
Tüm özelliklerin doğrulaması
"""
import urllib.request
import json

BASE = "http://localhost:5050"

def test_api(endpoint):
    try:
        req = urllib.request.urlopen(BASE + endpoint, timeout=10)
        data = json.loads(req.read().decode())
        return data
    except Exception as e:
        return {"success": False, "error": str(e)}

print("=" * 60)
print("  GMSTR SİSTEM DOĞRULAMA")
print("=" * 60)

# 1. Alım/Satım ekleme ekranı - trade_db ve /api/trades
print("\n[1] ALIM/SATIM EKLEME EKRANI")
data = test_api("/api/trades")
if data.get("success"):
    print(f"  ✅ /api/trades çalışıyor | {len(data.get('trades', []))} işlem kayıtlı")
else:
    print(f"  ❌ /api/trades HATA: {data.get('error')}")

# 2. Analiz ekranı - /api/analysis
print("\n[2] ALIM/SATIM ANALİZ EKRANI")
data = test_api("/api/analysis")
if data.get("success"):
    an = data.get("analysis", {})
    s = an.get("summary", {})
    p = an.get("performance", {})
    recs = an.get("recommendations", [])
    print(f"  ✅ /api/analysis çalışıyor")
    print(f"     Toplam işlem: {s.get('total_trades', 0)}")
    print(f"     Gerçekleşen K/Z: {s.get('realized_pl', 0):.2f} TL")
    print(f"     Kazanma oranı: %{p.get('win_rate', 0):.1f}")
    print(f"     Öneri sayısı: {len(recs)}")
else:
    print(f"  ❌ /api/analysis HATA: {data.get('error')}")

# 3. Model doğrulama - /api/model-validation
print("\n[3] MODEL EĞİTİMİ VE DOĞRULAMA")
data = test_api("/api/model-validation")
if data.get("success"):
    print(f"  ✅ /api/model-validation çalışıyor")
    print(f"     Genel skor: {data.get('overall_score', 0):.1f}/100")
    print(f"     Genel durum: {data.get('overall_status', '--')}")
    for r in data.get("validation_results", []):
        status = "✅" if r.get("is_realistic") or r.get("score", 0) >= 70 else "⚠️"
        print(f"     {status} {r['model']}: test=%{r['test_accuracy']*100:.1f} | AUC={r['test_auc']:.3f} | aylık=%{r['monthly_return_estimate']:.1f}")
else:
    print(f"  ❌ /api/model-validation HATA: {data.get('error')}")

# 4. Haber AI sistemi - /api/news
print("\n[4] HABER TABANLI AI SİSTEMİ")
data = test_api("/api/news")
if data.get("success"):
    news = data.get("news", [])
    sentiment = data.get("sentiment", {})
    combined = data.get("combined_prediction", {})
    print(f"  ✅ /api/news çalışıyor")
    print(f"     Haber sayısı: {len(news)} (hedef: 10-20)")
    print(f"     Haber sinyali: {sentiment.get('signal', '--')} | Güven: %{sentiment.get('confidence', 0):.1f}")
    print(f"     Kombine sinyal: {combined.get('final_signal', '--')}")
    print(f"     AI katkısı: %60 | Haber katkısı: %40")
else:
    print(f"  ❌ /api/news HATA: {data.get('error')}")

# 5. AI Tahminler - /api/predictions
print("\n[5] AI TAHMİN MONİTÖRÜ")
data = test_api("/api/predictions")
if data.get("success"):
    preds = data.get("predictions", {})
    print(f"  ✅ /api/predictions çalışıyor | {len(preds)} tahmin")
    for k, v in preds.items():
        direction = v.get("direction", "YOK")
        price = v.get("current_price", 0)
        pred_price = v.get("predicted_price", 0)
        chg = v.get("expected_change_pct", 0)
        has_direction = "✅" if direction and direction != "YOK" else "❌"
        has_live_price = "✅" if price and price > 600 else "❌"  # 693 TL civarı
        print(f"     {has_direction}{has_live_price} {k}: {direction} | {price} TL → {pred_price} TL ({chg:+.2f}%)")
else:
    print(f"  ❌ /api/predictions HATA: {data.get('error')}")

# 6. Dashboard - /api/dashboard
print("\n[6] DASHBOARD")
data = test_api("/api/dashboard")
if data.get("success"):
    price = data.get("price", {})
    print(f"  ✅ /api/dashboard çalışıyor")
    print(f"     Canlı fiyat: {price.get('price', '--')} TL | Kaynak: {price.get('source', '--')}")
else:
    print(f"  ❌ /api/dashboard HATA: {data.get('error')}")

print("\n" + "=" * 60)
print("  DOĞRULAMA TAMAMLANDI")
print("=" * 60)
