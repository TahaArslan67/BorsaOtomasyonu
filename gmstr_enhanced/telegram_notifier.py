"""
GMSTR Telegram Bildirim Sistemi
================================
Bot: Gmstrbildirimbot
Chat ID: 8590154095
"""
import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

TELEGRAM_TOKEN = "8980698319:AAFV-jMFGIfNNcjjsVDYJnapPGJ4MCyB7Bs"
CHAT_ID = "8590154095"
API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

ROOT = Path(__file__).parent.parent


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Telegram'a mesaj gönder."""
    try:
        url = f"{API_URL}/sendMessage"
        data = json.dumps({
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=10)
        result = json.loads(r.read())
        return result.get("ok", False)
    except Exception as e:
        print(f"[Telegram] Mesaj gönderilemedi: {e}")
        return False


def format_signal_emoji(signal: str) -> str:
    """Sinyal için emoji döndür."""
    if signal == "AL":
        return "🟢"
    elif signal == "SAT":
        return "🔴"
    else:
        return "🟡"


def send_ai_commentary(commentary: dict) -> bool:
    """AI yorumunu Telegram'a gönder."""
    c = commentary
    signal = c.get("dominant_signal", "BEKLE")
    emoji = format_signal_emoji(signal)
    action = c.get("action_text", "")
    price = c.get("current_price", 0)
    monthly = c.get("monthly_return_est", 0)
    pt = c.get("price_targets", {})
    main_comment = c.get("main_comment", "")
    strategy = c.get("strategy_comment", "")
    timestamp = c.get("timestamp", datetime.now().strftime("%d.%m.%Y %H:%M"))
    best = c.get("best_model", {})
    counts = c.get("signal_counts", {})
    
    # Hedef tarihleri
    target_dates = c.get("target_dates", {})
    t1_date = target_dates.get("target_1_date", "")
    t2_date = target_dates.get("target_2_date", "")
    
    text = f"""🥈 <b>GMSTR AI SİNYAL BİLDİRİMİ</b>
🕐 {timestamp}

{emoji} <b>{action}</b>
💰 Güncel Fiyat: <b>{price:.2f} ₺</b>

📊 <b>Model Sinyalleri:</b> AL:{counts.get('AL',0)} | SAT:{counts.get('SAT',0)} | BEKLE:{counts.get('BEKLE',0)}
🏆 En iyi model: {best.get('label','--')} (%{best.get('accuracy',0)*100:.1f} doğruluk)

💬 {main_comment}

📋 <b>Strateji:</b>
{strategy}"""

    if pt.get("buy_zone_low"):
        text += f"""

🎯 <b>Fiyat Hedefleri:</b>
📥 Alım Bölgesi: {pt.get('buy_zone_low',0):.2f} - {pt.get('buy_zone_high',0):.2f} ₺
🎯 Hedef 1: <b>{pt.get('target_1',0):.2f} ₺</b>{f' (~{t1_date})' if t1_date else ''}
🎯 Hedef 2: <b>{pt.get('target_2',0):.2f} ₺</b>{f' (~{t2_date})' if t2_date else ''}
🛑 Stop-Loss: {pt.get('stop_loss',0):.2f} ₺"""
    
    text += f"""

📈 Aylık Getiri Tahmini: ~%{monthly}

⚠️ <i>Bu mesaj yatırım tavsiyesi değildir.</i>"""
    
    return send_message(text)


def send_price_alert(price: float, change_pct: float, signal: str) -> bool:
    """Fiyat alarmı gönder."""
    emoji = "📈" if change_pct > 0 else "📉"
    sig_emoji = format_signal_emoji(signal)
    
    text = f"""🥈 <b>GMSTR FİYAT ALARMI</b>
{emoji} Fiyat: <b>{price:.2f} ₺</b>
📊 Değişim: {'+' if change_pct > 0 else ''}{change_pct:.2f}%
{sig_emoji} AI Sinyali: <b>{signal}</b>
🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}"""
    
    return send_message(text)


def send_strong_signal_alert(signal: str, prob: float, model: str, price: float) -> bool:
    """Güçlü sinyal alarmı gönder."""
    emoji = "🚀" if signal == "AL" else "⚠️"
    sig_emoji = format_signal_emoji(signal)
    
    text = f"""🥈 <b>GMSTR GÜÇLÜ SİNYAL!</b>
{emoji} {sig_emoji} <b>{signal} SİNYALİ</b>
💰 Fiyat: {price:.2f} ₺
📊 Olasılık: %{prob*100:.1f}
🤖 Model: {model}
🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}

⚠️ <i>Yatırım tavsiyesi değildir.</i>"""
    
    return send_message(text)


def send_daily_summary() -> bool:
    """Günlük özet gönder."""
    try:
        # Tahminleri yükle
        pred_path = ROOT / "gmstr_models" / "latest_predictions.json"
        if not pred_path.exists():
            return False
        
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        
        # Fiyatı al
        import yfinance as yf
        ticker = yf.Ticker("GMSTR.IS")
        hist = ticker.history(period="1d", interval="1m")
        price = float(hist["Close"].iloc[-1]) if len(hist) > 0 else 0
        
        lines = [f"🥈 <b>GMSTR GÜNLÜK ÖZET</b>", f"🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}", f"💰 Güncel Fiyat: <b>{price:.2f} ₺</b>", "", "📊 <b>AI Tahminleri:</b>"]
        
        labels = {
            "15m_15min": "15 Dakika",
            "1h_hourly": "1 Saat",
            "4h_hourly": "4 Saat",
            "1d_daily": "1 Gün",
            "3d_daily": "3 Gün",
            "5d_daily": "5 Gün",
            "10d_daily": "10 Gün",
        }
        
        for key, label in labels.items():
            p = preds.get(key)
            if not p:
                continue
            sig = p.get("signal", "BEKLE")
            prob = p.get("prob_up", 0.5)
            sig_emoji = format_signal_emoji(sig)
            lines.append(f"{sig_emoji} {label}: <b>{sig}</b> (%{prob*100:.0f})")
        
        lines.append("")
        lines.append("⚠️ <i>Yatırım tavsiyesi değildir.</i>")
        
        return send_message("\n".join(lines))
    except Exception as e:
        print(f"[Telegram] Günlük özet hatası: {e}")
        return False


if __name__ == "__main__":
    # Test
    print("Telegram bağlantısı test ediliyor...")
    ok = send_message("🥈 <b>GMSTR Bot aktif!</b>\nSistem başarıyla başlatıldı. Bildirimler açık. ✅")
    print("Sonuç:", "✅ Başarılı" if ok else "❌ Başarısız")
