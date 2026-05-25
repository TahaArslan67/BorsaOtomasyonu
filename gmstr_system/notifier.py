"""
GMSTR Bildirim Modülü - Telegram Entegrasyonu
Saatlik tahmin yönü değiştiğinde anlık bildirim gönderir.
"""
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


class TelegramNotifier:
    """Telegram Bot API üzerinden bildirim gönderen sınıf."""

    CONFIG_PATH = Path(__file__).parent.parent / 'gmstr_models' / 'telegram_config.json'

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = False
        self._load_config()

    def _load_config(self):
        """Kayıtlı Telegram ayarlarını yükle."""
        if self.CONFIG_PATH.exists():
            try:
                with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.bot_token = cfg.get('bot_token', self.bot_token)
                self.chat_id = cfg.get('chat_id', self.chat_id)
            except Exception:
                pass
        self.enabled = bool(self.bot_token and self.chat_id)

    def save_config(self):
        """Telegram ayarlarını kaydet."""
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'bot_token': self.bot_token,
                'chat_id': self.chat_id,
            }, f, ensure_ascii=False, indent=2)
        self.enabled = bool(self.bot_token and self.chat_id)

    def send_message(self, message: str) -> bool:
        """Telegram'a mesaj gönder. Başarılı ise True döner."""
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = urllib.parse.urlencode({
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
            }).encode('utf-8')
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[Notifier] Telegram hatası: {e}")
            return False

    def test_connection(self) -> bool:
        """Bağlantıyı test et."""
        return self.send_message("🤖 GMSTR Monitör aktif! Bildirimler çalışıyor.")

    def notify_direction_change(self, horizon: str, old_dir: str, new_dir: str,
                                 confidence: float, current_price: float,
                                 predicted_price: float):
        """Yön değişikliği bildirimi gönder."""
        emoji_old = "🟢" if "YUKARI" in old_dir else "🔴"
        emoji_new = "🟢" if "YUKARI" in new_dir else "🔴"
        msg = (
            f"<b>⚠️ GMSTR YÖN DEĞİŞİKLİĞİ</b>\n\n"
            f"<b>Vade:</b> {horizon}\n"
            f"<b>Eski:</b> {emoji_old} {old_dir}\n"
            f"<b>Yeni:</b> {emoji_new} {new_dir}\n\n"
            f"<b>Güven:</b> {confidence:.0%}\n"
            f"<b>Son Fiyat:</b> {current_price:.2f} TRY\n"
            f"<b>Tahmini:</b> {predicted_price:.2f} TRY\n\n"
            f"⏰ {self._now()}"
        )
        return self.send_message(msg)

    @staticmethod
    def _now() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
