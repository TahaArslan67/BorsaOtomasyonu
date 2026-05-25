"""
GMSTR Canlı Terminal Monitörü
Belirli aralıklarla tahmin üretir, terminale tablo şeklinde yazar.
Veri dosyası harici bir süreç tarafından güncellendiğinde yeni tahminler değişir.
"""
import time
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from .predictor import GMSTRPredictor


class LiveMonitor:
    """Sürekli çalışan canlı tahmin monitörü."""

    def __init__(self, model_dir: str = 'gmstr_models',
                 interval_sec: int = 300,
                 daily_csv: str = None,
                 hourly_csv: str = None,
                 history_size: int = 20):
        self.model_dir = model_dir
        self.interval_sec = interval_sec
        self.daily_csv = daily_csv
        self.hourly_csv = hourly_csv
        self.history_size = history_size
        self.history: List[Dict] = []
        self.predictor = GMSTRPredictor(model_dir, daily_csv, hourly_csv)

    def start(self):
        """Monitör döngüsünü başlat."""
        print("\n🔴 GMSTR CANLI TAHMİN MONİTÖRÜ BAŞLATILIYOR...")
        if not self.predictor.load_system():
            print("❌ Modeller yüklenemedi, çıkılıyor.")
            sys.exit(1)

        print(f"   Güncelleme aralığı: {self.interval_sec} saniye ({self.interval_sec / 60:.1f} dk)")
        print(f"   Çıkmak için: Ctrl+C\n")
        print("İlk tahmin üretiliyor...")

        try:
            while True:
                self._tick()
                # Geri sayım gösterimi
                for remaining in range(self.interval_sec, 0, -1):
                    mins, secs = divmod(remaining, 60)
                    sys.stdout.write(f"\r   ⏳ Sonraki güncelleme: {mins:02d}:{secs:02d}  (Ctrl+C ile durdur)")
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n\n🛑 Monitör kullanıcı tarafından durduruldu.")
            self._show_summary()

    def _tick(self):
        """Tek bir tahmin döngüsü."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            pred_daily = self.predictor.predict(is_hourly=False)
            pred_hourly = self.predictor.predict(is_hourly=True)
        except Exception as e:
            print(f"[{now}] ❌ Tahmin hatası: {e}")
            return

        record = {
            'time': now,
            'daily': pred_daily,
            'hourly': pred_hourly,
        }
        self.history.append(record)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        self._render(now, pred_daily, pred_hourly)
        self._save_latest(record)

    def _render(self, now: str, pred_daily: Dict, pred_hourly: Dict):
        """Terminal ekranını temizleyip tabloyu yaz."""
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 85)
        print(f"  GMSTR (Gümüş BYF) CANLI TAHMİN MONİTÖRÜ  |  Son Güncelleme: {now}")
        print("=" * 85)

        # ── Günlük Tahminler ──
        print("\n📅 GÜNLÜK TAHMİNLER")
        print("-" * 85)
        header = f"{'Vade':<14} {'Yön':<12} {'Güven':<10} {'Son Fiyat':<14} {'Tahmini':<14} {'Değişim':<10}"
        print(header)
        print("-" * 85)
        for h_name in ['1d_daily', '3d_daily', '5d_daily', '10d_daily']:
            if h_name in pred_daily:
                p = pred_daily[h_name]
                emoji = "🟢" if 'YUKARI' in p['direction'] else "🔴"
                print(f"{emoji} {h_name:<12} {p['direction']:<12} {p['confidence']:<10.0%} "
                      f"{p['current_price']:<14.2f} {p['predicted_price']:<14.2f} "
                      f"{p['expected_change_pct']:+.2f}%")

        # ── Saatlik Tahminler ──
        print("\n⏰ SAATLİK TAHMİNLER")
        print("-" * 85)
        print(header)
        print("-" * 85)
        for h_name in ['1h_hourly', '4h_hourly']:
            if h_name in pred_hourly:
                p = pred_hourly[h_name]
                emoji = "🟢" if 'YUKARI' in p['direction'] else "🔴"
                print(f"{emoji} {h_name:<12} {p['direction']:<12} {p['confidence']:<10.0%} "
                      f"{p['current_price']:<14.2f} {p['predicted_price']:<14.2f} "
                      f"{p['expected_change_pct']:+.2f}%")

        # ── Karar Özeti ──
        print("\n📊 KARAR DESTEK ÖZETİ")
        print("-" * 85)
        rec_daily = self.predictor.get_recommendation(pred_daily, min_confidence=0.60)
        # Sadece özet satırlarını al
        for line in rec_daily.strip().split('\n'):
            if line.strip() and '=' not in line:
                print(f"   {line.strip()}")

        # ── Geçmiş Trendi ──
        if len(self.history) > 1:
            print("\n📈 SON TAHMİN GEÇMİŞİ (1d_daily)")
            print("-" * 85)
            for rec in self.history[-6:]:
                t = rec['time']
                d = rec['daily'].get('1d_daily', {})
                direction = d.get('direction', '?')
                conf = d.get('confidence', 0)
                price = d.get('current_price', 0)
                bar = "█" * int(conf * 10)
                print(f"  {t} | {direction:<10} | Güven: {conf:.0%} {bar:<10} | Fiyat: {price:.2f}")

        print("\n" + "=" * 85)
        print(f"  Bir sonraki güncelleme: ~{self.interval_sec} sn  |  Çıkış: Ctrl+C")
        print("=" * 85)

    def _save_latest(self, record: Dict):
        """Son tahmini JSONL dosyasına ekle."""
        out = Path(self.model_dir) / 'live_monitor_history.jsonl'
        try:
            with open(out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _show_summary(self):
        """Durdurulduğunda özet göster."""
        if not self.history:
            return
        print("\n📋 OTURUM ÖZETİ")
        print("-" * 60)
        up_count = sum(1 for r in self.history
                       if 'YUKARI' in r['daily'].get('1d_daily', {}).get('direction', ''))
        down_count = len(self.history) - up_count
        print(f"   Toplam tahmin: {len(self.history)}")
        print(f"   Yukarı sinyali: {up_count}")
        print(f"   Aşağı sinyali: {down_count}")
        print(f"   Geçmiş dosyası: gmstr_models/live_monitor_history.jsonl")
        print("-" * 60)
