#!/usr/bin/env python3
"""
GMSTR Masaüstü Tahmin Uygulaması
- Her 5 dakikada otomatik güncelleme
- Renk kodlama (Yeşil=Yukarı, Kırmızı=Aşağı)
- Saatlik yön değişikliğinde Telegram bildirimi
- Tek dosya, .exe'ye derlenebilir
"""
import os
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Tkinter
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# GMSTR modülleri
sys.path.insert(0, str(Path(__file__).parent))
from gmstr_system.predictor import GMSTRPredictor
from gmstr_system.notifier import TelegramNotifier
from gmstr_system.price_fetcher import GMSTRPriceFetcher


# ================================================================
# TEMA RENKLERİ
# ================================================================
COLORS = {
    'bg': '#0D1117',
    'card': '#161B22',
    'card_border': '#30363D',
    'text': '#E6EDF3',
    'text_secondary': '#8B949E',
    'accent': '#58A6FF',
    'up': '#00C853',
    'up_dark': '#00E676',
    'down': '#FF1744',
    'down_dark': '#FF5252',
    'warning': '#FFA726',
    'header': '#21262D',
}


class GMSTRDesktopApp:
    """GMSTR Masaüstü Tahmin Uygulaması Ana Sınıfı."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GMSTR AI Tahmin Monitörü")
        self.root.geometry("900x750")
        self.root.configure(bg=COLORS['bg'])
        self.root.resizable(True, True)

        # State
        import sys
        if hasattr(sys, '_MEIPASS'):
            model_dir = Path(sys._MEIPASS) / 'gmstr_models'
        else:
            model_dir = 'gmstr_models'
        self.predictor = GMSTRPredictor(model_dir=str(model_dir))
        self.notifier = TelegramNotifier()
        self.last_predictions: Dict[str, Dict] = {}
        self.last_hourly_predictions: Dict[str, Dict] = {}
        self.countdown_sec = 300
        self.running = True

        # Fonts
        self.font_large = ('Segoe UI', 28, 'bold')
        self.font_title = ('Segoe UI', 14, 'bold')
        self.font_normal = ('Segoe UI', 11)
        self.font_small = ('Segoe UI', 9)
        self.font_mono = ('Consolas', 10)

        self._build_ui()
        self._initial_load()
        self._start_threads()

    # ================================================================
    # UI OLUŞTURMA
    # ================================================================
    def _build_ui(self):
        # Ana container
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Üst başlık
        header = tk.Frame(main_frame, bg=COLORS['header'], height=60)
        header.pack(fill=tk.X, pady=(0, 12))
        header.pack_propagate(False)

        tk.Label(header, text="GMSTR (Gümüş BYF)", font=self.font_title,
                 bg=COLORS['header'], fg=COLORS['text']).pack(side=tk.LEFT, padx=16, pady=8)
        tk.Label(header, text="AI Tahmin Sistemi", font=self.font_normal,
                 bg=COLORS['header'], fg=COLORS['text_secondary']).pack(side=tk.LEFT, pady=8)

        self.lbl_status = tk.Label(header, text="Başlatılıyor...", font=self.font_small,
                                    bg=COLORS['header'], fg=COLORS['accent'])
        self.lbl_status.pack(side=tk.RIGHT, padx=16, pady=8)

        # Notebook (Tablar)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TNotebook', background=COLORS['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', font=self.font_normal, padding=(16, 8),
                        background=COLORS['card'], foreground=COLORS['text'])
        style.map('TNotebook.Tab', background=[('selected', COLORS['accent'])],
                  foreground=[('selected', '#FFFFFF')])

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Tahminler
        self.tab_predictions = tk.Frame(self.notebook, bg=COLORS['bg'])
        self.notebook.add(self.tab_predictions, text="Tahminler")
        self._build_predictions_tab()

        # Tab 2: Ayarlar
        self.tab_settings = tk.Frame(self.notebook, bg=COLORS['bg'])
        self.notebook.add(self.tab_settings, text="Ayarlar")
        self._build_settings_tab()

        # Alt bilgi çubuğu
        footer = tk.Frame(main_frame, bg=COLORS['header'], height=30)
        footer.pack(fill=tk.X, pady=(12, 0))
        footer.pack_propagate(False)

        self.lbl_countdown = tk.Label(footer, text="Sonraki: 05:00", font=self.font_mono,
                                       bg=COLORS['header'], fg=COLORS['text_secondary'])
        self.lbl_countdown.pack(side=tk.RIGHT, padx=16, pady=4)

        self.lbl_last_update = tk.Label(footer, text="Son Güncelleme: -", font=self.font_small,
                                         bg=COLORS['header'], fg=COLORS['text_secondary'])
        self.lbl_last_update.pack(side=tk.LEFT, padx=16, pady=4)

    def _build_predictions_tab(self):
        tab = self.tab_predictions

        # Son fiyat banner
        self.price_frame = tk.Frame(tab, bg=COLORS['card'], highlightbackground=COLORS['card_border'],
                                     highlightthickness=1)
        self.price_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(self.price_frame, text="SON FİYAT", font=self.font_small,
                 bg=COLORS['card'], fg=COLORS['text_secondary']).pack(pady=(12, 0))
        self.lbl_current_price = tk.Label(self.price_frame, text="---", font=self.font_large,
                                           bg=COLORS['card'], fg=COLORS['text'])
        self.lbl_current_price.pack(pady=(0, 12))

        # Günlük tahmin kartları
        daily_title = tk.Label(tab, text="Günlük Tahminler", font=self.font_title,
                                bg=COLORS['bg'], fg=COLORS['text'])
        daily_title.pack(anchor=tk.W, pady=(8, 8))

        self.daily_frame = tk.Frame(tab, bg=COLORS['bg'])
        self.daily_frame.pack(fill=tk.X, pady=(0, 12))
        self.daily_cards: Dict[str, tk.Widget] = {}
        for h in ['1d_daily', '3d_daily', '5d_daily', '10d_daily']:
            self.daily_cards[h] = self._create_prediction_card(self.daily_frame, h)

        # Saatlik tahmin kartları
        hourly_title = tk.Label(tab, text="Saatlik Tahminler", font=self.font_title,
                                 bg=COLORS['bg'], fg=COLORS['text'])
        hourly_title.pack(anchor=tk.W, pady=(8, 8))

        self.hourly_frame = tk.Frame(tab, bg=COLORS['bg'])
        self.hourly_frame.pack(fill=tk.X)
        self.hourly_cards: Dict[str, tk.Widget] = {}
        for h in ['1h_hourly', '4h_hourly']:
            self.hourly_cards[h] = self._create_prediction_card(self.hourly_frame, h)

        # Bildirim logu
        log_title = tk.Label(tab, text="Bildirim Geçmişi", font=self.font_title,
                              bg=COLORS['bg'], fg=COLORS['text'])
        log_title.pack(anchor=tk.W, pady=(16, 8))

        self.txt_log = scrolledtext.ScrolledText(tab, height=6, font=self.font_mono,
                                                   bg=COLORS['card'], fg=COLORS['text'],
                                                   insertbackground=COLORS['text'],
                                                   state=tk.DISABLED)
        self.txt_log.pack(fill=tk.X)

    def _create_prediction_card(self, parent, horizon: str) -> Dict:
        """Tek tahmin kartı oluştur."""
        card = tk.Frame(parent, bg=COLORS['card'], highlightbackground=COLORS['card_border'],
                        highlightthickness=1, width=200, height=140)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)
        card.pack_propagate(False)

        lbl_name = tk.Label(card, text=horizon.replace('_', ' ').upper(), font=self.font_small,
                            bg=COLORS['card'], fg=COLORS['text_secondary'])
        lbl_name.pack(pady=(10, 2))

        lbl_dir = tk.Label(card, text="BEKLENİYOR", font=self.font_title,
                           bg=COLORS['card'], fg=COLORS['text'])
        lbl_dir.pack(pady=(2, 2))

        lbl_conf = tk.Label(card, text="Güven: --", font=self.font_normal,
                            bg=COLORS['card'], fg=COLORS['text_secondary'])
        lbl_conf.pack(pady=(2, 2))

        lbl_pred = tk.Label(card, text="Tahmini: --", font=self.font_small,
                            bg=COLORS['card'], fg=COLORS['text_secondary'])
        lbl_pred.pack(pady=(2, 10))

        return {
            'frame': card,
            'name': lbl_name,
            'direction': lbl_dir,
            'confidence': lbl_conf,
            'predicted': lbl_pred,
        }

    def _build_settings_tab(self):
        tab = self.tab_settings

        frame = tk.Frame(tab, bg=COLORS['card'], highlightbackground=COLORS['card_border'],
                         highlightthickness=1)
        frame.pack(fill=tk.X, pady=16, padx=8)

        tk.Label(frame, text="Telegram Bildirim Ayarları", font=self.font_title,
                 bg=COLORS['card'], fg=COLORS['text']).pack(anchor=tk.W, padx=16, pady=(16, 12))

        # Bot Token
        tk.Label(frame, text="Bot Token:", font=self.font_normal,
                 bg=COLORS['card'], fg=COLORS['text_secondary']).pack(anchor=tk.W, padx=16)
        self.entry_token = tk.Entry(frame, font=self.font_mono, width=60,
                                    bg=COLORS['bg'], fg=COLORS['text'],
                                    insertbackground=COLORS['text'])
        self.entry_token.pack(fill=tk.X, padx=16, pady=(4, 12))

        # Chat ID
        tk.Label(frame, text="Chat ID:", font=self.font_normal,
                 bg=COLORS['card'], fg=COLORS['text_secondary']).pack(anchor=tk.W, padx=16)
        self.entry_chatid = tk.Entry(frame, font=self.font_mono, width=60,
                                      bg=COLORS['bg'], fg=COLORS['text'],
                                      insertbackground=COLORS['text'])
        self.entry_chatid.pack(fill=tk.X, padx=16, pady=(4, 12))

        btn_frame = tk.Frame(frame, bg=COLORS['card'])
        btn_frame.pack(fill=tk.X, padx=16, pady=(8, 16))

        btn_save = tk.Button(btn_frame, text="Kaydet", font=self.font_normal,
                              bg=COLORS['accent'], fg='#FFFFFF',
                              activebackground='#79B8FF', borderwidth=0,
                              padx=20, pady=6, cursor='hand2',
                              command=self._save_telegram_settings)
        btn_save.pack(side=tk.LEFT, padx=(0, 8))

        btn_test = tk.Button(btn_frame, text="Test Mesajı Gönder", font=self.font_normal,
                              bg=COLORS['card'], fg=COLORS['text'],
                              activebackground=COLORS['header'], borderwidth=1,
                              highlightbackground=COLORS['card_border'],
                              padx=20, pady=6, cursor='hand2',
                              command=self._test_telegram)
        btn_test.pack(side=tk.LEFT)

        # Bilgi metni
        info = (
            "Nasıl kullanılır:\n"
            "1. Telegram'da @BotFather ile yeni bot oluşturun\n"
            "2. Bot token'ını kopyalayıp yukarıya yapıştırın\n"
            "3. @userinfobot'a gidip chat ID'nizi alın\n"
            "4. Chat ID'yi yukarıya yapıştırın\n"
            "5. 'Kaydet' ve ardından 'Test Mesajı Gönder' ile doğrulayın"
        )
        tk.Label(frame, text=info, font=self.font_small, justify=tk.LEFT,
                 bg=COLORS['card'], fg=COLORS['text_secondary']).pack(anchor=tk.W, padx=16, pady=(8, 16))

        # Mevcut ayarları yükle
        if self.notifier.bot_token:
            self.entry_token.insert(0, self.notifier.bot_token)
        if self.notifier.chat_id:
            self.entry_chatid.insert(0, str(self.notifier.chat_id))

    # ================================================================
    # İŞLEMLER
    # ================================================================
    def _initial_load(self):
        """İlk yükleme - arka planda çalıştır."""
        self.lbl_status.config(text="Modeller yükleniyor...", fg=COLORS['warning'])
        self.root.update()

        try:
            ok = self.predictor.load_system()
            if ok:
                self.lbl_status.config(text="Hazır", fg=COLORS['up'])
                self._refresh_once()
            else:
                self.lbl_status.config(text="Model yok! Önce eğitim yapın.", fg=COLORS['down'])
        except Exception as e:
            self.lbl_status.config(text=f"Hata: {e}", fg=COLORS['down'])

    def _refresh_once(self):
        """Tek seferlik tahmin güncelleme."""
        try:
            # ÖNCE canlı fiyatı çek ve banner'ı hemen güncelle
            live_price_data = GMSTRPriceFetcher.get_price_with_fallback()
            if live_price_data['price'] is not None:
                price_text = f"{live_price_data['price']:,.2f} TRY"
                if live_price_data['source'] == "API (Yahoo Finance)":
                    price_text += "  🌐"
                self.lbl_current_price.config(text=price_text)
                self._log(f"[FİYAT] {live_price_data['source']}: {live_price_data['price']:.2f} TRY")

            # Sonra tahminleri üret
            pred_daily = self.predictor.predict(is_hourly=False)
            pred_hourly = self.predictor.predict(is_hourly=True)

            self._update_ui(pred_daily, pred_hourly)
            self._check_direction_changes(pred_hourly)

            self.last_predictions = pred_daily
            self.last_hourly_predictions = pred_hourly

            now_str = datetime.now().strftime("%H:%M:%S")
            self.lbl_last_update.config(text=f"Son Güncelleme: {now_str}")
            self._log(f"[{now_str}] Tahminler güncellendi.")
        except Exception as e:
            self._log(f"[HATA] Güncelleme başarısız: {e}")

    def _update_ui(self, daily: Dict, hourly: Dict):
        """Tahmin kartlarını güncelle."""
        # Son fiyat
        if daily:
            first = next(iter(daily.values()))
            price = first.get('current_price', 0)
            self.lbl_current_price.config(text=f"{price:,.2f} TRY")

        # Günlük kartlar
        for h_name, card in self.daily_cards.items():
            if h_name in daily:
                self._set_card(card, daily[h_name])
            else:
                card['direction'].config(text="N/A", fg=COLORS['text_secondary'])

        # Saatlik kartlar
        for h_name, card in self.hourly_cards.items():
            if h_name in hourly:
                self._set_card(card, hourly[h_name])
            else:
                card['direction'].config(text="N/A", fg=COLORS['text_secondary'])

    def _set_card(self, card: Dict, pred: Dict):
        """Kart içeriğini ve rengini ayarla."""
        direction = pred.get('direction', '')
        confidence = pred.get('confidence', 0)
        pred_price = pred.get('predicted_price', 0)
        change = pred.get('expected_change_pct', 0)

        is_up = 'YUKARI' in direction
        color = COLORS['up'] if is_up else COLORS['down']
        text = "YUKARI ↑" if is_up else "AŞAĞI ↓"

        card['direction'].config(text=text, fg=color)
        card['confidence'].config(text=f"Güven: {confidence:.0%}", fg=color)
        card['predicted'].config(
            text=f"Tahmini: {pred_price:,.2f} ({change:+.2f}%)",
            fg=COLORS['text_secondary']
        )
        card['frame'].config(highlightbackground=color)

    def _check_direction_changes(self, hourly: Dict):
        """Saatlik tahminlerde yön değişikliği kontrolü."""
        if not self.last_hourly_predictions:
            return
        for h_name, pred in hourly.items():
            old = self.last_hourly_predictions.get(h_name)
            if old is None:
                continue
            old_up = 'YUKARI' in old.get('direction', '')
            new_up = 'YUKARI' in pred.get('direction', '')
            if old_up != new_up:
                self._notify_change(h_name, old, pred)

    def _notify_change(self, horizon: str, old: Dict, new: Dict):
        """Yön değişikliği bildirimi gönder."""
        ok = self.notifier.notify_direction_change(
            horizon=horizon,
            old_dir=old.get('direction', '?'),
            new_dir=new.get('direction', '?'),
            confidence=new.get('confidence', 0),
            current_price=new.get('current_price', 0),
            predicted_price=new.get('predicted_price', 0),
        )
        if ok:
            self._log(f"[BİLDİRİM] {horizon} yön değişikliği gönderildi.")
        else:
            self._log(f"[BİLDİRİM] {horizon} yön değişikliği başarısız (Telegram kapalı?).")

    def _log(self, msg: str):
        """Log alanına mesaj yaz."""
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    # ================================================================
    # THREAD'LER
    # ================================================================
    def _start_threads(self):
        threading.Thread(target=self._countdown_loop, daemon=True).start()
        threading.Thread(target=self._auto_refresh_loop, daemon=True).start()

    def _countdown_loop(self):
        while self.running:
            mins, secs = divmod(self.countdown_sec, 60)
            text = f"Sonraki: {mins:02d}:{secs:02d}"
            try:
                self.lbl_countdown.config(text=text)
            except tk.TclError:
                break
            time.sleep(1)
            self.countdown_sec -= 1
            if self.countdown_sec <= 0:
                self.countdown_sec = 300

    def _auto_refresh_loop(self):
        while self.running:
            time.sleep(300)  # 5 dk
            if self.running:
                try:
                    self.root.after(0, self._refresh_once)
                except tk.TclError:
                    break

    # ================================================================
    # AYARLAR
    # ================================================================
    def _save_telegram_settings(self):
        token = self.entry_token.get().strip()
        chat_id = self.entry_chatid.get().strip()
        if not token or not chat_id:
            messagebox.showwarning("Eksik Bilgi", "Bot Token ve Chat ID girilmeli.")
            return
        self.notifier.bot_token = token
        self.notifier.chat_id = chat_id
        self.notifier.save_config()
        messagebox.showinfo("Başarılı", "Telegram ayarları kaydedildi.")

    def _test_telegram(self):
        ok = self.notifier.test_connection()
        if ok:
            messagebox.showinfo("Başarılı", "Test mesajı gönderildi!")
        else:
            messagebox.showerror("Hata", "Mesaj gönderilemedi. Token ve Chat ID'yi kontrol edin.")

    # ================================================================
    # ÇALIŞTIRMA
    # ================================================================
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        self.root.destroy()


def main():
    app = GMSTRDesktopApp()
    app.run()


if __name__ == '__main__':
    main()
