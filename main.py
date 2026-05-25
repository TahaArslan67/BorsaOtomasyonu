"""
BorsaBot Mobil Uygulama - Kivy (Python)
========================================
GMSTR (Gumus/Altin BYF) ve Halka Arz Tahmin Motoru
iki ayri sekmede (TabbedPanel) sunulur.

Gereksinimler:
    pip install kivy

Calistirma:
    python mobile_combined_app.py
"""
import os
os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle

# Halka Arz sistemi importlari
from halka_arz_system.data_model import (
    HalkaArzGirdileri, DagitimYontemi, Sektorm, PiyasaDuyarliligi
)
from halka_arz_system.predictor import HalkaArzPredictor
from halka_arz_system.backtest import backtest_yap
from halka_arz_system.historical_data import get_gecmis_veriler

Window.clearcolor = (0.06, 0.08, 0.12, 1)


class Tema:
    ARKA_PLAN = (0.06, 0.08, 0.12, 1)
    KART = (0.10, 0.13, 0.18, 1)
    KART_ACIK = (0.13, 0.17, 0.24, 1)
    YESIL = (0.20, 0.78, 0.55, 1)
    KIRMIZI = (0.95, 0.30, 0.30, 1)
    TURUNCU = (1.0, 0.65, 0.15, 1)
    MAVI = (0.25, 0.55, 0.95, 1)
    BEYAZ = (0.95, 0.96, 0.98, 1)
    GRI = (0.55, 0.58, 0.62, 1)


def rlabel(text, renk=Tema.BEYAZ, font_size="14sp", bold=False, height=30, size_hint_x=None):
    return Label(
        text=text, color=renk, font_size=font_size, bold=bold,
        size_hint_y=None, height=height, size_hint_x=size_hint_x,
        halign="left", valign="middle", text_size=(None, None),
    )


class Kart(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 12
        self.spacing = 8
        self.size_hint_y = None
        with self.canvas.before:
            Color(*Tema.KART)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[12])
        self.bind(pos=self._upd, size=self._upd)

    def _upd(self, inst, val):
        self.rect.pos = inst.pos
        self.rect.size = inst.size


class HalkaArzTab(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 10
        self.spacing = 10
        self.predictor = HalkaArzPredictor()

        self.add_widget(rlabel(
            "HALKA ARZ TAVAN TAHMIN MOTORU",
            renk=Tema.YESIL, font_size="18sp", bold=True, height=40
        ))

        scroll = ScrollView(size_hint=(1, 1))
        self.form_layout = GridLayout(cols=1, spacing=12, size_hint_y=None, padding=10)
        self.form_layout.bind(minimum_height=self.form_layout.setter("height"))

        # Girdi Karti
        girdi = Kart()
        girdi.add_widget(rlabel("Girdiler", renk=Tema.MAVI, font_size="16sp", bold=True, height=30))

        self.input_sirket = TextInput(hint_text="Orn: Yeni Sirket A.S.")
        girdi.add_widget(self._satir("Sirket Adi:", self.input_sirket))

        self.spin_dagitim = Spinner(
            text="Bireysele Esit",
            values=["Bireysele Esit", "Tamami Esit", "Oransal", "Halka Arz Fonu", "Karma"],
            size_hint_y=None, height=40, background_color=Tema.KART_ACIK, color=Tema.BEYAZ,
        )
        girdi.add_widget(self._satir("Dagitim Yontemi:", self.spin_dagitim))

        self.input_boyut = TextInput(hint_text="TL (orn: 100000000)", input_filter="float")
        girdi.add_widget(self._satir("Arz Boyutu (TL):", self.input_boyut))

        self.input_katilimci = TextInput(hint_text="Bin kisi (orn: 1000)", input_filter="int")
        girdi.add_widget(self._satir("Katilimci (K):", self.input_katilimci))

        self.input_kurumsal = TextInput(hint_text="0.0-1.0 (orn: 0.25)", input_filter="float")
        girdi.add_widget(self._satir("Kurumsal Oran:", self.input_kurumsal))

        taahhut = BoxLayout(orientation="horizontal", size_hint_y=None, height=30)
        taahhut.add_widget(rlabel("Kurumsal Taahhut:", renk=Tema.GRI, height=30))
        self.chk_taahhut = CheckBox(size_hint_x=None, width=40)
        taahhut.add_widget(self.chk_taahhut)
        girdi.add_widget(taahhut)

        self.spin_sektor = Spinner(
            text="Teknoloji",
            values=["Teknoloji", "Enerji", "Savunma", "Saglik", "Finans",
                    "Gida", "Uretim", "Insaat", "Hizmet", "Diger"],
            size_hint_y=None, height=40, background_color=Tema.KART_ACIK, color=Tema.BEYAZ,
        )
        girdi.add_widget(self._satir("Sektor:", self.spin_sektor))

        self.input_borc = TextInput(hint_text="0.0-1.0 (orn: 0.40)", input_filter="float")
        girdi.add_widget(self._satir("Borcluluk:", self.input_borc))

        self.input_kar = TextInput(hint_text="-1.0 ile +1.0 (orn: 0.20)", input_filter="float")
        girdi.add_widget(self._satir("Kar Buyumesi:", self.input_kar))

        self.spin_piyasa = Spinner(
            text="Boga",
            values=["Guclu Boga", "Boga", "Yatay", "Ayi", "Guclu Ayi"],
            size_hint_y=None, height=40, background_color=Tema.KART_ACIK, color=Tema.BEYAZ,
        )
        girdi.add_widget(self._satir("Piyasa:", self.spin_piyasa))

        self.input_lot = TextInput(hint_text="Opsiyonel (TL)", input_filter="float")
        girdi.add_widget(self._satir("Lot Maliyet:", self.input_lot))

        # Varsayilan degerler
        self.input_sirket.text = "Enerji Yatirim A.S."
        self.input_boyut.text = "85000000"
        self.input_katilimci.text = "1400"
        self.input_kurumsal.text = "0.25"
        self.input_borc.text = "0.35"
        self.input_kar.text = "0.25"
        self.input_lot.text = "1200"

        self.form_layout.add_widget(girdi)

        btn_tahmin = Button(
            text="TAHMIN YAP", font_size="16sp", bold=True,
            size_hint_y=None, height=50,
            background_color=Tema.YESIL, color=(0, 0, 0, 1),
        )
        btn_tahmin.bind(on_press=self.tahmin_yap)
        self.form_layout.add_widget(btn_tahmin)

        btn_back = Button(
            text="BACKTEST CALISTIR", font_size="14sp", bold=True,
            size_hint_y=None, height=45,
            background_color=Tema.TURUNCU, color=(0, 0, 0, 1),
        )
        btn_back.bind(on_press=self.backtest_calistir)
        self.form_layout.add_widget(btn_back)

        self.sonuc_kart = Kart()
        self.sonuc_kart.add_widget(rlabel(
            "Henüz tahmin yapilmadi", renk=Tema.GRI, font_size="14sp", height=30
        ))
        self.form_layout.add_widget(self.sonuc_kart)

        scroll.add_widget(self.form_layout)
        self.add_widget(scroll)

    def _satir(self, etiket, widget):
        row = BoxLayout(orientation="horizontal", size_hint_y=None, height=40, spacing=5)
        row.add_widget(rlabel(etiket, renk=Tema.GRI, height=40, size_hint_x=0.45))
        widget.size_hint_x = 0.55
        row.add_widget(widget)
        return row

    def tahmin_yap(self, inst):
        try:
            dagitim_map = {
                "Bireysele Esit": DagitimYontemi.BIREYSELE_ESIT,
                "Tamami Esit": DagitimYontemi.TAMAMI_ESIT,
                "Oransal": DagitimYontemi.ORANSAL,
                "Halka Arz Fonu": DagitimYontemi.HALKA_ARZ_FONU,
                "Karma": DagitimYontemi.KARMA,
            }
            sektor_map = {
                "Teknoloji": Sektorm.TEKNOLOJI, "Enerji": Sektorm.ENERJI,
                "Savunma": Sektorm.SAVUNMA, "Saglik": Sektorm.SAGLIK,
                "Finans": Sektorm.FINANS, "Gida": Sektorm.GIDA,
                "Uretim": Sektorm.URETIM, "Insaat": Sektorm.INSAAT,
                "Hizmet": Sektorm.HIZMET, "Diger": Sektorm.DIGER,
            }
            piyasa_map = {
                "Guclu Boga": PiyasaDuyarliligi.GUCJU_BOGA,
                "Boga": PiyasaDuyarliligi.BOGA,
                "Yatay": PiyasaDuyarliligi.YATAY,
                "Ayi": PiyasaDuyarliligi.AYI,
                "Guclu Ayi": PiyasaDuyarliligi.GUCJU_AYI,
            }

            girdi = HalkaArzGirdileri(
                sirket_adi=self.input_sirket.text or "Bilinmeyen",
                dagitim_yontemi=dagitim_map[self.spin_dagitim.text],
                halka_arz_boyutu_tl=float(self.input_boyut.text or 0),
                katilimci_beklentisi=int(self.input_katilimci.text or 0),
                kurumsal_oran=float(self.input_kurumsal.text or 0),
                kurumsal_taahhut=self.chk_taahhut.active,
                sektor=sektor_map[self.spin_sektor.text],
                borcluluk_orani=float(self.input_borc.text or 0),
                net_kar_buyumesi=float(self.input_kar.text or 0),
                piyasa_duyarliligi=piyasa_map[self.spin_piyasa.text],
                lot_basi_dusen_maliyet=float(self.input_lot.text) if self.input_lot.text else None,
            )
            tahmin = self.predictor.tahmin_yap(girdi)
            self._sonuc_goster(tahmin)
        except Exception as e:
            self._popup("Hata", f"Tahmin hatasi:\n{str(e)}")

    def _sonuc_goster(self, tahmin):
        self.sonuc_kart.clear_widgets()
        self.sonuc_kart.height = 420

        if tahmin.toplam_skor >= 80:
            skor_renk, emoji = Tema.YESIL, "GUCU"
        elif tahmin.toplam_skor >= 60:
            skor_renk, emoji = Tema.TURUNCU, "ORTA"
        elif tahmin.toplam_skor >= 40:
            skor_renk, emoji = Tema.MAVI, "ZAYIF"
        else:
            skor_renk, emoji = Tema.KIRMIZI, "RISKI"

        self.sonuc_kart.add_widget(rlabel(
            f"{emoji} {tahmin.sirket_adi}",
            renk=Tema.BEYAZ, font_size="16sp", bold=True, height=30
        ))

        skor_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=30)
        skor_row.add_widget(rlabel("Skor:", renk=Tema.GRI, height=30, size_hint_x=0.2))
        skor_row.add_widget(ProgressBar(max=100, value=tahmin.toplam_skor, size_hint_y=None, height=20))
        skor_row.add_widget(rlabel(
            f"{tahmin.toplam_skor:.1f}/100", renk=skor_renk,
            font_size="14sp", bold=True, height=30, size_hint_x=0.25
        ))
        self.sonuc_kart.add_widget(skor_row)

        self.sonuc_kart.add_widget(rlabel(
            f"Kategori: {tahmin.kategori.value.upper()}",
            renk=skor_renk, font_size="14sp", bold=True, height=25
        ))
        self.sonuc_kart.add_widget(rlabel(
            f"Tahmin: {tahmin.tahmin_gun_araligi}",
            renk=Tema.BEYAZ, font_size="14sp", bold=True, height=25
        ))
        self.sonuc_kart.add_widget(rlabel(
            f"Guven: %{tahmin.guven_orani:.0f}", renk=Tema.GRI, font_size="13sp", height=22
        ))
        self.sonuc_kart.add_widget(rlabel("Tavsiye:", renk=Tema.TURUNCU, font_size="13sp", bold=True, height=22))
        self.sonuc_kart.add_widget(rlabel(tahmin.tavsiye, renk=Tema.BEYAZ, font_size="12sp", height=50))

        self.sonuc_kart.add_widget(rlabel("Detayli Skorlar:", renk=Tema.MAVI, font_size="13sp", bold=True, height=22))
        for skor in tahmin.detayli_skorlar:
            self.sonuc_kart.add_widget(rlabel(
                f"  {skor.parametre_adi:<20} {skor.normalize_skor:>5.1f} (Katki: {skor.agirlikli_skor:>5.2f})",
                renk=Tema.GRI, font_size="11sp", height=20
            ))

        self.sonuc_kart.add_widget(rlabel("Analiz:", renk=Tema.MAVI, font_size="12sp", bold=True, height=22))
        self.sonuc_kart.add_widget(rlabel(tahmin.aciklama, renk=Tema.GRI, font_size="11sp", height=80))

    def backtest_calistir(self, inst):
        try:
            veriler = get_gecmis_veriler()
            rapor = backtest_yap(veriler, self.predictor)
            icerik = (
                f"BACKTEST SONUCLARI\n{'='*40}\n"
                f"Toplam Ornek: {rapor.toplam_ornek}\n"
                f"Kategori Dogruluk: %{rapor.kategori_dogruluk_orani:.1f}\n"
                f"Ort. Gun Hatasi: {rapor.ortalama_gun_hatasi:.1f} gun\n"
                f"MAE: {rapor.mae:.1f} gun\n"
                f"RMSE: {rapor.rmse:.1f} gun\n\nDetaylar:\n"
            )
            for s in rapor.detayli_sonuclar:
                durum = "OK" if s.kategori_dogru else "X"
                icerik += f"{durum} {s.sirket_adi:<18} Skor:{s.tahmin_skor:>5.1f} Gercek:{s.gercek_tavan_gunu}g\n"
            self._popup("Backtest Raporu", icerik)
        except Exception as e:
            self._popup("Hata", f"Backtest hatasi:\n{str(e)}")

    def _popup(self, baslik, mesaj):
        content = BoxLayout(orientation="vertical", padding=15, spacing=10)
        content.add_widget(Label(text=mesaj, color=Tema.BEYAZ, font_size="13sp",
                                 halign="left", valign="top", text_size=(350, None)))
        btn = Button(text="KAPAT", size_hint_y=None, height=40,
                     background_color=Tema.KIRMIZI, color=(1, 1, 1, 1))
        content.add_widget(btn)
        popup = Popup(title=baslik, content=content, size_hint=(None, None),
                      size=(420, 520), auto_dismiss=False,
                      title_color=Tema.BEYAZ, background_color=Tema.KART)
        btn.bind(on_press=popup.dismiss)
        popup.open()


class GMSTRTab(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 15
        self.spacing = 12

        self.add_widget(rlabel(
            "GMSTR (Gumus BYF) SISTEMI",
            renk=Tema.MAVI, font_size="18sp", bold=True, height=40
        ))

        info = Kart()
        info.add_widget(rlabel("GMSTR Modulu", renk=Tema.TURUNCU, font_size="16sp", bold=True, height=30))
        info.add_widget(rlabel("Bu sekme GMSTR BYF tahmin sistemi icin ayrilmistir.",
                               renk=Tema.GRI, font_size="13sp", height=25))
        info.add_widget(rlabel("Kullanilabilir islemler:", renk=Tema.BEYAZ, font_size="13sp", bold=True, height=25))
        info.add_widget(rlabel("- Model Egitimi: python -m gmstr_system.main --mode train",
                               renk=Tema.GRI, font_size="12sp", height=22))
        info.add_widget(rlabel("- Tahmin: python -m gmstr_system.main --mode predict",
                               renk=Tema.GRI, font_size="12sp", height=22))
        info.add_widget(rlabel("- Canli Monitor: python -m gmstr_system.main --mode live",
                               renk=Tema.GRI, font_size="12sp", height=22))

        # GMSTR canli skor (simulasyon)
        info.add_widget(rlabel("Son GMSTR Tahmini (Ornek):", renk=Tema.YESIL,
                               font_size="13sp", bold=True, height=25))
        info.add_widget(rlabel("Yon: YUKARI | Guven: %72 | Vade: 1G",
                               renk=Tema.BEYAZ, font_size="13sp", height=22))
        self.add_widget(info)

        btn_gmstr = Button(
            text="GMSTR TAHMINLERINI GUNCELLE", font_size="14sp", bold=True,
            size_hint_y=None, height=45,
            background_color=Tema.MAVI, color=(1, 1, 1, 1),
        )
        btn_gmstr.bind(on_press=self._gmstr_guncelle)
        self.add_widget(btn_gmstr)

        self.gmstr_sonuc = rlabel("GMSTR tahmini burada gorunecek...", renk=Tema.GRI, font_size="13sp", height=30)
        self.add_widget(self.gmstr_sonuc)

    def _gmstr_guncelle(self, inst):
        self.gmstr_sonuc.text = "GMSTR tahmini: Yukari trend | Guven: %75 (Simulasyon)"


class AnaTabPanel(TabbedPanel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_default_tab = False
        self.tab_pos = "top_mid"
        self.background_color = Tema.ARKA_PLAN

        tab_halka = TabbedPanelHeader(text="Halka Arz")
        tab_halka.content = HalkaArzTab()
        self.add_widget(tab_halka)

        tab_gmstr = TabbedPanelHeader(text="GMSTR")
        tab_gmstr.content = GMSTRTab()
        self.add_widget(tab_gmstr)

        self.switch_to(tab_halka)


class BorsaBotApp(App):
    def build(self):
        self.title = "BorsaBot - Halka Arz & GMSTR"
        return AnaTabPanel()


if __name__ == "__main__":
    BorsaBotApp().run()
