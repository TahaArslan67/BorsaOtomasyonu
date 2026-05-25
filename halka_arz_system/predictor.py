"""
Halka Arz Tavan Serisi Tahmin Motoru
=====================================
Matematiksel model, parametrik skorlama ve ağırlıklandırma sistemi.

Ağırlıklandırma (Toplam: %100):
1. Dağıtım Yöntemi          : %25
2. Halka Arz Büyüklüğü      : %20
3. Katılımcı Sayısı         : %15
4. Kurumsal Yatırımcı Oranı : %15
5. Sektör ve Finansallar    : %15
6. Piyasa Duyarlılığı       : %10
"""
import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .data_model import (
    HalkaArzGirdileri,
    ParametreSkoru,
    TavanTahmini,
    TavanKategorisi,
    DagitimYontemi,
    Sektorm,
    PiyasaDuyarliligi,
)


# ──────────────────────────────────────────────────────────────────────────────
# AĞIRLIKLAR
# ──────────────────────────────────────────────────────────────────────────────
AGIRLIKLAR = {
    "dagitim_yontemi": 0.25,
    "halka_arz_boyutu": 0.20,
    "katilimci_sayisi": 0.15,
    "kurumsal_yatirimci": 0.15,
    "sektor_finansallar": 0.15,
    "piyasa_duyarliligi": 0.10,
}


def _normalize_dagitim(dagitim: DagitimYontemi) -> Tuple[float, str]:
    """
    Dağıtım yöntemi skoru:
    - Bireysele eşit: küçük yatırımcıya az lot, maliyet düşük,
      satış baskısı azalır -> Yüksek skor
    - Oransal: büyük yatırımcı talep eder, satış baskısı yüksek -> Düşük skor
    """
    skorlar = {
        DagitimYontemi.BIREYSELE_ESIT: 100,
        DagitimYontemi.TAMAMI_ESIT: 90,
        DagitimYontemi.KARMA: 75,
        DagitimYontemi.HALKA_ARZ_FONU: 60,
        DagitimYontemi.ORANSAL: 45,
    }
    skor = skorlar.get(dagitim, 50)
    aciklama = f"{dagitim.value} dağıtımı -> skor={skor}"
    return skor, aciklama


def _normalize_boyut(boyut_tl: float) -> Tuple[float, str]:
    """
    Halka arz büyüklüğü skoru:
    Küçük tahta = manipülasyon/destek kolay = uzun tavan.
    """
    if boyut_tl <= 50_000_000:          # 50M TL altı
        skor = 100
        aciklama = f"Çok küçük tahta ({boyut_tl:,.0f} TL) -> skor=100"
    elif boyut_tl <= 100_000_000:       # 50-100M
        skor = 90
        aciklama = f"Küçük tahta ({boyut_tl:,.0f} TL) -> skor=90"
    elif boyut_tl <= 250_000_000:       # 100-250M
        skor = 75
        aciklama = f"Orta tahta ({boyut_tl:,.0f} TL) -> skor=75"
    elif boyut_tl <= 500_000_000:       # 250-500M
        skor = 60
        aciklama = f"Büyük tahta ({boyut_tl:,.0f} TL) -> skor=60"
    elif boyut_tl <= 1_000_000_000:     # 500M-1B
        skor = 45
        aciklama = f"Çok büyük tahta ({boyut_tl:,.0f} TL) -> skor=45"
    else:                               # 1B+
        skor = 30
        aciklama = f"Dev tahta ({boyut_tl:,.0f} TL) -> skor=30"
    return skor, aciklama


def _normalize_katilimci(katilimci_bin: int) -> Tuple[float, str]:
    """
    Katılımcı sayısı skoru:
    Çok yüksek katılımcı = talep patlaması = uzun tavan.
    Aşırı yüksek = satış baskısı riski (azaltma uygulanır).
    """
    if katilimci_bin >= 1500:           # 1.5M+
        skor = 100
        aciklama = f"Aşırı yüksek katılımcı ({katilimci_bin:,}K) -> skor=100"
    elif katilimci_bin >= 1000:         # 1M-1.5M
        skor = 95
        aciklama = f"Çok yüksek katılımcı ({katilimci_bin:,}K) -> skor=95"
    elif katilimci_bin >= 700:          # 700K-1M
        skor = 88
        aciklama = f"Yüksek katılımcı ({katilimci_bin:,}K) -> skor=88"
    elif katilimci_bin >= 500:          # 500-700K
        skor = 80
        aciklama = f"Güçlü katılımcı ({katilimci_bin:,}K) -> skor=80"
    elif katilimci_bin >= 300:          # 300-500K
        skor = 70
        aciklama = f"Orta-yüksek katılımcı ({katilimci_bin:,}K) -> skor=70"
    elif katilimci_bin >= 150:          # 150-300K
        skor = 58
        aciklama = f"Orta katılımcı ({katilimci_bin:,}K) -> skor=58"
    elif katilimci_bin >= 80:           # 80-150K
        skor = 45
        aciklama = f"Düşük katılımcı ({katilimci_bin:,}K) -> skor=45"
    else:
        skor = 30
        aciklama = f"Çok düşük katılımcı ({katilimci_bin:,}K) -> skor=30"
    return skor, aciklama


def _normalize_kurumsal(kurumsal_oran: float, taahhut: bool) -> Tuple[float, str]:
    """
    Kurumsal yatırımcı skoru:
    - Dengeli kurumsal pay + taahhüt = en güvenilir
    - Sadece bireysel = spekülatif, riskli
    """
    base_skor = 0
    if 0.30 <= kurumsal_oran <= 0.60:
        base_skor = 90                    # Dengeli
    elif kurumsal_oran > 0.60:
        base_skor = 75                    # Kurumsal ağırlıklı (düşük likidite riski)
    elif kurumsal_oran >= 0.15:
        base_skor = 65                    # Orta
    elif kurumsal_oran > 0:
        base_skor = 50                    # Düşük
    else:
        base_skor = 35                    # Sadece bireysel

    if taahhut:
        base_skor = min(100, base_skor + 10)
        aciklama = f"Kurumsal=%{kurumsal_oran*100:.0f}, Taahhüt=VAR -> skor={base_skor}"
    else:
        aciklama = f"Kurumsal=%{kurumsal_oran*100:.0f}, Taahhüt=YOK -> skor={base_skor}"
    return base_skor, aciklama


def _normalize_sektor_finansal(
    sektor: Sektorm,
    borcluluk: float,
    kar_buyumesi: float,
) -> Tuple[float, str]:
    """
    Sektör ve finansal sağlık skoru.
    
    Sektör ağırlıkları (popülerlik/ilgi):
    - Teknoloji, Enerji, Savunma: en yüksek
    - Finans, Sağlık: yüksek
    - Gıda, Üretim: orta
    - İnşaat, Hizmet, Diğer: düşük
    """
    sektor_skorlari = {
        Sektorm.TEKNOLOJI: 100,
        Sektorm.ENERJI: 95,
        Sektorm.SAVUNMA: 92,
        Sektorm.SAGLIK: 85,
        Sektorm.FINANS: 80,
        Sektorm.GIDA: 70,
        Sektorm.URETIM: 65,
        Sektorm.HIZMET: 55,
        Sektorm.INSAAT: 50,
        Sektorm.DIGER: 45,
    }
    sektor_skor = sektor_skorlari.get(sektor, 50)

    # Finansal sağlık ayarlaması
    finansal_ayarlama = 0
    
    # Borçluluk: < %40 ideal, > %80 riskli
    if borcluluk < 0.30:
        finansal_ayarlama += 8
    elif borcluluk < 0.50:
        finansal_ayarlama += 4
    elif borcluluk > 0.80:
        finansal_ayarlama -= 10
    elif borcluluk > 0.60:
        finansal_ayarlama -= 5

    # Kâr büyümesi
    if kar_buyumesi > 0.50:
        finansal_ayarlama += 7
    elif kar_buyumesi > 0.20:
        finansal_ayarlama += 4
    elif kar_buyumesi > 0:
        finansal_ayarlama += 1
    elif kar_buyumesi < -0.30:
        finansal_ayarlama -= 8
    elif kar_buyumesi < 0:
        finansal_ayarlama -= 3

    son_skor = max(0, min(100, sektor_skor + finansal_ayarlama))
    aciklama = (
        f"Sektör={sektor.value} (skor={sektor_skor}), "
        f"Borçluluk=%{borcluluk*100:.0f}, Kâr büyüme=%{kar_buyumesi*100:.0f} "
        f"-> finansal ayar={finansal_ayarlama:+d} -> son={son_skor}"
    )
    return son_skor, aciklama


def _normalize_piyasa(duyarlilik: PiyasaDuyarliligi) -> Tuple[float, str]:
    """BIST 100 trend skoru."""
    skorlar = {
        PiyasaDuyarliligi.GUCJU_BOGA: 100,
        PiyasaDuyarliligi.BOGA: 82,
        PiyasaDuyarliligi.YATAY: 60,
        PiyasaDuyarliligi.AYI: 38,
        PiyasaDuyarliligi.GUCJU_AYI: 20,
    }
    skor = skorlar.get(duyarlilik, 50)
    aciklama = f"Piyasa={duyarlilik.value} -> skor={skor}"
    return skor, aciklama


def _lot_maliyet_ayarlamasi(lot_maliyet: Optional[float], mevcut_skor: float) -> float:
    """
    Lot başına düşen maliyet ayarlaması.
    Çok düşük maliyet = bireysel yatırımcı zarara düşmeden tutar = uzun tavan.
    """
    if lot_maliyet is None:
        return 0
    if lot_maliyet <= 500:
        return 3
    elif lot_maliyet <= 1500:
        return 2
    elif lot_maliyet <= 3000:
        return 0
    elif lot_maliyet <= 8000:
        return -2
    else:
        return -4


def _talep_konsantrasyon_ayarlamasi(konsantrasyon: Optional[float]) -> float:
    """Talep konsantrasyonu ayarlaması."""
    if konsantrasyon is None:
        return 0
    if konsantrasyon >= 90:
        return 3
    elif konsantrasyon >= 70:
        return 1.5
    elif konsantrasyon >= 50:
        return 0
    else:
        return -2


# ──────────────────────────────────────────────────────────────────────────────
# ANA TAHMİN FONKSİYONU
# ──────────────────────────────────────────────────────────────────────────────

class HalkaArzPredictor:
    """
    Halka Arz Tavan Serisi Tahmin Motoru.
    
    Kullanım:
        predictor = HalkaArzPredictor()
        tahmin = predictor.tahmin_yap(girdiler)
        print(tahmin.ozet())
    """

    def __init__(self, agirliklar: Optional[Dict[str, float]] = None):
        """
        Args:
            agirliklar: Özel ağırlık sözlüğü (varsayılan AGIRLIKLAR)
        """
        self.agirliklar = agirliklar or AGIRLIKLAR.copy()
        self._toplam_agirlik_kontrol()

    def _toplam_agirlik_kontrol(self):
        toplam = sum(self.agirliklar.values())
        if abs(toplam - 1.0) > 0.001:
            raise ValueError(f"Ağırlıklar toplamı 1.0 olmalı, mevcut={toplam:.3f}")

    def tahmin_yap(self, girdi: HalkaArzGirdileri) -> TavanTahmini:
        """
        Ham girdilerden Tavan Gücü Skoru (0-100) hesaplar.
        
        Returns:
            TavanTahmini: Detaylı skorlar ve kategori tahmini
        """
        detaylar: List[ParametreSkoru] = []

        # 1. Dağıtım Yöntemi (%25)
        dagitim_skor, dagitim_aciklama = _normalize_dagitim(girdi.dagitim_yontemi)
        detaylar.append(ParametreSkoru(
            parametre_adi="Dağıtım Yöntemi",
            ham_deger=girdi.dagitim_yontemi.value,
            normalize_skor=dagitim_skor,
            agirlik=self.agirliklar["dagitim_yontemi"],
            agirlikli_skor=dagitim_skor * self.agirliklar["dagitim_yontemi"],
            aciklama=dagitim_aciklama,
        ))

        # 2. Halka Arz Büyüklüğü (%20)
        boyut_skor, boyut_aciklama = _normalize_boyut(girdi.halka_arz_boyutu_tl)
        detaylar.append(ParametreSkoru(
            parametre_adi="Halka Arz Büyüklüğü",
            ham_deger=girdi.halka_arz_boyutu_tl,
            normalize_skor=boyut_skor,
            agirlik=self.agirliklar["halka_arz_boyutu"],
            agirlikli_skor=boyut_skor * self.agirliklar["halka_arz_boyutu"],
            aciklama=boyut_aciklama,
        ))

        # 3. Katılımcı Sayısı (%15)
        katilimci_skor, katilimci_aciklama = _normalize_katilimci(girdi.katilimci_beklentisi)
        detaylar.append(ParametreSkoru(
            parametre_adi="Katılımcı Sayısı",
            ham_deger=girdi.katilimci_beklentisi,
            normalize_skor=katilimci_skor,
            agirlik=self.agirliklar["katilimci_sayisi"],
            agirlikli_skor=katilimci_skor * self.agirliklar["katilimci_sayisi"],
            aciklama=katilimci_aciklama,
        ))

        # 4. Kurumsal Yatırımcı Oranı (%15)
        kurumsal_skor, kurumsal_aciklama = _normalize_kurumsal(
            girdi.kurumsal_oran, girdi.kurumsal_taahhut
        )
        detaylar.append(ParametreSkoru(
            parametre_adi="Kurumsal Yatırımcı Oranı",
            ham_deger=girdi.kurumsal_oran,
            normalize_skor=kurumsal_skor,
            agirlik=self.agirliklar["kurumsal_yatirimci"],
            agirlikli_skor=kurumsal_skor * self.agirliklar["kurumsal_yatirimci"],
            aciklama=kurumsal_aciklama,
        ))

        # 5. Sektör ve Finansallar (%15)
        sektor_skor, sektor_aciklama = _normalize_sektor_finansal(
            girdi.sektor, girdi.borcluluk_orani, girdi.net_kar_buyumesi
        )
        detaylar.append(ParametreSkoru(
            parametre_adi="Sektör ve Finansallar",
            ham_deger=girdi.sektor.value,
            normalize_skor=sektor_skor,
            agirlik=self.agirliklar["sektor_finansallar"],
            agirlikli_skor=sektor_skor * self.agirliklar["sektor_finansallar"],
            aciklama=sektor_aciklama,
        ))

        # 6. Piyasa Duyarlılığı (%10)
        piyasa_skor, piyasa_aciklama = _normalize_piyasa(girdi.piyasa_duyarliligi)
        detaylar.append(ParametreSkoru(
            parametre_adi="Piyasa Duyarlılığı",
            ham_deger=girdi.piyasa_duyarliligi.value,
            normalize_skor=piyasa_skor,
            agirlik=self.agirliklar["piyasa_duyarliligi"],
            agirlikli_skor=piyasa_skor * self.agirliklar["piyasa_duyarliligi"],
            aciklama=piyasa_aciklama,
        ))

        # ── Toplam skor hesaplama ──
        toplam = sum(d.agirlikli_skor for d in detaylar)

        # Opsiyonel ayarlamalar
        lot_ayar = _lot_maliyet_ayarlamasi(girdi.lot_basi_dusen_maliyet, toplam)
        talep_ayar = _talep_konsantrasyon_ayarlamasi(girdi.talep_konsantrasyonu)
        toplam_ayarli = max(0, min(100, toplam + lot_ayar + talep_ayar))

        # Kategori belirleme
        kategori, tahmin_aralik, tavsiye = self._kategori_belirle(toplam_ayarli)

        # Güven oranı (deterministik modelde, skorun merkeze uzaklığına göre)
        guven = self._guven_hesapla(toplam_ayarli)

        # Açıklama oluştur
        aciklama = self._aciklama_olustur(girdi, toplam_ayarli, kategori)

        return TavanTahmini(
            sirket_adi=girdi.sirket_adi,
            toplam_skor=round(toplam_ayarli, 2),
            kategori=kategori,
            tahmin_gun_araligi=tahmin_aralik,
            detayli_skorlar=detaylar,
            guven_orani=round(guven, 1),
            aciklama=aciklama,
            tavsiye=tavsiye,
        )

    def _kategori_belirle(self, skor: float) -> Tuple[TavanKategorisi, str, str]:
        """Skora göre kategori, gün aralığı ve tavsiye döndürür."""
        if skor >= 80:
            return (
                TavanKategorisi.GUCJU,
                "7-10+ Gün Tavan",
                "GÜÇLÜ TAVAN SERİSİ bekleniyor. İlk 3 gün satış yapmayın, "
                "tavan serisi sonuna kadar bekleyin."
            )
        elif skor >= 60:
            return (
                TavanKategorisi.ORTA,
                "4-6 Gün Tavan",
                "ORTA TAVAN SERİSİ bekleniyor. 3-4. günlerde kademeli satış "
                "stratejisi uygulayabilirsiniz."
            )
        elif skor >= 40:
            return (
                TavanKategorisi.ZAYIF,
                "2-3 Gün Tavan",
                "ZAYIF TAVAN SERİSİ bekleniyor. 2. gün tavan kırılabilir, "
                "satış için hazırlıklı olun."
            )
        else:
            return (
                TavanKategorisi.RISKI,
                "1-2 Gün veya Erken Bozma",
                "RİSKLİ ARZ. İlk gün veya 2. gün tavan kırılabilir. "
                "Satış stratejisi kritik önem taşıyor."
            )

    def _guven_hesapla(self, skor: float) -> float:
        """
        Tahmin güven oranı:
        - Uç değerler (0-20 ve 80-100) daha yüksek güven
        - Merkezde (40-60) daha düşük güven
        """
        if skor >= 80:
            return 70 + (skor - 80) * 1.5      # 70-100
        elif skor >= 60:
            return 55 + (skor - 60) * 0.75     # 55-70
        elif skor >= 40:
            return 45 + (skor - 40) * 0.5      # 45-55
        else:
            return 25 + skor * 0.5              # 25-45

    def _aciklama_olustur(self, girdi: HalkaArzGirdileri, skor: float, kategori: TavanKategorisi) -> str:
        """Detaylı analiz metni oluşturur."""
        paragraflar = []
        
        paragraflar.append(
            f"'{girdi.sirket_adi}' için Tavan Gücü Skoru: {skor:.1f}/100"
        )
        
        # Ana etkenler
        etkenler = []
        if girdi.dagitim_yontemi in (DagitimYontemi.BIREYSELE_ESIT, DagitimYontemi.TAMAMI_ESIT):
            etkenler.append("eşit dağıtım")
        if girdi.halka_arz_boyutu_tl <= 100_000_000:
            etkenler.append("küçük tahta")
        if girdi.katilimci_beklentisi >= 700:
            etkenler.append("yüksek katılımcı")
        if girdi.kurumsal_taahhut:
            etkenler.append("kurumsal taahhüt")
        if girdi.sektor in (Sektorm.TEKNOLOJI, Sektorm.ENERJI, Sektorm.SAVUNMA):
            etkenler.append(f"popüler sektör ({girdi.sektor.value})")
        
        if etkenler:
            paragraflar.append("Güçlü tavan serisini destekleyen etkenler: " + ", ".join(etkenler) + ".")
        
        # Risk etkenleri
        riskler = []
        if girdi.dagitim_yontemi == DagitimYontemi.ORANSAL:
            riskler.append("oransal dağıtım")
        if girdi.halka_arz_boyutu_tl >= 500_000_000:
            riskler.append("büyük tahta")
        if girdi.katilimci_beklentisi < 150:
            riskler.append("düşük katılımcı")
        if girdi.borcluluk_orani > 0.70:
            riskler.append("yüksek borçluluk")
        if girdi.net_kar_buyumesi < 0:
            riskler.append("negatif kâr büyümesi")
        
        if riskler:
            paragraflar.append("Dikkat edilmesi gereken risk etkenleri: " + ", ".join(riskler) + ".")
        
        return " ".join(paragraflar)

    def toplu_tahmin(self, girdiler: List[HalkaArzGirdileri]) -> List[TavanTahmini]:
        """Birden fazla halka arz için toplu tahmin."""
        return [self.tahmin_yap(g) for g in girdiler]
