"""
Halka Arz Tahmin Sistemi - Veri Modelleri
=========================================
Parametrik girdiler, skorlar ve tahmin çıktıları için tip güvenli dataclass'lar.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class DagitimYontemi(Enum):
    """Halka arzda kullanılan dağıtım yöntemleri."""
    BIREYSELE_ESIT = "bireysele_esit"       # Bireysele eşit dağıtım
    TAMAMI_ESIT = "tamami_esit"             # Tüm katılımcılara eşit
    ORANSAL = "oransal"                     # Talep oranına göre
    HALKA_ARZ_FONU = "halka_arz_fonu"       # Halka arz fonu aracılığıyla
    KARMA = "karma"                         # Karma yöntem


class Sektorm(Enum):
    """Şirket sektörleri (popülerlik sırasına göre)."""
    TEKNOLOJI = "teknoloji"
    ENERJI = "enerji"
    SAVUNMA = "savunma"
    SAGLIK = "saglik"
    FINANS = "finans"
    GIDA = "gida"
    URETIM = "uretim"
    INSAAT = "insaat"
    HIZMET = "hizmet"
    DIGER = "diger"


class PiyasaDuyarliligi(Enum):
    """BIST 100 genel trend durumu."""
    GUCJU_BOGA = "guclu_boga"       # > %15 yukarı trend (son 3 ay)
    BOGA = "boga"                   # > %5 yukarı trend
    YATAY = "yatay"                 # +/- %5 aralığında
    AYI = "ayi"                     # > %5 aşağı trend
    GUCJU_AYI = "guclu_ayi"         # > %15 aşağı trend


class TavanKategorisi(Enum):
    """Tavan serisi tahmin kategorileri."""
    GUCJU = "guclu_tavan"           # 7-10+ gün
    ORTA = "orta_tavan"             # 4-6 gün
    ZAYIF = "zayif_tavan"           # 2-3 gün
    RISKI = "riskli"                # 1-2 gün veya erken bozma


@dataclass
class HalkaArzGirdileri:
    """
    Tahmin motoruna verilen ham girdiler.
    
    Attributes:
        sirket_adi: Şirketin ticari unvanı
        dagitim_yontemi: Halka arzda kullanılan dağıtım yöntemi
        halka_arz_boyutu_tl: Halka arzın toplam TL değeri
        katilimci_beklentisi: Tahmini katılımcı sayısı (bin kişi cinsinden)
        kurumsal_oran: Kurumsal yatırımcılara ayrılan pay oranı (0.0 - 1.0)
        kurumsal_taahhut: Kurumsal yatırımcıların satmama taahhüdü var mı
        sektor: Şirketin faaliyet gösterdiği sektör
        borcluluk_orani: Son bilanço dönemi borçluluk oranı (0.0 - 1.0+)
        net_kar_buyumesi: Son yıllık net kâr büyüme oranı (negatif olabilir)
        piyasa_duyarliligi: Halka arz dönemindeki BIST 100 trendi
        talep_konsantrasyonu: Yüksek talep gören fiyat aralığı yoğunluğu (0-100)
        lot_basi_dusen_maliyet: Küçük yatırımcıya düşen lot başına maliyet (TL)
    """
    sirket_adi: str
    dagitim_yontemi: DagitimYontemi
    halka_arz_boyutu_tl: float
    katilimci_beklentisi: int          # bin kişi
    kurumsal_oran: float               # 0.0 - 1.0
    kurumsal_taahhut: bool
    sektor: Sektorm
    borcluluk_orani: float             # 0.0 - 1.0+
    net_kar_buyumesi: float            # -1.0 ile +1.0+ arası
    piyasa_duyarliligi: PiyasaDuyarliligi
    
    # Opsiyonel ama etkili parametreler
    talep_konsantrasyonu: Optional[float] = None   # 0-100
    lot_basi_dusen_maliyet: Optional[float] = None # TL
    
    def to_dict(self) -> Dict:
        """Sözlük olarak döndürür."""
        return {
            "sirket_adi": self.sirket_adi,
            "dagitim_yontemi": self.dagitim_yontemi.value,
            "halka_arz_boyutu_tl": self.halka_arz_boyutu_tl,
            "katilimci_beklentisi": self.katilimci_beklentisi,
            "kurumsal_oran": self.kurumsal_oran,
            "kurumsal_taahhut": self.kurumsal_taahhut,
            "sektor": self.sektor.value,
            "borcluluk_orani": self.borcluluk_orani,
            "net_kar_buyumesi": self.net_kar_buyumesi,
            "piyasa_duyarliligi": self.piyasa_duyarliligi.value,
            "talep_konsantrasyonu": self.talep_konsantrasyonu,
            "lot_basi_dusen_maliyet": self.lot_basi_dusen_maliyet,
        }


@dataclass
class ParametreSkoru:
    """Tek bir parametrenin normalize edilmiş skoru ve açıklaması."""
    parametre_adi: str
    ham_deger: float
    normalize_skor: float              # 0 - 100
    agirlik: float                     # 0.0 - 1.0
    agirlikli_skor: float              # normalize_skor * agirlik
    aciklama: str


@dataclass
class TavanTahmini:
    """Tahmin motorunun çıktısı."""
    sirket_adi: str
    toplam_skor: float                 # 0 - 100
    kategori: TavanKategorisi
    tahmin_gun_araligi: str
    detayli_skorlar: List[ParametreSkoru]
    guven_orani: float                 # Backtest geçmişine dayalı güven
    aciklama: str
    tavsiye: str
    
    def ozet(self) -> str:
        """Tek satırlık özet."""
        return (f"{self.sirket_adi}: Skor={self.toplam_skor:.1f}/100 | "
                f"Kategori={self.kategori.value} | "
                f"Tahmin={self.tahmin_gun_araligi} | "
                f"Güven=%{self.guven_orani:.0f}")


@dataclass
class GecmisHalkaArz:
    """Backtest ve model kalibrasyonu için geçmiş halka arz verisi."""
    sirket_adi: str
    halka_arz_tarihi: str              # YYYY-MM-DD
    girdiler: HalkaArzGirdileri
    gerceklesen_tavan_gunu: int
    notlar: Optional[str] = None
