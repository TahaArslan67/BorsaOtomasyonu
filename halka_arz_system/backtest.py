"""
Halka Arz Tahmin Motoru - Backtest Modülü
==========================================
Geçmiş halka arz verileriyle model doğruluğunu test eder.

Metrikler:
- Kategori Doğruluk Oranı: Tahmin edilen kategorinin gerçekleşenle
  ne kadar örtüştüğü
- MAE (Mean Absolute Error): Gün tahminindeki ortalama mutlak hata
- R² Skoru: Regresyon uyumu
"""
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from statistics import mean, stdev

from .data_model import GecmisHalkaArz, TavanKategorisi
from .predictor import HalkaArzPredictor


@dataclass
class BacktestSonucu:
    """Tek bir halka arzın backtest sonucu."""
    sirket_adi: str
    tahmin_skor: float
    tahmin_kategori: TavanKategorisi
    tahmin_gun_araligi: str
    gercek_tavan_gunu: int
    kategori_dogru: bool
    gun_hatasi: int          # |tahmin_orta_deger - gercek|


@dataclass
class BacktestRaporu:
    """Toplu backtest raporu."""
    toplam_ornek: int
    kategori_dogru_sayisi: int
    kategori_dogruluk_orani: float
    ortalama_gun_hatasi: float
    mae: float               # Mean Absolute Error
    rmse: float              # Root Mean Squared Error
    kategori_karisiklik_matrisi: Dict[str, Dict[str, int]]
    detayli_sonuclar: List[BacktestSonucu]
    
    def ozet(self) -> str:
        lines = [
            "=" * 60,
            "HALKA ARZ TAHMİN MOTORU - BACKTEST RAPORU",
            "=" * 60,
            f"Toplam Örnek Sayısı     : {self.toplam_ornek}",
            f"Kategori Doğruluk Oranı : %{self.kategori_dogruluk_orani:.1f}",
            f"Ortalama Gün Hatası     : {self.ortalama_gun_hatasi:.1f} gün",
            f"MAE (Mutlak Ortalama)   : {self.mae:.1f} gün",
            f"RMSE (Karesel Hata)     : {self.rmse:.1f} gün",
            "=" * 60,
        ]
        return "\n".join(lines)


def _kategori_gun_ortasi(kategori: TavanKategorisi) -> int:
    """Kategori için ortalama gün değeri (hata hesabı için)."""
    orta_degerler = {
        TavanKategorisi.GUCJU: 9,    # 7-10+ -> 9
        TavanKategorisi.ORTA: 5,     # 4-6   -> 5
        TavanKategorisi.ZAYIF: 3,    # 2-3   -> 3
        TavanKategorisi.RISKI: 1,    # 1-2   -> 1
    }
    return orta_degerler.get(kategori, 5)


def _gercek_kategori_belirle(gercek_gun: int) -> TavanKategorisi:
    """Gerçekleşen tavan gününe göre kategori belirler."""
    if gercek_gun >= 7:
        return TavanKategorisi.GUCJU
    elif gercek_gun >= 4:
        return TavanKategorisi.ORTA
    elif gercek_gun >= 2:
        return TavanKategorisi.ZAYIF
    else:
        return TavanKategorisi.RISKI


def backtest_yap(
    gecmis_veriler: List[GecmisHalkaArz],
    predictor: HalkaArzPredictor = None,
) -> BacktestRaporu:
    """
    Geçmiş halka arz verileriyle modeli test eder.
    
    Args:
        gecmis_veriler: Geçmiş halka arz listesi
        predictor: Kullanılacak predictor (varsayılan: HalkaArzPredictor())
    
    Returns:
        BacktestRaporu: Detaylı backtest raporu
    """
    if predictor is None:
        predictor = HalkaArzPredictor()
    
    sonuclar: List[BacktestSonucu] = []
    kategori_matrisi: Dict[str, Dict[str, int]] = {
        "guclu_tavan": {"guclu_tavan": 0, "orta_tavan": 0, "zayif_tavan": 0, "riskli": 0},
        "orta_tavan": {"guclu_tavan": 0, "orta_tavan": 0, "zayif_tavan": 0, "riskli": 0},
        "zayif_tavan": {"guclu_tavan": 0, "orta_tavan": 0, "zayif_tavan": 0, "riskli": 0},
        "riskli": {"guclu_tavan": 0, "orta_tavan": 0, "zayif_tavan": 0, "riskli": 0},
    }
    
    for veri in gecmis_veriler:
        # Tahmin üret
        tahmin = predictor.tahmin_yap(veri.girdiler)
        gercek_kategori = _gercek_kategori_belirle(veri.gerceklesen_tavan_gunu)
        
        # Kategori doğruluğu
        kategori_dogru = (tahmin.kategori == gercek_kategori)
        
        # Gün hatası (kategori orta değeri ile karşılaştır)
        tahmin_gun = _kategori_gun_ortasi(tahmin.kategori)
        gun_hatasi = abs(tahmin_gun - veri.gerceklesen_tavan_gunu)
        
        sonuclar.append(BacktestSonucu(
            sirket_adi=veri.sirket_adi,
            tahmin_skor=tahmin.toplam_skor,
            tahmin_kategori=tahmin.kategori,
            tahmin_gun_araligi=tahmin.tahmin_gun_araligi,
            gercek_tavan_gunu=veri.gerceklesen_tavan_gunu,
            kategori_dogru=kategori_dogru,
            gun_hatasi=gun_hatasi,
        ))
        
        # Karışıklık matrisi
        gercek_key = gercek_kategori.value
        tahmin_key = tahmin.kategori.value
        kategori_matrisi[gercek_key][tahmin_key] += 1
    
    # İstatistikler
    toplam = len(sonuclar)
    dogru_sayi = sum(1 for s in sonuclar if s.kategori_dogru)
    dogruluk = (dogru_sayi / toplam * 100) if toplam > 0 else 0
    
    hatalar = [s.gun_hatasi for s in sonuclar]
    ort_hata = mean(hatalar) if hatalar else 0
    mae = mean(hatalar) if hatalar else 0
    rmse = (sum(h**2 for h in hatalar) / len(hatalar)) ** 0.5 if hatalar else 0
    
    return BacktestRaporu(
        toplam_ornek=toplam,
        kategori_dogru_sayisi=dogru_sayi,
        kategori_dogruluk_orani=dogruluk,
        ortalama_gun_hatasi=ort_hata,
        mae=mae,
        rmse=rmse,
        kategori_karisiklik_matrisi=kategori_matrisi,
        detayli_sonuclar=sonuclar,
    )


def agirlik_optimizasyonu(
    gecmis_veriler: List[GecmisHalkaArz],
    iterasyon: int = 100,
) -> Tuple[Dict[str, float], BacktestRaporu]:
    """
    Grid search ile en iyi ağırlık kombinasyonunu bulur.
    
    NOT: Bu basit bir grid search'tür. Daha gelişmiş optimizasyon için
    scipy.optimize veya optuna kullanılabilir.
    """
    import random
    
    en_iyi_agirliklar = None
    en_iyi_mae = float("inf")
    en_iyi_rapor = None
    
    # Rastgele ağırlık kombinasyonları dene
    for _ in range(iterasyon):
        # Rastgele ağırlıklar (toplam 1.0 olacak şekilde normalize)
        r = [random.random() for _ in range(6)]
        toplam = sum(r)
        agirliklar = {
            "dagitim_yontemi": r[0] / toplam,
            "halka_arz_boyutu": r[1] / toplam,
            "katilimci_sayisi": r[2] / toplam,
            "kurumsal_yatirimci": r[3] / toplam,
            "sektor_finansallar": r[4] / toplam,
            "piyasa_duyarliligi": r[5] / toplam,
        }
        
        try:
            predictor = HalkaArzPredictor(agirliklar=agirliklar)
            rapor = backtest_yap(gecmis_veriler, predictor)
            
            if rapor.mae < en_iyi_mae:
                en_iyi_mae = rapor.mae
                en_iyi_agirliklar = agirliklar
                en_iyi_rapor = rapor
        except ValueError:
            continue
    
    return en_iyi_agirliklar, en_iyi_rapor


def raporu_yazdir(rapor: BacktestRaporu, detayli: bool = True):
    """Backtest raporunu konsola yazdırır."""
    print(rapor.ozet())
    
    if detayli:
        print("\n--- DETAYLI SONUÇLAR ---")
        print(f"{'Şirket':<20} {'Skor':<8} {'Tahmin':<12} {'Gerçek':<8} {'Hata':<6} {'Durum':<8}")
        print("-" * 70)
        for s in rapor.detayli_sonuclar:
            durum = "✓" if s.kategori_dogru else "✗"
            print(f"{s.sirket_adi:<20} {s.tahmin_skor:<8.1f} "
                  f"{s.tahmin_gun_araligi:<12} {s.gercek_tavan_gunu:<8} "
                  f"{s.gun_hatasi:<6} {durum:<8}")
        
        print("\n--- KARMAŞIKLIK MATRİSİ ---")
        print(f"{'Gerçek \\ Tahmin':<18} {'Güçlü':<8} {'Orta':<8} {'Zayıf':<8} {'Riskli':<8}")
        print("-" * 50)
        for gercek, tahminler in rapor.kategori_karisiklik_matrisi.items():
            print(f"{gercek:<18} {tahminler['guclu_tavan']:<8} "
                  f"{tahminler['orta_tavan']:<8} {tahminler['zayif_tavan']:<8} "
                  f"{tahminler['riskli']:<8}")


def raporu_json_kaydet(rapor: BacktestRaporu, dosya_yolu: str = "backtest_raporu.json"):
    """Backtest raporunu JSON olarak kaydeder."""
    data = {
        "ozet": {
            "toplam_ornek": rapor.toplam_ornek,
            "kategori_dogruluk_orani": rapor.kategori_dogruluk_orani,
            "ortalama_gun_hatasi": rapor.ortalama_gun_hatasi,
            "mae": rapor.mae,
            "rmse": rapor.rmse,
        },
        "karmaşıklık_matrisi": rapor.kategori_karisiklik_matrisi,
        "detaylar": [
            {
                "sirket": s.sirket_adi,
                "tahmin_skor": s.tahmin_skor,
                "tahmin_kategori": s.tahmin_kategori.value,
                "gercek_gun": s.gercek_tavan_gunu,
                "dogru": s.kategori_dogru,
                "gun_hatasi": s.gun_hatasi,
            }
            for s in rapor.detayli_sonuclar
        ],
    }
    with open(dosya_yolu, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nRapor kaydedildi: {dosya_yolu}")
