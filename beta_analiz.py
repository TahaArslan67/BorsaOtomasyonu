"""
Beta Enerji (BETAE) Halka Arz Analizi
"""
import sys
sys.path.insert(0, 'd:/otonomBorsa')

from halka_arz_system.predictor import HalkaArzPredictor
from halka_arz_system.data_model import HalkaArzGirdileri, DagitimYontemi, Sektorm, PiyasaDuyarliligi

# Beta Enerji verileri
# Fiyat: 40 TL
# Pay: 60.750.000 lot
# Büyüklük: 60.750.000 * 40 = 2.430.000.000 TL
# Dağıtım: Eşit Dağıtım (Bireysele eşit)
# Sektör: Enerji + Teknoloji
# Halka açıklık: %5 (çok düşük)
# Satmama taahhüdü: 1 yıl
# Piyasa: Düşüş trendinde (ayı piyasası)

girdi = HalkaArzGirdileri(
    sirket_adi="Beta Enerji ve Teknoloji A.S.",
    dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,  # Eşit dağıtım
    halka_arz_boyutu_tl=2_430_000_000,  # 60.75M lot * 40 TL
    katilimci_beklentisi=2500,  # 2.5M katılımcı (eşit dağıtım + popüler)
    kurumsal_oran=0.20,  # %20 kurumsal (tahmini)
    kurumsal_taahhut=True,  # 1 yıl taahhüt var
    sektor=Sektorm.ENERJI,  # Enerji sektörü
    borcluluk_orani=0.50,  # Tahmini
    net_kar_buyumesi=0.30,  # %30 büyüme (tahmini)
    piyasa_duyarliligi=PiyasaDuyarliligi.AYI,  # Düşüş trendi
    lot_basi_dusen_maliyet=40.0,  # 40 TL / lot
)

predictor = HalkaArzPredictor()
tahmin = predictor.tahmin_yap(girdi)

print("=" * 70)
print("BETA ENERJI (BETAE) HALKA ARZ ANALIZI")
print("=" * 70)
print(f"\n{tahmin.ozet()}")
print(f"\nSkor: {tahmin.toplam_skor:.1f}/100")
print(f"Kategori: {tahmin.kategori.value}")
print(f"Tahmin: {tahmin.tahmin_gun_araligi}")
print(f"Guven: %{tahmin.guven_orani:.0f}")
print(f"\nTavsiye: {tahmin.tavsiye}")

print("\n" + "=" * 70)
print("DETAYLI SKORLAR")
print("=" * 70)
for skor in tahmin.detayli_skorlar:
    print(f"\n{skor.parametre_adi}:")
    print(f"  Skor: {skor.normalize_skor:.0f}/100 (Agirlikli: {skor.agirlikli_skor:.2f})")
    print(f"  {skor.aciklama}")

# Islem tarihi
print("\n" + "=" * 70)
print("ONEMLI BILGILER")
print("=" * 70)
print("Halka Arz Tarihi: 23-24-25 Haziran 2026")
print("Fiyat: 40.00 TL")
print("Pay: 60.750.000 Lot")
print("Buyukluk: ~2.4 Milyar TL")
print("Pazar: Yildiz Pazar (Tavan limiti: %20)")
print("Dagitim: Esit dagitim")
print("Satmama Taahhudu: 1 yil")
print("Halka Aciklik: %5 (cok dusuk - spekulasyon icin uygun)")

# Tahmini lot dagilimi
print("\n" + "=" * 70)
print("TAHMINI LOT DAGILIMI")
print("=" * 70)
katilim_senaryolari = [
    (350, 86, "350K katilimci"),
    (500, 61, "500K katilimci"),
    (700, 43, "700K katilimci"),
    (1100, 28, "1.1M katilimci"),
    (1600, 19, "1.6M katilimci"),
    (2200, 14, "2.2M katilimci"),
]

print("Senaryo         | Lot | Maliyet | Potansiyel (10 tavan)")
print("-" * 60)
for katilim, lot, aciklama in katilim_senaryolari:
    maliyet = lot * 40
    tavan_fiyat = 40 * (1.20 ** 10)  # 10 gun tavan
    potansiyel = lot * tavan_fiyat
    kar = potansiyel - maliyet
    print(f"{aciklama:<15} | {lot:>3} | {maliyet:>6,} TL | {kar:>10,.0f} TL kâr")
