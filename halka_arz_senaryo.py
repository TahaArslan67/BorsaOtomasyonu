"""
5 Halka Arz - En Kötü Senaryo Analizi
"""
import sys
sys.path.insert(0, 'd:/otonomBorsa')

from halka_arz_system.predictor import HalkaArzPredictor
from halka_arz_system.data_model import HalkaArzGirdileri, DagitimYontemi, Sektorm, PiyasaDuyarliligi

predictor = HalkaArzPredictor()

# 5 Halka arz - sektore gore varsayimsal parametreler
halka_arzlar = [
    {
        "isim": "Orzaks İlaç (ORZAX)",
        "girdi": HalkaArzGirdileri(
            sirket_adi="Orzaks İlaç",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=150_000_000,
            katilimci_beklentisi=1200,
            kurumsal_oran=0.25,
            kurumsal_taahhut=True,
            sektor=Sektorm.SAGLIK,
            borcluluk_orani=0.35,
            net_kar_buyumesi=0.20,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        )
    },
    {
        "isim": "Intercity Turizm (ICITY)",
        "girdi": HalkaArzGirdileri(
            sirket_adi="Intercity Turizm",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=200_000_000,
            katilimci_beklentisi=800,
            kurumsal_oran=0.30,
            kurumsal_taahhut=False,
            sektor=Sektorm.HIZMET,
            borcluluk_orani=0.50,
            net_kar_buyumesi=0.10,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        )
    },
    {
        "isim": "Soho Giyim (SOHO)",
        "girdi": HalkaArzGirdileri(
            sirket_adi="Soho Giyim ve Enerji",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=180_000_000,
            katilimci_beklentisi=1000,
            kurumsal_oran=0.20,
            kurumsal_taahhut=True,
            sektor=Sektorm.URETIM,  # Karma - enerji+giyim
            borcluluk_orani=0.40,
            net_kar_buyumesi=0.15,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        )
    },
    {
        "isim": "İsvea Seramik (ISVEA)",
        "girdi": HalkaArzGirdileri(
            sirket_adi="İsvea Seramik",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=120_000_000,
            katilimci_beklentisi=600,
            kurumsal_oran=0.30,
            kurumsal_taahhut=False,
            sektor=Sektorm.INSAAT,
            borcluluk_orani=0.45,
            net_kar_buyumesi=0.05,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        )
    },
    {
        "isim": "Golda Gıda (GOLDA)",
        "girdi": HalkaArzGirdileri(
            sirket_adi="Golda Gıda",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=100_000_000,
            katilimci_beklentisi=900,
            kurumsal_oran=0.25,
            kurumsal_taahhut=True,
            sektor=Sektorm.GIDA,
            borcluluk_orani=0.35,
            net_kar_buyumesi=0.12,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        )
    },
]

print("=" * 80)
print("5 HALKA ARZ ANALIZI (Varsayimsal Parametreler)")
print("=" * 80)

# Her biri icin tahmin
toplam_skor = 0
toplam_tavan = 0

for i, arz in enumerate(halka_arzlar, 1):
    tahmin = predictor.tahmin_yap(arz["girdi"])
    print(f"\n{i}. {arz['isim']}")
    print(f"   Skor: {tahmin.toplam_skor:.1f}/100 | {tahmin.tahmin_gun_araligi}")
    print(f"   Guven: %{tahmin.guven_orani:.0f} | Kategori: {tahmin.kategori.value}")
    
    # Tahmini gun sayisi (araligin ortasi)
    if tahmin.kategori.value == "guclu_tavan":
        tavan_gun = 8
    elif tahmin.kategori.value == "orta_tavan":
        tavan_gun = 5
    elif tahmin.kategori.value == "zayif_tavan":
        tavan_gun = 2
    else:
        tavan_gun = 1
    
    toplam_tavan += tavan_gun
    topalam_skor = 0  # dummy

# En kotu senaryo hesaplama
print("\n" + "=" * 80)
print("EN KOTU SENARYO ANALIZI")
print("=" * 80)

# Her halka arzda 3000 TL varsayalim
yatirim_basi = 3000
toplam_yatirim = yatirim_basi * 5

print(f"\nVarsayim: Her halka arzda {yatirim_basi:,} TL yatirim")
print(f"Toplam yatirim: {toplam_yatirim:,} TL")

# En kotu senaryo: Her biri sadece 1 gun tavan veya erken bozma
# Yildiz pazar %20 tavan, Ana pazar %10

senaryolar = [
    ("Cok Kotu", 0, 0, "Hic tavan yapmaz, basa bas"),  # 0 gun
    ("Kotu", 1, 0.20, "1 gun %20 tavan (yildiz)"),
    ("Kotu-Ana", 1, 0.10, "1 gun %10 tavan (ana pazar)"),
    ("Zayif", 2, 0.20, "2 gun %20 tavan"),
    ("Orta-Kotu", 2, 0.10, "2 gun %10 tavan"),
]

print("\nSenaryo          | Gun | Tavan | Toplam Satis | KAR      | Sure")
print("-" * 80)

for adi, gun, tavan, aciklama in senaryolar:
    # Her halka arzda ayni yatirim, ayni getiri
    # Basit hesaplama: gun sayisi kadar tavan uygula
    lot_maliyet = 40  # ortalama varsayimsal lot fiyati
    lot = int(yatirim_basi / lot_maliyet)
    
    # gun sayisi tavan
    fiyat = lot_maliyet
    for _ in range(gun):
        fiyat *= (1 + tavan)
    
    satis = lot * fiyat
    kar = satis - yatirim_basi
    toplam_kar = kar * 5
    
    print(f"{adi:<16} | {gun:>3} | %{tavan*100:>4.0f}  | {satis*5:>10,.0f} TL | {toplam_kar:>8,.0f} TL | ~{gun} gun")

# En gercekci kotu senaryo
print("\n" + "=" * 80)
print("GERCEKCI EN KOTU SENARYO")
print("=" * 80)

# Bazilari kotu gider, bazilari orta
gercekci = [
    ("Orzaks", 3000, 2, 0.20),  # Saglik - orta
    ("Intercity", 3000, 1, 0.10),  # Turizm - kotu
    ("Soho", 3000, 3, 0.20),  # Karma - orta
    ("Isvea", 3000, 1, 0.10),  # Insaat - kotu
    ("Golda", 3000, 2, 0.20),  # Gida - orta
]

print("\nHalka Arz       | Yatirim | Gun | Tavan | Satis    | KAR")
print("-" * 80)

toplam_yatirim = 0
toplam_satis = 0

for isim, yatirim, gun, tavan in gercekci:
    fiyat = 40
    lot = int(yatirim / fiyat)
    
    for _ in range(gun):
        fiyat *= (1 + tavan)
    
    satis = lot * fiyat
    kar = satis - yatirim
    
    print(f"{isim:<15} | {yatirim:>6,.0f} | {gun:>3} | %{tavan*100:>3.0f}  | {satis:>8,.0f} | {kar:>6,.0f}")
    
    toplam_yatirim += yatirim
    toplam_satis += satis

print("-" * 80)
print(f"{'TOPLAM':<15} | {toplam_yatirim:>6,.0f} |     |       | {toplam_satis:>8,.0f} | {toplam_satis-toplam_yatirim:>6,.0f}")

kar_orani = (toplam_satis - toplam_yatirim) / toplam_yatirim * 100
print(f"\nToplam KAR: {toplam_satis-toplam_yatirim:,.0f} TL (%{kar_orani:.1f})")
print(f"Sure: Her biri ayri ayri, toplam ~1-2 ay")

# Eger hepsi 1 gun tavan yaparsa
print("\n" + "=" * 80)
print("HEPSI 1 GUN TAVAN (En Kotu Genel Senaryo)")
print("=" * 80)
hepsi_1gun = toplam_yatirim * 1.10  # %10 veya %20 karisik
print(f"Toplam Satis: ~{hepsi_1gun:,.0f} TL")
print(f"KAR: ~{hepsi_1gun-toplam_yatirim:,.0f} TL (%{((hepsi_1gun/toplam_yatirim)-1)*100:.1f})")
