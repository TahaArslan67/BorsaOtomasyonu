#!/usr/bin/env python3
"""
Halka Arz Tavan Serisi Tahmin Sistemi - CLI Arayüz
====================================================
Kullanım:
  python -m halka_arz_system.main --mode predict --sirket "Örnek A.Ş."
  python -m halka_arz_system.main --mode backtest
  python -m halka_arz_system.main --mode demo

Parametreler:
  --sirket           : Şirket adı
  --dagitim          : Dagitim yöntemi (bireysele_esit, tamami_esit, oransal, halka_arz_fonu, karma)
  --boyut            : Halka arz büyüklüğü (TL)
  --katilimci        : Katılımcı beklentisi (bin kişi)
  --kurumsal         : Kurumsal oran (0.0-1.0)
  --taahhut          : Kurumsal taahhüt var mı
  --sektor           : Sektör (teknoloji, enerji, savunma, saglik, finans, gida, uretim, insaat, hizmet, diger)
  --borcluluk        : Borçluluk oranı (0.0-1.0)
  --kar-buyume       : Net kâr büyümesi (-1.0 ile +1.0)
  --piyasa           : Piyasa duyarlılığı (guclu_boga, boga, yatay, ayi, guclu_ayi)
  --lot-maliyet      : Lot başına düşen maliyet (TL, opsiyonel)
  --talep-konsantrasyon : Talep konsantrasyonu (0-100, opsiyonel)
"""
import argparse
import sys
import json
from pathlib import Path

from .data_model import HalkaArzGirdileri, DagitimYontemi, Sektorm, PiyasaDuyarliligi
from .predictor import HalkaArzPredictor
from .historical_data import get_gecmis_veriler
from .backtest import backtest_yap, raporu_yazdir, raporu_json_kaydet


def print_banner():
    print("\n" + "=" * 70)
    print("  HALKA ARZ TAVAN SERİSİ TAHMİN MOTORU v1.0")
    print("  Parametrik Skorlama | Backtest Uyumlu | Borsa İstanbul")
    print("=" * 70)
    print()


def demo_mode():
    """Örnek bir tahmin demo'su çalıştırır."""
    print_banner()
    print("[MOD] DEMO TAHMİN\n")

    predictor = HalkaArzPredictor()

    # Örnek girdiler
    ornekler = [
        HalkaArzGirdileri(
            sirket_adi="Enerji Yatırım A.Ş.",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=85_000_000,
            katilimci_beklentisi=1400,
            kurumsal_oran=0.25,
            kurumsal_taahhut=False,
            sektor=Sektorm.ENERJI,
            borcluluk_orani=0.35,
            net_kar_buyumesi=0.25,
            piyasa_duyarliligi=PiyasaDuyarliligi.BOGA,
            lot_basi_dusen_maliyet=1200,
        ),
        HalkaArzGirdileri(
            sirket_adi="Tekno Girişim S.A.",
            dagitim_yontemi=DagitimYontemi.TAMAMI_ESIT,
            halka_arz_boyutu_tl=45_000_000,
            katilimci_beklentisi=1800,
            kurumsal_oran=0.10,
            kurumsal_taahhut=False,
            sektor=Sektorm.TEKNOLOJI,
            borcluluk_orani=0.15,
            net_kar_buyumesi=0.55,
            piyasa_duyarliligi=PiyasaDuyarliligi.GUCJU_BOGA,
            lot_basi_dusen_maliyet=800,
        ),
        HalkaArzGirdileri(
            sirket_adi="Büyük Holding A.Ş.",
            dagitim_yontemi=DagitimYontemi.ORANSAL,
            halka_arz_boyutu_tl=900_000_000,
            katilimci_beklentisi=300,
            kurumsal_oran=0.55,
            kurumsal_taahhut=False,
            sektor=Sektorm.FINANS,
            borcluluk_orani=0.70,
            net_kar_buyumesi=-0.05,
            piyasa_duyarliligi=PiyasaDuyarliligi.AYI,
            lot_basi_dusen_maliyet=15000,
        ),
    ]

    for girdi in ornekler:
        tahmin = predictor.tahmin_yap(girdi)
        print(tahmin.ozet())
        print(f"  → Tavsiye: {tahmin.tavsiye}")
        print(f"  → Güven: %{tahmin.guven_orani:.0f}")
        print()

        print("  Detaylı Skorlar:")
        for skor in tahmin.detayli_skorlar:
            print(f"    • {skor.parametre_adi:<20} | Skor: {skor.normalize_skor:>6.1f} | "
                  f"Ağırlık: %{skor.agirlik*100:>5.0f} | Katkı: {skor.agirlikli_skor:>6.2f}")
        print("-" * 70)
        print()


def predict_mode(args):
    """Kullanıcı girdileriyle tahmin yapar."""
    print_banner()
    print("[MOD] TAHMİN\n")

    try:
        girdi = HalkaArzGirdileri(
            sirket_adi=args.sirket,
            dagitim_yontemi=DagitimYontemi(args.dagitim),
            halka_arz_boyutu_tl=args.boyut,
            katilimci_beklentisi=args.katilimci,
            kurumsal_oran=args.kurumsal,
            kurumsal_taahhut=args.taahhut,
            sektor=Sektorm(args.sektor),
            borcluluk_orani=args.borcluluk,
            net_kar_buyumesi=args.kar_buyume,
            piyasa_duyarliligi=PiyasaDuyarliligi(args.piyasa),
            lot_basi_dusen_maliyet=args.lot_maliyet,
            talep_konsantrasyonu=args.talep_konsantrasyon,
        )
    except ValueError as e:
        print(f"❌ Geçersiz parametre: {e}")
        print("Geçerli değerler için --help kullanın.")
        sys.exit(1)

    predictor = HalkaArzPredictor()
    tahmin = predictor.tahmin_yap(girdi)

    print(f"\n{'='*60}")
    print(f"  {tahmin.sirket_adi} - TAVAN TAHMİNİ")
    print(f"{'='*60}")
    print(f"\n  📊 Tavan Gücü Skoru: {tahmin.toplam_skor:.1f}/100")
    print(f"  📈 Kategori: {tahmin.kategori.value}")
    print(f"  📅 Tahmin: {tahmin.tahmin_gun_araligi}")
    print(f"  ✅ Güven Oranı: %{tahmin.guven_orani:.0f}")
    print(f"\n  💡 Tavsiye:")
    print(f"     {tahmin.tavsiye}")
    print(f"\n  📝 Analiz:")
    print(f"     {tahmin.aciklama}")

    print(f"\n  Detaylı Skorlar:")
    print(f"  {'Parametre':<22} {'Skor':>8} {'Ağırlık':>8} {'Katkı':>8} {'Açıklama'}")
    print(f"  {'-'*70}")
    for skor in tahmin.detayli_skorlar:
        print(f"  {skor.parametre_adi:<22} {skor.normalize_skor:>7.1f} "
              f"%{skor.agirlik*100:>6.0f} {skor.agirlikli_skor:>7.2f}  {skor.aciklama}")

    # JSON kaydet
    out = {
        "sirket": tahmin.sirket_adi,
        "skor": tahmin.toplam_skor,
        "kategori": tahmin.kategori.value,
        "tahmin_gun": tahmin.tahmin_gun_araligi,
        "guven": tahmin.guven_orani,
        "tavsiye": tahmin.tavsiye,
        "detaylar": [
            {
                "parametre": s.parametre_adi,
                "ham_deger": s.ham_deger,
                "normalize_skor": s.normalize_skor,
                "agirlik": s.agirlik,
                "agirlikli_skor": s.agirlikli_skor,
                "aciklama": s.aciklama,
            }
            for s in tahmin.detayli_skorlar
        ],
    }
    out_path = Path("halka_arz_tahmini.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Tahmin kaydedildi: {out_path.absolute()}")


def backtest_mode():
    """Geçmiş verilerle backtest çalıştırır."""
    print_banner()
    print("[MOD] BACKTEST\n")

    veriler = get_gecmis_veriler()
    print(f"Yüklenen geçmiş veri sayısı: {len(veriler)}\n")

    predictor = HalkaArzPredictor()
    rapor = backtest_yap(veriler, predictor)
    raporu_yazdir(rapor, detayli=True)
    raporu_json_kaydet(rapor, "halka_arz_backtest_raporu.json")


def main():
    parser = argparse.ArgumentParser(
        description="Halka Arz Tavan Serisi Tahmin Motoru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Demo tahminler
  python -m halka_arz_system.main --mode demo

  # Tek tahmin
  python -m halka_arz_system.main --mode predict \\
    --sirket "Örnek A.Ş." --dagitim bireysele_esit --boyut 100000000 \\
    --katilimci 1000 --kurumsal 0.25 --taahhut \\
    --sektor enerji --borcluluk 0.35 --kar-buyume 0.20 --piyasa boga

  # Backtest
  python -m halka_arz_system.main --mode backtest
        """,
    )

    parser.add_argument("--mode", choices=["predict", "backtest", "demo"],
                        default="demo", help="Çalışma modu")

    # Tahmin parametreleri
    parser.add_argument("--sirket", type=str, default="Yeni Halka Arz A.Ş.",
                        help="Şirket adı")
    parser.add_argument("--dagitim", type=str, default="bireysele_esit",
                        choices=[d.value for d in DagitimYontemi],
                        help="Dağıtım yöntemi")
    parser.add_argument("--boyut", type=float, default=100_000_000,
                        help="Halka arz büyüklüğü (TL)")
    parser.add_argument("--katilimci", type=int, default=500,
                        help="Katılımcı beklentisi (bin kişi)")
    parser.add_argument("--kurumsal", type=float, default=0.25,
                        help="Kurumsal yatırımcı oranı (0.0-1.0)")
    parser.add_argument("--taahhut", action="store_true",
                        help="Kurumsal taahhüt var")
    parser.add_argument("--sektor", type=str, default="teknoloji",
                        choices=[s.value for s in Sektorm],
                        help="Sektör")
    parser.add_argument("--borcluluk", type=float, default=0.40,
                        help="Borçluluk oranı (0.0-1.0)")
    parser.add_argument("--kar-buyume", type=float, default=0.15,
                        help="Net kâr büyümesi (-1.0 ile +1.0)")
    parser.add_argument("--piyasa", type=str, default="boga",
                        choices=[p.value for p in PiyasaDuyarliligi],
                        help="Piyasa duyarlılığı")
    parser.add_argument("--lot-maliyet", type=float, default=None,
                        help="Lot başına düşen maliyet (TL)")
    parser.add_argument("--talep-konsantrasyon", type=float, default=None,
                        help="Talep konsantrasyonu (0-100)")

    args = parser.parse_args()

    try:
        if args.mode == "demo":
            demo_mode()
        elif args.mode == "predict":
            predict_mode(args)
        elif args.mode == "backtest":
            backtest_mode()
    except KeyboardInterrupt:
        print("\n\nKullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
