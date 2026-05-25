"""
Geçmiş Halka Arz Verileri ve Backtest Altyapısı
================================================
2023-2025 döneminden örnek halka arz verileri ve
model kalibrasyonu için benchmark dataset.

NOT: Bu veriler kamuoyuna açık bilgilerden derlenmiştir.
Gerçek backtest için KAP (kamuyu aydınlatma platformu) ve
Borsa İstanbul verileriyle genişletilmelidir.
"""
from typing import List
from .data_model import HalkaArzGirdileri, GecmisHalkaArz, DagitimYontemi, Sektorm, PiyasaDuyarliligi


# ──────────────────────────────────────────────────────────────────────────────
# ÖRNEK GEÇMİŞ VERİLER (2023-2025)
# ──────────────────────────────────────────────────────────────────────────────

_ORNEK_GECMIS_VERILER: List[GecmisHalkaArz] = [
    # 2024 - Güçlü tavan serileri
    GecmisHalkaArz(
        sirket_adi="Baydöner (BAYD)",
        halka_arz_tarihi="2024-01-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Baydöner",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=75_000_000,
            katilimci_beklentisi=1200,
            kurumsal_oran=0.25,
            kurumsal_taahhut=False,
            sektor=Sektorm.GIDA,
            borcluluk_orani=0.35,
            net_kar_buyumesi=0.15,
            piyasa_duyarliligi=PiyasaDuyarliligi.BOGA,
        ),
        gerceklesen_tavan_gunu=8,
    ),
    GecmisHalkaArz(
        sirket_adi="Beyçimento (BEYÇ)",
        halka_arz_tarihi="2024-02-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Beyçimento",
            dagitim_yontemi=DagitimYontemi.TAMAMI_ESIT,
            halka_arz_boyutu_tl=120_000_000,
            katilimci_beklentisi=950,
            kurumsal_oran=0.30,
            kurumsal_taahhut=True,
            sektor=Sektorm.INSAAT,
            borcluluk_orani=0.55,
            net_kar_buyumesi=0.10,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        ),
        gerceklesen_tavan_gunu=5,
    ),
    GecmisHalkaArz(
        sirket_adi="Koton (KOTON)",
        halka_arz_tarihi="2024-03-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Koton",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=250_000_000,
            katilimci_beklentisi=800,
            kurumsal_oran=0.35,
            kurumsal_taahhut=False,
            sektor=Sektorm.HIZMET,
            borcluluk_orani=0.45,
            net_kar_buyumesi=0.05,
            piyasa_duyarliligi=PiyasaDuyarliligi.BOGA,
        ),
        gerceklesen_tavan_gunu=4,
    ),
    GecmisHalkaArz(
        sirket_adi="Mogan Enerji (MOGAN)",
        halka_arz_tarihi="2024-04-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Mogan Enerji",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=180_000_000,
            katilimci_beklentisi=1500,
            kurumsal_oran=0.20,
            kurumsal_taahhut=False,
            sektor=Sektorm.ENERJI,
            borcluluk_orani=0.40,
            net_kar_buyumesi=0.30,
            piyasa_duyarliligi=PiyasaDuyarliligi.GUCJU_BOGA,
        ),
        gerceklesen_tavan_gunu=10,
    ),
    GecmisHalkaArz(
        sirket_adi="Astor Enerji (ASTOR)",
        halka_arz_tarihi="2024-05-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Astor Enerji",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=350_000_000,
            katilimci_beklentisi=1100,
            kurumsal_oran=0.40,
            kurumsal_taahhut=True,
            sektor=Sektorm.ENERJI,
            borcluluk_orani=0.30,
            net_kar_buyumesi=0.45,
            piyasa_duyarliligi=PiyasaDuyarliligi.BOGA,
        ),
        gerceklesen_tavan_gunu=6,
    ),
    # 2024 - Orta/Zayıf tavan serileri
    GecmisHalkaArz(
        sirket_adi="Tümosan (TMSN)",
        halka_arz_tarihi="2024-06-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Tümosan",
            dagitim_yontemi=DagitimYontemi.ORANSAL,
            halka_arz_boyutu_tl=500_000_000,
            katilimci_beklentisi=400,
            kurumsal_oran=0.50,
            kurumsal_taahhut=True,
            sektor=Sektorm.URETIM,
            borcluluk_orani=0.50,
            net_kar_buyumesi=0.12,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        ),
        gerceklesen_tavan_gunu=3,
    ),
    GecmisHalkaArz(
        sirket_adi="Kervan Gıda (KRVAN)",
        halka_arz_tarihi="2024-07-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Kervan Gıda",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=95_000_000,
            katilimci_beklentisi=600,
            kurumsal_oran=0.20,
            kurumsal_taahhut=False,
            sektor=Sektorm.GIDA,
            borcluluk_orani=0.38,
            net_kar_buyumesi=0.08,
            piyasa_duyarliligi=PiyasaDuyarliligi.YATAY,
        ),
        gerceklesen_tavan_gunu=4,
    ),
    # 2024 - Riskli / Erken bozma
    GecmisHalkaArz(
        sirket_adi="Büyük Prodüksiyon (BÜYÜK)",
        halka_arz_tarihi="2024-08-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Büyük Prodüksiyon",
            dagitim_yontemi=DagitimYontemi.ORANSAL,
            halka_arz_boyutu_tl=800_000_000,
            katilimci_beklentisi=200,
            kurumsal_oran=0.60,
            kurumsal_taahhut=False,
            sektor=Sektorm.HIZMET,
            borcluluk_orani=0.65,
            net_kar_buyumesi=-0.10,
            piyasa_duyarliligi=PiyasaDuyarliligi.AYI,
        ),
        gerceklesen_tavan_gunu=1,
    ),
    # 2025 - Teknoloji/Savunma popülerliği
    GecmisHalkaArz(
        sirket_adi="Tekno Yatırım (TEKNO)",
        halka_arz_tarihi="2025-01-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Tekno Yatırım",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=60_000_000,
            katilimci_beklentisi=1800,
            kurumsal_oran=0.15,
            kurumsal_taahhut=False,
            sektor=Sektorm.TEKNOLOJI,
            borcluluk_orani=0.20,
            net_kar_buyumesi=0.60,
            piyasa_duyarliligi=PiyasaDuyarliligi.GUCJU_BOGA,
        ),
        gerceklesen_tavan_gunu=12,
    ),
    GecmisHalkaArz(
        sirket_adi="Savunma Metal (SAVUN)",
        halka_arz_tarihi="2025-02-01",
        girdiler=HalkaArzGirdileri(
            sirket_adi="Savunma Metal",
            dagitim_yontemi=DagitimYontemi.BIREYSELE_ESIT,
            halka_arz_boyutu_tl=150_000_000,
            katilimci_beklentisi=900,
            kurumsal_oran=0.30,
            kurumsal_taahhut=True,
            sektor=Sektorm.SAVUNMA,
            borcluluk_orani=0.28,
            net_kar_buyumesi=0.35,
            piyasa_duyarliligi=PiyasaDuyarliligi.BOGA,
        ),
        gerceklesen_tavan_gunu=7,
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# VERİ ERİŞİM FONKSİYONLARI
# ──────────────────────────────────────────────────────────────────────────────

def get_gecmis_veriler() -> List[GecmisHalkaArz]:
    """Tüm geçmiş halka arz verilerini döndürür."""
    return _ORNEK_GECMIS_VERILER.copy()


def get_veri_by_sektor(sektor: Sektorm) -> List[GecmisHalkaArz]:
    """Belirli bir sektöre ait geçmiş verileri döndürür."""
    return [v for v in _ORNEK_GECMIS_VERILER if v.girdiler.sektor == sektor]


def get_veri_by_kategori(min_tavan: int, max_tavan: int) -> List[GecmisHalkaArz]:
    """Tavan gün aralığına göre geçmiş verileri döndürür."""
    return [
        v for v in _ORNEK_GECMIS_VERILER
        if min_tavan <= v.gerceklesen_tavan_gunu <= max_tavan
    ]


def veri_ekle(gecmis: GecmisHalkaArz) -> None:
    """Yeni geçmiş verisi ekler (runtime)."""
    _ORNEK_GECMIS_VERILER.append(gecmis)


def verileri_json_olarak_kaydet(dosya_yolu: str = "halka_arz_gecmis.json") -> None:
    """Geçmiş verileri JSON olarak kaydeder."""
    import json
    veriler = []
    for v in _ORNEK_GECMIS_VERILER:
        veriler.append({
            "sirket_adi": v.sirket_adi,
            "halka_arz_tarihi": v.halka_arz_tarihi,
            "gerceklesen_tavan_gunu": v.gerceklesen_tavan_gunu,
            "girdiler": v.girdiler.to_dict(),
            "notlar": v.notlar,
        })
    with open(dosya_yolu, "w", encoding="utf-8") as f:
        json.dump(veriler, f, indent=2, ensure_ascii=False, default=str)


def verileri_json_dan_yukle(dosya_yolu: str = "halka_arz_gecmis.json") -> List[GecmisHalkaArz]:
    """JSON dosyasından geçmiş verileri yükler."""
    import json
    with open(dosya_yolu, "r", encoding="utf-8") as f:
        veriler = json.load(f)
    
    gecmis_listesi = []
    for v in veriler:
        g = v["girdiler"]
        girdi = HalkaArzGirdileri(
            sirket_adi=g["sirket_adi"],
            dagitim_yontemi=DagitimYontemi(g["dagitim_yontemi"]),
            halka_arz_boyutu_tl=g["halka_arz_boyutu_tl"],
            katilimci_beklentisi=g["katilimci_beklentisi"],
            kurumsal_oran=g["kurumsal_oran"],
            kurumsal_taahhut=g["kurumsal_taahhut"],
            sektor=Sektorm(g["sektor"]),
            borcluluk_orani=g["borcluluk_orani"],
            net_kar_buyumesi=g["net_kar_buyumesi"],
            piyasa_duyarliligi=PiyasaDuyarliligi(g["piyasa_duyarliligi"]),
            talep_konsantrasyonu=g.get("talep_konsantrasyonu"),
            lot_basi_dusen_maliyet=g.get("lot_basi_dusen_maliyet"),
        )
        gecmis_listesi.append(GecmisHalkaArz(
            sirket_adi=v["sirket_adi"],
            halka_arz_tarihi=v["halka_arz_tarihi"],
            girdiler=girdi,
            gerceklesen_tavan_gunu=v["gerceklesen_tavan_gunu"],
            notlar=v.get("notlar"),
        ))
    return gecmis_listesi
