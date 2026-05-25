# GMSTR AI Model Doğruluk Test Raporu
**Test Tarihi:** 23 Mayıs 2026 13:58

---

## GMSTR - BIST Hisse Senedi (Ana Hedef)

### 4 Saatlik Tahmin Modeli
| Metrik | Değer |
|--------|-------|
| Yön Doğruluğu (Direction Accuracy) | **%73.98** |
| MAPE (Hata Oranı) | **%1.22** (çok düşük) |
| Kazanma Oranı (Win Rate) | **%68.37** |
| Test Edilen İşlem Sayısı | 196 |
| Net Kar/Zarar | **+$58.90** |
| Fiyat: Tahmin vs Gerçek | $705.89 vs $701.87 (çok yakın) |

### 1 Günlük Tahmin Modeli
| Metrik | Değer |
|--------|-------|
| Yön Doğruluğu | **%73.30** |
| MAPE | **%2.98** |
| Kazanma Oranı | **%72.73** |
| Test Edilen İşlem Sayısı | 176 |
| Net Kar/Zarar | **+$100.02** (en karlı) |

### 1 Haftalık Tahmin Modeli
| Metrik | Değer |
|--------|-------|
| Yön Doğruluğu | **%90.62** (en yüksek!) |
| MAPE | **%2.14** |
| Kazanma Oranı | **%87.50** |
| Test Edilen İşlem Sayısı | 32 |
| Net Kar/Zarar | **+$28.30** |

### GMSTR Genel Değerlendirme
- **Ortalama Yön Doğruluğu:** %79.30 ✅ **ÇOK BAŞARILI**
- **Toplam Net Kar:** +$187.21
- **En İyi Model:** 1 haftalık (168h) - %90.62 yön doğruluğu
- **En Karlı Model:** 1 günlük (24h) - $100.02 kar

---

## BTCUSDT Karşılaştırma (Kripto)

| Timeframe | Yön Doğruluğu | Win Rate | Net Kar |
|-----------|:------------:|:--------:|:-------:|
| 4 saatlik | **%46.94** | %35.20 | **-$3,502** ❌ |
| 1 günlük | **%46.59** | %39.77 | **-$4,364** ❌ |
| 1 haftalık | **%21.88** | %21.88 | **-$5,976** ❌ |

**BTC Sonuç:** Başarısız. Modeller yönü doğru tahmin edemiyor (<%50), kazanma oranı düşük, yüksek zarar.

---

## ETHUSDT Karşılaştırma (Kripto)

| Timeframe | Yön Doğruluğu | Win Rate | Net Kar |
|-----------|:------------:|:--------:|:-------:|
| 4 saatlik | **%54.08** | %44.39 | +$23.54 |
| 1 günlük | **%61.36** | %57.39 | **+$197.95** ✅ |
| 1 haftalık | **%31.25** | %31.25 | -$191.77 ❌ |

**ETH Sonuç:** 24 saatlik model orta başarılı (%61 yön doğruluğu ve $197 kar), diğerleri başarısız.

---

## ÖZET VE KARAR

Model | Yön Doğruluğu | Başarı Durumu | Açıklama
------|:------------:|:-------------:|---------
**GMSTR** ✅ | **%73-91** | ✅ **ÇOK BAŞARILI** | Tüm zaman dilimlerinde güçlü performans
BTC ❌ | %22-47 | ❌ Başarısız | Rastgele tahminden ($%50) bile kötü
ETH ⚠️ | %31-61 | ⚠️ Orta | Sadece 24h modeli kabul edilebilir

### GMSTR Modeli Neden Başarılı?
1. **Düşük MAPE (%1.2-3)**: Fiyatı gerçeğe çok yakın tahmin ediyor
2. **Yüksek Yön Doğruluğu (%74-91)**: Fiyatın yönünü (yukarı/aşağı) başarıyla tahmin ediyor
3. **Yüksek Kazanma Oranı (%68-88)**: Yaptığı işlemlerin çoğu karla sonuçlanıyor
4. **Tüm Timeframe'lerde Tutarlı**: Kısa, orta ve uzun vadede hep başarılı

### Sonuç
**EVET - GMSTR botunun AI modeli gerçekten başarılı ve doğru tahmin ediyor!** 🎯
- Özellikle BIST hissesi (GMSTR) için modeller çok iyi çalışıyor
- BTC ve ETH modelleri ise yeterince başarılı değil