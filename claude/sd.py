import yfinance as yf

# Günlük 5 yıllık veri (saatlik max 2 yıl olduğu için)
df = yf.download("GMSTR.IS", period="5y", interval="1d")

print(f"Satır: {len(df)}")
print(f"Tarih: {df.index[0]} → {df.index[-1]}")
print(df.tail(3))

df.to_csv("gmstr_gunluk.csv")
print("Kaydedildi: gmstr_gunluk.csv")