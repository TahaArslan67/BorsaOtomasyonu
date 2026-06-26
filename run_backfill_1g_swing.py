#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Son 1 gun icin her 30dk swing tahminlerini uretir ve kaydeder.
"""

import sys
sys.path.insert(0, r'd:/otonomBorsa')

from datetime import datetime, timedelta
from gmstr_prediction_system import prediction_system, swing_predictor

DAYS = 1

now = datetime.now()
total_swing = 0
skipped = 0

print(f"Son {DAYS} gun icin swing backfill basliyor...")

if swing_predictor is None:
    print("Swing predictor bulunamadi, cikiliyor.")
    sys.exit(1)

# 30dk veriyi bir kere cek
print("GMSTR 30dk verisi cekiliyor...")
gmstr_30m = prediction_system.fetch_gmstr_data(period="60d", interval="30m")
if gmstr_30m is None:
    print("GMSTR verisi cekilemedi, cikiliyor.")
    sys.exit(1)
if gmstr_30m.index.tz is not None:
    gmstr_30m.index = gmstr_30m.index.tz_localize(None)
print(f"GMSTR 30dk: {len(gmstr_30m)} satir")

for day_offset in range(DAYS):
    as_of_day = now - timedelta(days=day_offset)
    if as_of_day.weekday() >= 5:
        continue
    
    for hour in range(9, 18):
        for minute in (0, 30):
            if hour == 9 and minute < 30:
                continue
            if hour == 17 and minute > 0:
                continue
            
            as_of = as_of_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if as_of >= now:
                continue
            
            # Ayni zamanda zaten kayit varsa atla
            try:
                conn = prediction_system.get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM predictions WHERE timestamp = ? AND model_type = ?",
                    (as_of, 'swing')
                )
                if cursor.fetchone()[0] > 0:
                    conn.close()
                    skipped += 1
                    continue
                conn.close()
            except Exception as e:
                print(f"Duplicate kontrol hatasi: {e}")
                continue
            
            swing_pred = swing_predictor.predict_historical(as_of)
            if swing_pred:
                # Mevcut fiyat 30dk veriden gercek o anki fiyat olarak al
                current_price = gmstr_30m[gmstr_30m.index < as_of].Close.iloc[-1]
                target = float(current_price) * (1.01 if swing_pred['direction'] == 'YUKSELIS' else 0.99)
                prediction_system.save_historical_prediction(
                    as_of, float(current_price), swing_pred['direction'],
                    target, swing_pred['confidence'], '1h', 'swing'
                )
                total_swing += 1
                print(f"Swing kaydedildi @ {as_of}: {swing_pred['direction']} %{swing_pred['confidence']*100:.1f}")
            else:
                print(f"Swing tahmin basarisiz @ {as_of}")

print(f"\nTamamlandi: {total_swing} swing tahmin kaydedildi, {skipped} atlanmis.")
