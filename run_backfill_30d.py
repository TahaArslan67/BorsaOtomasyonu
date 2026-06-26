#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Son 30 gun icin gecmise donuk normal + swing tahminleri uretir ve kaydeder.
"""

import sys
sys.path.insert(0, r'd:/otonomBorsa')

from datetime import datetime, timedelta
from gmstr_prediction_system import prediction_system, swing_predictor

DAYS = 30

now = datetime.now()
total_normal = 0
total_swing = 0
skipped = 0

print(f"Backfill basliyor: son {DAYS} gun, her 30dk normal + swing tahmin...")

# Verileri bir kere cek (performance icin)
print("GMSTR ve piyasa verileri cekiliyor...")
gmstr_full = prediction_system.fetch_gmstr_data(period="2y")
market_full = prediction_system.fetch_market_data(period="2y")
if gmstr_full is None:
    print("GMSTR verisi cekilemedi, cikiliyor.")
    sys.exit(1)
if gmstr_full.index.tz is not None:
    gmstr_full.index = gmstr_full.index.tz_localize(None)
print(f"GMSTR: {len(gmstr_full)} satir, Piyasa verisi: {len(market_full) if market_full else 0} kaynak")

for day_offset in range(DAYS, 0, -1):
    as_of_day = now - timedelta(days=day_offset)
    if as_of_day.weekday() >= 5:
        continue
    
    for hour in (9, 11, 13, 15, 17):
        minute = 30 if hour < 17 else 0
        if hour == 9 and minute < 30:
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
                (as_of, 'normal')
            )
            if cursor.fetchone()[0] > 0:
                conn.close()
                skipped += 1
                continue
            conn.close()
        except Exception as e:
            print(f"Duplicate kontrol hatasi: {e}")
            continue
        
        # Normal tahmin
        normal_pred = prediction_system.make_backfill_prediction(as_of, '4h', gmstr_full, market_full)
        if normal_pred:
            prediction_system.save_historical_prediction(
                as_of, normal_pred['current_price'], normal_pred['direction'],
                normal_pred['target_price'], normal_pred['confidence'], '4h', 'normal'
            )
            total_normal += 1
        else:
            print(f"Normal tahmin basarisiz @ {as_of}")
        
        # Swing tahmin
        if swing_predictor is not None:
            swing_pred = swing_predictor.predict_historical(as_of)
            if swing_pred:
                # Mevcut fiyat 30dk veriden gercek o anki fiyat olarak al
                current_price = gmstr_full[gmstr_full.index < as_of].Close.iloc[-1]
                prediction_system.save_historical_prediction(
                    as_of, float(current_price), swing_pred['direction'],
                    float(current_price) * (1.01 if swing_pred['direction'] == 'YUKSELIS' else 0.99),
                    swing_pred['confidence'], '1h', 'swing'
                )
                total_swing += 1
        
        if (total_normal + total_swing) % 10 == 0:
            print(f"  İlerleme: {total_normal} normal, {total_swing} swing, {skipped} atlanmis")

print(f"\nBackfill tamamlandi:")
print(f"  Normal tahmin: {total_normal}")
print(f"  Swing tahmin: {total_swing}")
print(f"  Atlanmis (zaten kayitli): {skipped}")
