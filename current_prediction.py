from gmstr_prediction_system import GMSTRPredictionSystem
import json

ps = GMSTRPredictionSystem()
result = ps.make_prediction('4h')

if result:
    print('=' * 60)
    print('GMSTR GUNCEL TAHMIN')
    print('=' * 60)
    print(f"Fiyat: {result['current_price']:.2f}")
    print(f"Yon: {result['direction']}")
    print(f"Hedef: {result['target_price']:.2f}")
    print(f"Guven: {result['confidence']*100:.1f}%")
    print(f"Rejim: {result['regime']} (ADX={result['adx']:.1f})")
    print(f"Volatilite: {result['volatility_annual']:.2f}")
    print()
    
    # 30,000 TL pozisyon degerlendirmesi
    current_val = 30000
    if result['direction'] == 'YUKSELIS':
        target_val = current_val * (result['target_price'] / result['current_price'])
        pot_profit = target_val - current_val
        print(f'Pozisyon: 30,000 TL')
        print(f'Hedef deger: {target_val:,.0f} TL')
        print(f'Potansiyel kar: +{pot_profit:,.0f} TL')
        print()
        print('ONERI: Tut veya ekle (kalite filtresi gectiyse)')
    elif result['direction'] == 'DUSUS':
        target_val = current_val * (result['target_price'] / result['current_price'])
        pot_loss = current_val - target_val
        print(f'Pozisyon: 30,000 TL')
        print(f'Hedef deger: {target_val:,.0f} TL')
        print(f'Potansiyel kayip: -{pot_loss:,.0f} TL')
        print()
        print('ONERI: Dususu bekleyip daha dusukten tekrar al, veya stop-loss uygula')
    else:
        print('Pozisyon: 30,000 TL')
        print('ONERI: BEKLE - Sistem emin degil, pozisyonu koru')
    
    print()
    print('Detay:', json.dumps({k:v for k,v in result.items() if k not in ['risk_info', 'tf_details']}, indent=2, default=str))
else:
    print('Tahmin alinamadi')
