import urllib.request, json

# 1. Health check
r = urllib.request.urlopen('http://localhost:5050/api/health')
health = json.loads(r.read())
print('=== HEALTH ===')
print('Status:', health['status'])
print('Models:', health['checks']['model_count'])
print('DB:', health['checks']['database'])
print('Predictions:', health['checks']['predictions'])

# 2. AI Commentary
r2 = urllib.request.urlopen('http://localhost:5050/api/ai-commentary')
ai = json.loads(r2.read())
c = ai['commentary']
print()
print('=== AI YORUMU ===')
print('Signal:', c['dominant_signal'], '-', c['action_text'])
print('Price:', c['current_price'], 'TL')
print('Monthly Return Est:', c['monthly_return_est'], '%')
pt = c['price_targets']
print('Buy Zone:', pt.get('buy_zone_low'), '-', pt.get('buy_zone_high'), 'TL')
print('Target 1:', pt.get('target_1'), 'TL')
print('Target 2:', pt.get('target_2'), 'TL')
print('Stop Loss:', pt.get('stop_loss'), 'TL')
print('Signal counts:', c['signal_counts'])
bm = c['best_model']
print('Best model:', bm['label'], '- Accuracy:', round(bm['accuracy']*100, 1), '%')
print()
print('=== MODEL SINYALLERI ===')
for sd in c['signal_details']:
    print(' ', sd['timeframe'], ':', sd['signal'], 'prob_up=', sd['prob_up'], 'acc=', sd['accuracy'])

# 3. Trades endpoint
r3 = urllib.request.urlopen('http://localhost:5050/api/trades')
trades = json.loads(r3.read())
print()
print('=== ISLEMLER ===')
print('Success:', trades['success'], '| Count:', trades['count'])

# 4. Predictions
r4 = urllib.request.urlopen('http://localhost:5050/api/predictions')
preds = json.loads(r4.read())
print()
print('=== TAHMINLER ===')
for key, p in preds.get('predictions', {}).items():
    print(' ', key, ':', p.get('signal', '--'), 'prob_up=', p.get('prob_up', 0))

# 5. Model validation
r5 = urllib.request.urlopen('http://localhost:5050/api/model-validation')
val = json.loads(r5.read())
print()
print('=== MODEL DOGRULAMA ===')
print('Overall status:', val['overall_status'])
print('Overall score:', val['overall_score'])
for r in val['validation_results']:
    print(' ', r['model'], '- test_acc:', round(r['test_accuracy']*100,1), '% - monthly_est:', r['monthly_return_estimate'], '%')
