"""
GMSTR Gelişmiş Monitör - Flask API Sunucusu
==========================================
Özellikler:
1. Alım/Satım kayıt ve analiz
2. AI tahmin monitörü
3. Haber tabanlı AI tahmin
4. Model doğrulama ve backtest
"""
import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Proje kök dizinini path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gmstr_enhanced.trade_db import (
    add_trade, get_all_trades, get_trade_by_id,
    delete_trade, get_analysis, init_db
)
from gmstr_enhanced.news_analyzer import get_analyzer

app = Flask(__name__, static_folder=str(Path(__file__).parent))
CORS(app)

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def load_predictions():
    """Mevcut AI tahminlerini yükle."""
    pred_path = ROOT / 'gmstr_models' / 'latest_predictions.json'
    if pred_path.exists():
        try:
            with open(pred_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}


def load_training_results():
    """Model eğitim sonuçlarını yükle."""
    results_path = ROOT / 'gmstr_models' / 'training_results.json'
    if results_path.exists():
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}


# Fiyat önbelleği (60 saniye geçerli)
_price_cache = {'price': None, 'source': None, 'timestamp': None, 'fetched_at': None}

def get_live_price():
    """Canlı GMSTR fiyatını çek (önbellekli, 60 saniye geçerli)."""
    import time
    
    # Önbellek kontrolü (60 saniye)
    now = time.time()
    if (_price_cache['price'] and _price_cache['fetched_at'] and 
            now - _price_cache['fetched_at'] < 60):
        return {
            'price': _price_cache['price'],
            'source': _price_cache['source'],
            'timestamp': _price_cache['timestamp'],
        }
    
    try:
        import yfinance as yf
        ticker = yf.Ticker("GMSTR.IS")
        hist = ticker.history(period="1d", interval="1m")
        if len(hist) > 0:
            price = float(hist['Close'].iloc[-1])
            result = {
                'price': price,
                'source': 'Yahoo Finance (canlı)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            _price_cache.update({
                'price': price,
                'source': result['source'],
                'timestamp': result['timestamp'],
                'fetched_at': now,
            })
            return result
    except Exception as e:
        print(f"[PriceFetcher] Yahoo Finance hatası: {e}")
    
    # Fallback: tahmin dosyasından
    preds = load_predictions()
    if preds:
        first_pred = next(iter(preds.values()), {})
        cached_price = first_pred.get('current_price')
        if cached_price:
            return {'price': cached_price, 'source': 'Tahmin verisi (önbellek)',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    return {'price': None, 'source': 'Bilinmiyor',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}


# ============================================================
# ANA SAYFA
# ============================================================

@app.route('/')
def index():
    """Ana HTML sayfasını sun."""
    html_path = Path(__file__).parent / 'gmstr_monitor.html'
    if html_path.exists():
        return send_from_directory(str(Path(__file__).parent), 'gmstr_monitor.html')
    return jsonify({'status': 'GMSTR Enhanced API çalışıyor', 'version': '2.0'})


# ============================================================
# FIYAT VE TAHMİN API'LERİ
# ============================================================

@app.route('/api/price', methods=['GET'])
def api_price():
    """Canlı fiyat bilgisi."""
    try:
        result = get_live_price()
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """AI tahminlerini getir."""
    try:
        predictions = load_predictions()
        training = load_training_results()
        
        # Model kalite özeti
        model_quality = {}
        for key, result in training.items():
            model_quality[key] = {
                'cv_accuracy': result.get('cv_accuracy', 0),
                'test_accuracy': result.get('test_accuracy', 0),
                'test_auc': result.get('test_auc', 0),
                'is_reliable': result.get('test_accuracy', 0) > 0.55,
            }
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'model_quality': model_quality,
            'last_update': predictions.get('1d_daily', {}).get('date', 'Bilinmiyor')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-validation', methods=['GET'])
def api_model_validation():
    """Model doğrulama ve gerçekçilik kontrolü."""
    try:
        training = load_training_results()
        predictions = load_predictions()
        
        validation_results = []
        overall_score = 0
        count = 0
        
        for key, result in training.items():
            cv_acc = result.get('cv_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            test_auc = result.get('test_auc', 0)
            pos_rate = result.get('positive_rate', 0.5)
            
            # Gerçekçilik kontrolleri
            issues = []
            score = 100
            
            # 1. Overfit kontrolü
            if cv_acc > 0 and test_acc > 0:
                overfit_gap = cv_acc - test_acc
                if overfit_gap > 0.15:
                    issues.append(f'⚠️ Overfit riski: CV-Test farkı {overfit_gap:.1%}')
                    score -= 20
            
            # 2. Rastgele tahmin kontrolü (>50% olmalı)
            if test_acc < 0.50:
                issues.append(f'❌ Test doğruluğu rastgele tahminden düşük: {test_acc:.1%}')
                score -= 30
            elif test_acc < 0.53:
                issues.append(f'⚠️ Test doğruluğu zayıf: {test_acc:.1%}')
                score -= 15
            
            # 3. AUC kontrolü
            if test_auc < 0.50:
                issues.append(f'❌ AUC < 0.5: Model işe yaramıyor')
                score -= 25
            elif test_auc < 0.55:
                issues.append(f'⚠️ AUC düşük: {test_auc:.3f}')
                score -= 10
            
            # 4. Sınıf dengesi
            if pos_rate < 0.3 or pos_rate > 0.7:
                issues.append(f'⚠️ Dengesiz sınıf: Pozitif oran {pos_rate:.1%}')
                score -= 10
            
            # 5. Aylık %15 hedefi analizi
            # Her model için teorik aylık getiri hesapla (eşik koymadan)
            # Günlük beklenen getiri = (doğruluk - 0.5) * 2 * ortalama_hareket
            avg_daily_move = 0.015  # %1.5 ortalama günlük hareket (GMSTR volatilitesi)
            if test_acc > 0.5:
                daily_edge = (test_acc - 0.5) * 2 * avg_daily_move
                monthly_return = (1 + daily_edge) ** 22 - 1  # 22 işlem günü
            else:
                # Negatif edge: model zararlı
                daily_edge = (test_acc - 0.5) * 2 * avg_daily_move
                monthly_return = (1 + daily_edge) ** 22 - 1
            monthly_15_achievable = (test_acc >= 0.55 and test_auc >= 0.52 and monthly_return >= 0.15)
            
            if not issues:
                issues.append('✅ Model sağlıklı görünüyor')
            
            status = 'SAĞLIKLI' if score >= 70 else ('ZAYIF' if score >= 50 else 'SORUNLU')
            
            validation_results.append({
                'model': key,
                'cv_accuracy': round(cv_acc, 4),
                'test_accuracy': round(test_acc, 4),
                'test_auc': round(test_auc, 4),
                'positive_rate': round(pos_rate, 4),
                'score': max(0, score),
                'status': status,
                'issues': issues,
                'monthly_return_estimate': round(monthly_return * 100, 2),
                'monthly_15_achievable': monthly_15_achievable,
            })
            
            overall_score += max(0, score)
            count += 1
        
        avg_score = overall_score / count if count > 0 else 0
        
        # Genel değerlendirme
        if avg_score >= 75:
            overall_status = 'GÜÇLÜ ✅'
            overall_note = 'Modeller genel olarak sağlıklı ve güvenilir.'
        elif avg_score >= 55:
            overall_status = 'ORTA ⚠️'
            overall_note = 'Modeller çalışıyor ancak iyileştirme gerekebilir.'
        else:
            overall_status = 'ZAYIF ❌'
            overall_note = 'Modeller yeniden eğitilmeli. Daha fazla veri veya özellik gerekiyor.'
        
        # %15 aylık hedef analizi
        achievable_models = [r for r in validation_results if r['monthly_15_achievable']]
        monthly_target_note = (
            f'{len(achievable_models)}/{len(validation_results)} model teorik olarak '
            f'aylık %15 hedefine ulaşabilir. Ancak gerçek performans piyasa koşullarına bağlıdır.'
        )
        
        return jsonify({
            'success': True,
            'validation_results': validation_results,
            'overall_score': round(avg_score, 1),
            'overall_status': overall_status,
            'overall_note': overall_note,
            'monthly_target_note': monthly_target_note,
            'model_count': count,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


# ============================================================
# HABER API'LERİ
# ============================================================

@app.route('/api/news', methods=['GET'])
def api_news():
    """Gümüş haberleri ve duygu analizi."""
    try:
        analyzer = get_analyzer()
        use_sim = request.args.get('simulation', 'true').lower() == 'true'
        
        news = analyzer.fetch_news(use_simulation=use_sim)
        sentiment = analyzer.get_news_sentiment_score(news)
        context = analyzer.get_silver_market_context()
        
        # AI tahminleriyle kombine et
        predictions = load_predictions()
        combined = analyzer.get_combined_prediction(predictions, sentiment)
        
        return jsonify({
            'success': True,
            'news': news,
            'sentiment': sentiment,
            'market_context': context,
            'combined_prediction': combined,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/news/refresh', methods=['POST'])
def api_news_refresh():
    """Haberleri yenile (cache'i temizle)."""
    try:
        analyzer = get_analyzer()
        analyzer.cache = {}
        analyzer.cache_time = None
        
        news = analyzer.fetch_news(use_simulation=True)
        sentiment = analyzer.get_news_sentiment_score(news)
        
        return jsonify({
            'success': True,
            'message': f'{len(news)} haber yüklendi',
            'sentiment': sentiment,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# ALIM/SATIM İŞLEM API'LERİ
# ============================================================

@app.route('/api/trades', methods=['GET'])
def api_get_trades():
    """Tüm işlemleri getir."""
    try:
        trades = get_all_trades()
        return jsonify({'success': True, 'trades': trades, 'count': len(trades)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trades', methods=['POST'])
def api_add_trade():
    """Yeni işlem ekle."""
    try:
        data = request.get_json()
        
        # Zorunlu alanlar
        required = ['trade_type', 'trade_date', 'trade_time', 'price', 'quantity']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Eksik alan: {field}'}), 400
        
        # Validasyon
        trade_type = data['trade_type'].upper()
        if trade_type not in ['BUY', 'SELL']:
            return jsonify({'success': False, 'error': 'trade_type BUY veya SELL olmalı'}), 400
        
        price = float(data['price'])
        quantity = float(data['quantity'])
        
        if price <= 0:
            return jsonify({'success': False, 'error': 'Fiyat 0\'dan büyük olmalı'}), 400
        if quantity <= 0:
            return jsonify({'success': False, 'error': 'Miktar 0\'dan büyük olmalı'}), 400
        
        # Tarih formatı kontrolü
        try:
            datetime.strptime(data['trade_date'], '%Y-%m-%d')
        except ValueError:
            return jsonify({'success': False, 'error': 'Tarih formatı YYYY-MM-DD olmalı'}), 400
        
        trade_id = add_trade(
            trade_type=trade_type,
            trade_date=data['trade_date'],
            trade_time=data['trade_time'],
            price=price,
            quantity=quantity,
            commission=float(data.get('commission', 0)),
            notes=data.get('notes', ''),
            bot_signal=data.get('bot_signal', ''),
        )
        
        return jsonify({
            'success': True,
            'trade_id': trade_id,
            'message': f'İşlem #{trade_id} başarıyla eklendi',
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trades/<int:trade_id>', methods=['GET'])
def api_get_trade(trade_id):
    """Belirli bir işlemi getir."""
    try:
        trade = get_trade_by_id(trade_id)
        if not trade:
            return jsonify({'success': False, 'error': 'İşlem bulunamadı'}), 404
        return jsonify({'success': True, 'trade': trade})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trades/<int:trade_id>', methods=['DELETE'])
def api_delete_trade(trade_id):
    """İşlemi sil."""
    try:
        success = delete_trade(trade_id)
        if not success:
            return jsonify({'success': False, 'error': 'İşlem bulunamadı'}), 404
        return jsonify({'success': True, 'message': f'İşlem #{trade_id} silindi'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# ANALİZ API'LERİ
# ============================================================

@app.route('/api/analysis', methods=['GET'])
def api_analysis():
    """Kapsamlı kar/zarar analizi."""
    try:
        analysis = get_analysis()
        
        # Canlı fiyatı ekle (açık pozisyonlar için)
        price_data = get_live_price()
        current_price = price_data.get('price')
        
        if current_price and analysis['open_positions']:
            for pos in analysis['open_positions']:
                open_price = pos.get('open_price', 0)
                qty = pos.get('quantity', 0)
                if open_price and qty:
                    unrealized_pl = (current_price - open_price) * qty
                    unrealized_pct = ((current_price - open_price) / open_price) * 100
                    pos['current_price'] = current_price
                    pos['unrealized_pl'] = round(unrealized_pl, 2)
                    pos['unrealized_pct'] = round(unrealized_pct, 2)
        
        analysis['current_price'] = current_price
        
        return jsonify({'success': True, 'analysis': analysis})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/analysis/monthly', methods=['GET'])
def api_monthly_analysis():
    """Aylık performans analizi."""
    try:
        analysis = get_analysis()
        monthly_pl = analysis.get('monthly_pl', {})
        
        # Aylık hedef karşılaştırması
        monthly_stats = []
        for month, pl in monthly_pl.items():
            # Aylık başlangıç sermayesini tahmin et (basit)
            month_trades = [t for t in analysis['all_trades'] 
                          if t['trade_date'].startswith(month)]
            
            if month_trades:
                avg_price = sum(t['price'] for t in month_trades) / len(month_trades)
                avg_qty = sum(t['quantity'] for t in month_trades) / len(month_trades)
                estimated_capital = avg_price * avg_qty * 10  # Tahmin
                monthly_return_pct = (pl / estimated_capital * 100) if estimated_capital > 0 else 0
            else:
                monthly_return_pct = 0
            
            monthly_stats.append({
                'month': month,
                'profit_loss': pl,
                'return_pct': round(monthly_return_pct, 2),
                'target_15_met': monthly_return_pct >= 15,
                'status': '✅ Hedef Aşıldı' if monthly_return_pct >= 15 else 
                         ('📈 Pozitif' if pl > 0 else '📉 Negatif'),
            })
        
        return jsonify({
            'success': True,
            'monthly_stats': monthly_stats,
            'total_months': len(monthly_stats),
            'profitable_months': sum(1 for m in monthly_stats if m['profit_loss'] > 0),
            'target_met_months': sum(1 for m in monthly_stats if m['target_15_met']),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# MODEL EĞİTİM API'Sİ
# ============================================================

@app.route('/api/train', methods=['POST'])
def api_train():
    """Model eğitimini başlat (arka planda)."""
    try:
        import subprocess
        import threading
        
        def run_training():
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'gmstr_system.main', '--mode', 'full',
                     '--target-mode', 'dynamic'],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 dakika timeout
                )
                # Sonucu kaydet
                log_path = ROOT / 'gmstr_enhanced' / 'training_log.txt'
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== EĞİTİM TAMAMLANDI: {datetime.now()} ===\n")
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n=== HATALAR ===\n")
                        f.write(result.stderr)
            except Exception as e:
                log_path = ROOT / 'gmstr_enhanced' / 'training_log.txt'
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"EĞİTİM HATASI: {e}\n")
        
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model eğitimi başlatıldı. Bu işlem 5-10 dakika sürebilir.',
            'log_endpoint': '/api/train/log'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train/log', methods=['GET'])
def api_train_log():
    """Eğitim logunu getir."""
    try:
        log_path = ROOT / 'gmstr_enhanced' / 'training_log.txt'
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({'success': True, 'log': content})
        return jsonify({'success': True, 'log': 'Henüz eğitim başlatılmadı.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# DASHBOARD API'Sİ (Tüm verileri tek seferde)
# ============================================================

@app.route('/api/dashboard', methods=['GET'])
def api_dashboard():
    """Dashboard için tüm verileri tek seferde getir."""
    try:
        # Fiyat
        price_data = get_live_price()
        
        # Tahminler
        predictions = load_predictions()
        
        # Haberler
        analyzer = get_analyzer()
        news = analyzer.fetch_news(use_simulation=True)
        sentiment = analyzer.get_news_sentiment_score(news)
        context = analyzer.get_silver_market_context()
        combined = analyzer.get_combined_prediction(predictions, sentiment)
        
        # Analiz özeti
        analysis = get_analysis()
        
        # Model kalitesi
        training = load_training_results()
        model_scores = {}
        for key, result in training.items():
            model_scores[key] = {
                'test_accuracy': result.get('test_accuracy', 0),
                'test_auc': result.get('test_auc', 0),
            }
        
        return jsonify({
            'success': True,
            'price': price_data,
            'predictions': predictions,
            'news_count': len(news),
            'news_sentiment': sentiment,
            'market_context': context,
            'combined_signal': combined,
            'trade_summary': analysis.get('summary', {}),
            'performance': analysis.get('performance', {}),
            'open_positions': len(analysis.get('open_positions', [])),
            'model_scores': model_scores,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


# ============================================================
# TAHMİN GRAFİĞİ API'Sİ
# ============================================================

@app.route('/api/price-chart', methods=['GET'])
def api_price_chart():
    """
    Geçmiş GMSTR fiyat verisi + gelecek AI tahmin gidişatı.
    Grafik için: son 90 gün gerçek fiyat + 1d/3d/5d/10d tahmin noktaları.
    """
    try:
        import pickle
        import numpy as np
        
        # Veri yükle
        csv_path = ROOT / 'claude' / 'areaxdatetime.csv'
        if not csv_path.exists():
            return jsonify({'success': False, 'error': 'Veri dosyası bulunamadı'}), 404
        
        import pandas as pd
        # CSV'yi doğrudan oku (data_loader bağımlılığını kaldır)
        raw = pd.read_csv(csv_path)
        raw.columns = [str(c).strip() for c in raw.columns]
        # Tarih kolonunu bul
        date_col = None
        for candidate in ['category', 'Unnamed: 0', 'Date', 'date']:
            if candidate in raw.columns:
                date_col = candidate
                break
        if date_col == 'category':
            raw['Date'] = pd.to_datetime(raw[date_col], format='%a %b %d %Y', errors='coerce')
        elif date_col:
            raw['Date'] = pd.to_datetime(raw[date_col], errors='coerce')
        else:
            raw['Date'] = pd.to_datetime(raw.index, errors='coerce')
        raw = raw.dropna(subset=['Date']).set_index('Date').sort_index()

        # Fiyat ölçekleme: gercek_data.csv varsa lineer regresyon
        fund_ret = pd.to_numeric(raw['Net Getiri'], errors='coerce')
        gercek_csv = ROOT / 'claude' / 'gercek_data.csv'
        close_price = None
        if gercek_csv.exists():
            try:
                graw = pd.read_csv(gercek_csv, header=None)
                first_row_g = graw.iloc[0].astype(str).tolist()
                if 'Price' in first_row_g or 'Close' in first_row_g:
                    header_row = graw.iloc[0].tolist()
                    gdata = graw.iloc[3:].copy()
                    gdata.columns = header_row
                    date_c = next((c for c in gdata.columns if str(c).lower() in ['price','date','datetime']), None)
                    if date_c:
                        gdata = gdata.rename(columns={date_c: 'Date'}).set_index('Date')
                else:
                    gdata = graw.copy()
                    gdata.columns = ['Date','Open','High','Low','Close','Volume']
                    gdata = gdata.set_index('Date')
                gdata['Close'] = pd.to_numeric(gdata['Close'], errors='coerce')
                gdata.index = pd.to_datetime(gdata.index, errors='coerce', utc=True)
                gdata = gdata[gdata.index.notna()][['Close']].dropna().sort_index()
                daily = gdata.resample('D').agg({'Close': 'last'}).dropna()
                if daily.index.tz is not None:
                    daily.index = daily.index.tz_localize(None)
                daily_renamed = daily.rename(columns={'Close': 'Close_real'})
                common = raw.join(daily_renamed, how='inner')
                if len(common) >= 30:
                    x = common['Net Getiri'].astype(float).fillna(0).values
                    y = common['Close_real'].ffill().fillna(0).values
                    valid = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[valid], y[valid]
                    if len(x) >= 30 and np.var(x) > 0:
                        b_slope = np.cov(x, y)[0, 1] / np.var(x)
                        a_intercept = np.mean(y) - b_slope * np.mean(x)
                        close_price = a_intercept + b_slope * fund_ret
            except Exception as ex:
                print(f"[app.py] Fiyat ölçekleme hatası: {ex}")

        if close_price is None:
            close_price = 100.0 * (1.0 + fund_ret / 100.0)

        # Sentetik OHLCV
        log_price = np.log(close_price.replace(0, np.nan))
        daily_log_ret = log_price.diff()
        vol = daily_log_ret.rolling(window=20, min_periods=5).std().fillna(daily_log_ret.std())
        open_price = close_price.shift(1).fillna(close_price)
        high_price = np.maximum(close_price * np.exp(np.abs(daily_log_ret) * 0.5 + vol * 0.3),
                                np.maximum(open_price, close_price))
        low_price = np.minimum(close_price * np.exp(-np.abs(daily_log_ret) * 0.5 - vol * 0.3),
                               np.minimum(open_price, close_price))
        volume = (np.abs(daily_log_ret) * 1e6).fillna(0).astype(int)

        df = pd.DataFrame({
            'Open': open_price, 'High': high_price, 'Low': low_price,
            'Close': close_price, 'Volume': volume,
        }).dropna()

        # Son N günlük veri
        days = int(request.args.get('days', 90))
        days = max(30, min(365, days))
        df_recent = df.tail(days)
        
        # Geçmiş fiyat verisi
        historical = []
        for idx, row in df_recent.iterrows():
            historical.append({
                'date': idx.strftime('%Y-%m-%d'),
                'price': round(float(row['Close']), 2),
                'high': round(float(row.get('High', row['Close'])), 2),
                'low': round(float(row.get('Low', row['Close'])), 2),
                'volume': int(row.get('Volume', 0)) if not np.isnan(row.get('Volume', 0)) else 0,
            })
        
        # Son fiyat - canlı fiyatı kullan (varsa)
        live_price_data = get_live_price()
        live_price = live_price_data.get('price')
        
        last_price_csv = float(df_recent['Close'].iloc[-1])
        last_date = df_recent.index[-1]
        
        # Canlı fiyat varsa ve CSV'den farklıysa, bugünün tarihiyle ekle
        if live_price and abs(live_price - last_price_csv) > 1:
            today_str = datetime.now().strftime('%Y-%m-%d')
            # Eğer son CSV tarihi bugün değilse, canlı fiyatı ekle
            if last_date.strftime('%Y-%m-%d') != today_str:
                historical.append({
                    'date': today_str,
                    'price': round(live_price, 2),
                    'high': round(live_price, 2),
                    'low': round(live_price, 2),
                    'volume': 0,
                    'is_live': True,
                })
            last_price = live_price
        else:
            last_price = last_price_csv
        
        # AI tahmin noktaları (gelecek)
        predictions = load_predictions()
        forecast_points = []
        
        # Mevcut tahminlerden gelecek noktaları oluştur
        horizons = {'1d_daily': 1, '3d_daily': 3, '5d_daily': 5, '10d_daily': 10}
        
        for model_key, days_ahead in horizons.items():
            pred = predictions.get(model_key, {})
            if not pred:
                continue
            
            prob_up = pred.get('prob_up', 0.5)
            confidence = pred.get('confidence', 0.5)
            
            # Fiyat tahmini: olasılığa göre yön ve büyüklük
            # Ortalama günlük hareket ~%1.5
            avg_daily_move = 0.015
            direction = 1 if prob_up > 0.5 else -1
            magnitude = abs(prob_up - 0.5) * 2 * avg_daily_move * days_ahead
            
            forecast_price = last_price * (1 + direction * magnitude)
            
            # Güven aralığı (±1 std)
            volatility = 0.015 * np.sqrt(days_ahead)  # Günlük vol * sqrt(gün)
            upper_bound = forecast_price * (1 + volatility)
            lower_bound = forecast_price * (1 - volatility)
            
            # Tarih hesapla - BUGÜNDEN itibaren iş günleri say (CSV son tarihi değil!)
            from datetime import timedelta
            today_date = datetime.now().date()
            forecast_date = today_date
            business_days = 0
            while business_days < days_ahead:
                forecast_date += timedelta(days=1)
                if forecast_date.weekday() < 5:  # Pazartesi-Cuma
                    business_days += 1
            
            forecast_points.append({
                'model': model_key,
                'date': forecast_date.strftime('%Y-%m-%d'),
                'days_ahead': days_ahead,
                'forecast_price': round(forecast_price, 2),
                'upper_bound': round(upper_bound, 2),
                'lower_bound': round(lower_bound, 2),
                'prob_up': round(prob_up, 3),
                'confidence': round(confidence, 3),
                'direction': 'UP' if prob_up > 0.5 else 'DOWN',
                'signal': pred.get('signal', 'BEKLE'),
            })
        
        # Teknik göstergeler (son değerler) - doğrudan hesapla
        indicators = {}
        try:
            close_s = df_recent['Close']
            
            # MA'lar
            for w in [5, 10, 20, 50]:
                ma = close_s.rolling(w).mean()
                val = ma.iloc[-1]
                if not np.isnan(val):
                    indicators[f'ma_{w}'] = round(float(val), 2)
            
            # RSI (14)
            delta = close_s.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]
            if not np.isnan(rsi_val):
                indicators['rsi_14'] = round(float(rsi_val), 1)
            
            # Bollinger Bands (20)
            bb_mid = close_s.rolling(20).mean()
            bb_std = close_s.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            for col, val in [('bb_mid', bb_mid.iloc[-1]), ('bb_upper', bb_upper.iloc[-1]), ('bb_lower', bb_lower.iloc[-1])]:
                if not np.isnan(val):
                    indicators[col] = round(float(val), 2)
        except Exception as ind_err:
            print(f"[app.py] Gösterge hesaplama hatası: {ind_err}")
        
        return jsonify({
            'success': True,
            'historical': historical,
            'forecast': forecast_points,
            'last_price': round(last_price, 2),
            'last_date': last_date.strftime('%Y-%m-%d'),
            'indicators': indicators,
            'data_points': len(historical),
            'forecast_count': len(forecast_points),
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


# ============================================================
# AI YORUMU ENDPOİNT'İ
# ============================================================

@app.route('/api/ai-commentary', methods=['GET'])
def api_ai_commentary():
    """AI tabanlı alım/satım yorumu üret."""
    try:
        predictions = load_predictions()
        training = load_training_results()
        price_data = get_live_price()
        current_price = price_data.get('price')
        
        if not current_price:
            return jsonify({'success': False, 'error': 'Fiyat alınamadı'}), 500
        
        # En güvenilir modeli bul (en yüksek test doğruluğu)
        best_model_key = None
        best_accuracy = 0
        for key, result in training.items():
            acc = result.get('test_accuracy', 0)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_key = key
        
        # Tüm modellerin sinyallerini topla
        signals = {'AL': 0, 'SAT': 0, 'BEKLE': 0}
        signal_details = []
        total_prob_up = 0
        model_count = 0
        
        timeframe_labels = {
            '15m_15min': '15 Dakika',
            '1h_hourly': '1 Saat',
            '4h_hourly': '4 Saat',
            '1d_daily': '1 Gün',
            '3d_daily': '3 Gün',
            '5d_daily': '5 Gün',
            '10d_daily': '10 Gün',
        }
        
        for key, pred in predictions.items():
            signal = pred.get('signal', 'BEKLE')
            prob_up = pred.get('prob_up', 0.5)
            strength = pred.get('strength', 'NÖTR')
            label = timeframe_labels.get(key, key)
            acc = training.get(key, {}).get('test_accuracy', 0)
            
            if signal in signals:
                signals[signal] += 1
            
            total_prob_up += prob_up
            model_count += 1
            
            signal_details.append({
                'timeframe': label,
                'signal': signal,
                'strength': strength,
                'prob_up': round(prob_up, 4),
                'accuracy': round(acc, 4),
            })
        
        avg_prob_up = total_prob_up / model_count if model_count > 0 else 0.5
        
        # Genel sinyal belirle (çoğunluk oylaması)
        dominant_signal = max(signals, key=signals.get)
        
        # Fiyat hedefleri hesapla
        # Volatilite tahmini: günlük %1.5 hareket varsayımı
        daily_vol = 0.015
        
        if dominant_signal == 'AL':
            buy_low = round(current_price * 0.99, 2)
            buy_high = round(current_price * 1.005, 2)
            target_1 = round(current_price * 1.03, 2)
            target_2 = round(current_price * 1.06, 2)
            stop_loss = round(current_price * 0.97, 2)
            action_text = "ALIM FIRSATI"
            action_emoji = "🟢"
        elif dominant_signal == 'SAT':
            buy_low = None
            buy_high = None
            target_1 = round(current_price * 0.97, 2)
            target_2 = round(current_price * 0.94, 2)
            stop_loss = round(current_price * 1.03, 2)
            action_text = "SATIŞ SİNYALİ"
            action_emoji = "🔴"
        else:
            buy_low = round(current_price * 0.985, 2)
            buy_high = round(current_price * 1.015, 2)
            target_1 = round(current_price * 1.025, 2)
            target_2 = round(current_price * 1.05, 2)
            stop_loss = round(current_price * 0.975, 2)
            action_text = "BEKLEME MODU"
            action_emoji = "🟡"
        
        # Aylık getiri tahmini
        avg_accuracy = best_accuracy if best_accuracy > 0 else 0.54
        edge = (avg_accuracy - 0.5) * 2
        monthly_vol = daily_vol * (22 ** 0.5)
        monthly_return_est = edge * monthly_vol * 100 * 3
        monthly_return_est = max(0, min(monthly_return_est, 15))
        
        # Hedef tarihleri hesapla (iş günleri bazında)
        from datetime import timedelta
        def add_business_days(start_date, n_days):
            d = start_date
            added = 0
            while added < n_days:
                d += timedelta(days=1)
                if d.weekday() < 5:
                    added += 1
            return d
        
        today = datetime.now().date()
        # Hedef 1: %3 hareket için ortalama kaç gün? (günlük vol %1.5 → ~2 gün)
        # Hedef 2: %6 hareket için ~4 gün
        if dominant_signal == 'AL':
            t1_pct = 0.03  # %3
            t2_pct = 0.06  # %6
        elif dominant_signal == 'SAT':
            t1_pct = 0.03
            t2_pct = 0.06
        else:
            t1_pct = 0.025
            t2_pct = 0.05
        
        # Tahmini gün sayısı: hedef% / günlük_vol
        t1_days = max(1, int(t1_pct / daily_vol))
        t2_days = max(2, int(t2_pct / daily_vol))
        
        t1_date = add_business_days(today, t1_days)
        t2_date = add_business_days(today, t2_days)
        
        target_dates = {
            'target_1_date': t1_date.strftime('%d.%m.%Y'),
            'target_2_date': t2_date.strftime('%d.%m.%Y'),
            'target_1_days': t1_days,
            'target_2_days': t2_days,
        }
        
        # En iyi model bilgisi
        best_pred = predictions.get(best_model_key, {})
        best_signal = best_pred.get('signal', 'BEKLE')
        best_prob = best_pred.get('prob_up', 0.5)
        best_label = timeframe_labels.get(best_model_key, best_model_key)
        
        # Yorum metni oluştur
        now_str = datetime.now().strftime('%d.%m.%Y %H:%M')
        
        # Ana yorum
        if dominant_signal == 'AL':
            main_comment = (
                f"GMSTR şu an {current_price:.2f} TL seviyesinde işlem görüyor. "
                f"Modellerimizin {signals['AL']}/{model_count}'i AL sinyali veriyor. "
                f"En güvenilir modelimiz ({best_label}, %{best_accuracy*100:.1f} doğruluk) "
                f"{best_signal} sinyali üretiyor (yukarı olasılık: %{best_prob*100:.1f}). "
                f"Alım için {buy_low:.2f}-{buy_high:.2f} TL bölgesi uygun görünüyor."
            )
            strategy_comment = (
                f"Strateji: {buy_low:.2f}-{buy_high:.2f} TL arasında kademeli alım yapın. "
                f"İlk hedef {target_1:.2f} TL, ikinci hedef {target_2:.2f} TL. "
                f"Stop-loss: {stop_loss:.2f} TL altında pozisyonu kapatın."
            )
        elif dominant_signal == 'SAT':
            main_comment = (
                f"GMSTR şu an {current_price:.2f} TL seviyesinde işlem görüyor. "
                f"Modellerimizin {signals['SAT']}/{model_count}'i SAT sinyali veriyor. "
                f"En güvenilir modelimiz ({best_label}, %{best_accuracy*100:.1f} doğruluk) "
                f"{best_signal} sinyali üretiyor (yukarı olasılık: %{best_prob*100:.1f}). "
                f"Mevcut pozisyonlar için dikkatli olunması önerilir."
            )
            strategy_comment = (
                f"Strateji: {current_price:.2f} TL üzerinde satış fırsatı değerlendirin. "
                f"Destek seviyeleri: {target_1:.2f} TL ve {target_2:.2f} TL. "
                f"Stop-loss: {stop_loss:.2f} TL üzerinde pozisyonu kapatın."
            )
        else:
            main_comment = (
                f"GMSTR şu an {current_price:.2f} TL seviyesinde işlem görüyor. "
                f"Modeller karışık sinyal veriyor (AL:{signals['AL']}, SAT:{signals['SAT']}, BEKLE:{signals['BEKLE']}). "
                f"En güvenilir modelimiz ({best_label}, %{best_accuracy*100:.1f} doğruluk) "
                f"{best_signal} sinyali üretiyor (yukarı olasılık: %{best_prob*100:.1f}). "
                f"Piyasa yönü belirsiz, temkinli yaklaşım önerilir."
            )
            strategy_comment = (
                f"Strateji: {buy_low:.2f}-{buy_high:.2f} TL arasında küçük alım yapılabilir. "
                f"Hedef: {target_1:.2f}-{target_2:.2f} TL. "
                f"Stop-loss: {stop_loss:.2f} TL altında pozisyonu kapatın."
            )
        
        monthly_comment = (
            f"Aylık Getiri Tahmini: Model doğruluğu (%{avg_accuracy*100:.1f}) baz alınarak "
            f"aylık ~%{monthly_return_est:.1f} getiri potansiyeli hesaplanmaktadır. "
            f"(Risk uyarısı: Geçmiş performans gelecek getiriyi garanti etmez.)"
        )
        
        return jsonify({
            'success': True,
            'commentary': {
                'timestamp': now_str,
                'current_price': current_price,
                'dominant_signal': dominant_signal,
                'action_text': action_text,
                'action_emoji': action_emoji,
                'main_comment': main_comment,
                'strategy_comment': strategy_comment,
                'monthly_comment': monthly_comment,
                'monthly_return_est': round(monthly_return_est, 1),
                'avg_prob_up': round(avg_prob_up, 4),
                'signal_counts': signals,
                'price_targets': {
                    'buy_zone_low': buy_low,
                    'buy_zone_high': buy_high,
                    'target_1': target_1,
                    'target_2': target_2,
                    'stop_loss': stop_loss,
                },
                'best_model': {
                    'key': best_model_key,
                    'label': best_label,
                    'accuracy': round(best_accuracy, 4),
                    'signal': best_signal,
                    'prob_up': round(best_prob, 4),
                },
                'signal_details': signal_details,
                'target_dates': target_dates,
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


# ============================================================
# TELEGRAM BİLDİRİM API'Sİ
# ============================================================

@app.route('/api/telegram/send-commentary', methods=['POST'])
def api_telegram_send_commentary():
    """AI yorumunu Telegram'a gönder."""
    try:
        from gmstr_enhanced.telegram_notifier import send_ai_commentary
        
        # Önce AI yorumunu al
        predictions = load_predictions()
        training = load_training_results()
        price_data = get_live_price()
        current_price = price_data.get('price')
        
        if not current_price:
            return jsonify({'success': False, 'error': 'Fiyat alınamadı'}), 500
        
        # ai-commentary endpoint'inden veri al (iç çağrı)
        import urllib.request
        r = urllib.request.urlopen('http://localhost:5050/api/ai-commentary', timeout=10)
        ai_data = json.loads(r.read())
        
        if not ai_data.get('success'):
            return jsonify({'success': False, 'error': 'AI yorumu alınamadı'}), 500
        
        ok = send_ai_commentary(ai_data['commentary'])
        
        return jsonify({
            'success': ok,
            'message': 'Telegram mesajı gönderildi' if ok else 'Telegram mesajı gönderilemedi',
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/telegram/send-daily', methods=['POST'])
def api_telegram_send_daily():
    """Günlük özeti Telegram'a gönder."""
    try:
        from gmstr_enhanced.telegram_notifier import send_daily_summary
        ok = send_daily_summary()
        return jsonify({
            'success': ok,
            'message': 'Günlük özet gönderildi' if ok else 'Gönderilemedi',
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/telegram/test', methods=['POST'])
def api_telegram_test():
    """Telegram bağlantısını test et."""
    try:
        from gmstr_enhanced.telegram_notifier import send_message
        ok = send_message('🥈 <b>GMSTR Bot Test</b>\nBağlantı başarılı! ✅')
        return jsonify({
            'success': ok,
            'message': 'Test mesajı gönderildi' if ok else 'Gönderilemedi',
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# SAĞLIK KONTROLÜ
# ============================================================

@app.route('/api/health', methods=['GET'])
def api_health():
    """API sağlık kontrolü."""
    checks = {
        'api': True,
        'database': False,
        'predictions': False,
        'models': False,
    }
    
    try:
        from gmstr_enhanced.trade_db import get_connection
        conn = get_connection()
        conn.close()
        checks['database'] = True
    except:
        pass
    
    pred_path = ROOT / 'gmstr_models' / 'latest_predictions.json'
    checks['predictions'] = pred_path.exists()
    
    model_files = list((ROOT / 'gmstr_models').glob('simple_*.pkl'))
    checks['models'] = len(model_files) > 0
    checks['model_count'] = len(model_files)
    
    all_ok = all(v for k, v in checks.items() if isinstance(v, bool))
    
    return jsonify({
        'status': 'OK' if all_ok else 'PARTIAL',
        'checks': checks,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  GMSTR GELİŞMİŞ MONİTÖR v2.0")
    print("  http://localhost:5050")
    print("="*60 + "\n")
    
    init_db()
    app.run(host='0.0.0.0', port=5050, debug=False)
