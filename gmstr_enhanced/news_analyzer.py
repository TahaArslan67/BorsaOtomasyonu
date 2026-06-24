"""
GMSTR Haber Analizi ve AI Tahmin Sistemi
Gümüş (GMSTR) ile ilgili haberleri analiz eder ve fiyat etkisini tahmin eder.
"""
import json
import re
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import urllib.request
import urllib.parse


# Gümüş ile ilgili anahtar kelimeler ve etki skorları
SILVER_KEYWORDS = {
    # Pozitif sinyaller
    'positive': {
        'gümüş': 1.5, 'silver': 1.5, 'gmstr': 2.0,
        'altın': 0.8, 'gold': 0.8,
        'enflasyon': 1.2, 'inflation': 1.2,
        'faiz düşüş': 1.5, 'rate cut': 1.5, 'fed dovish': 1.5,
        'dolar zayıf': 1.3, 'dollar weak': 1.3, 'usd düşüş': 1.3,
        'güvenli liman': 1.4, 'safe haven': 1.4,
        'sanayi talebi': 1.2, 'industrial demand': 1.2,
        'güneş enerjisi': 1.1, 'solar': 1.1, 'yeşil enerji': 1.1,
        'jeopolitik': 1.0, 'geopolitical': 1.0,
        'merkez bankası alım': 1.5, 'central bank buying': 1.5,
        'rekor': 0.8, 'yükseliş': 1.0, 'artış': 0.8, 'rally': 1.0,
        'bist yükseliş': 0.7, 'borsa artış': 0.7,
        'tl değer kaybı': 1.2, 'kur artış': 1.0,
        'etf alım': 1.3, 'emtia rallisi': 1.2, 'değerli metal': 1.0,
        'portföy çeşitlendirme': 0.9, 'hedge': 0.8,
    },
    # Negatif sinyaller
    'negative': {
        'faiz artış': -1.5, 'rate hike': -1.5, 'fed hawkish': -1.5,
        'dolar güçlü': -1.3, 'dollar strong': -1.3, 'usd artış': -1.3,
        'deflasyon': -1.2, 'deflation': -1.2,
        'sanayi düşüş': -1.0, 'industrial slowdown': -1.0,
        'satış baskısı': -1.2, 'sell off': -1.2,
        'düşüş': -0.8, 'gerileme': -0.8, 'kayıp': -0.8,
        'resesyon': -0.7, 'recession': -0.7,
        'bist düşüş': -0.7, 'borsa düşüş': -0.7,
        'tl değer kazanç': -0.8, 'kur düşüş': -0.8,
        'kripto yükseliş': -0.5,
        'etf satış': -1.1, 'pozisyon kapatma': -0.9,
        'likidite sıkışması': -1.0, 'risk iştahı azalma': -0.8,
    }
}

# RSS/Haber kaynakları (ücretsiz, API gerektirmeyen)
NEWS_SOURCES = [
    {
        'name': 'Investing.com Silver',
        'url': 'https://tr.investing.com/commodities/silver-news',
        'type': 'web'
    },
    {
        'name': 'Yahoo Finance Silver',
        'url': 'https://finance.yahoo.com/rss/headline?s=SI=F',
        'type': 'rss'
    },
]

# Simüle edilmiş haber veritabanı - 15 haber (10-20 arası)
SIMULATED_NEWS_TEMPLATES = [
    {
        'title': 'Fed faiz kararı beklentileri gümüş fiyatlarını etkiliyor',
        'summary': 'ABD Merkez Bankası\'nın faiz kararı öncesinde gümüş fiyatları dalgalanıyor. Analistler faiz indirim beklentisinin gümüşü destekleyebileceğini söylüyor.',
        'sentiment': 'positive',
        'impact': 1.3,
        'source': 'Finans Haberleri',
        'category': 'Makro'
    },
    {
        'title': 'Sanayi talebi gümüş fiyatlarını destekliyor',
        'summary': 'Güneş paneli ve elektrikli araç üretimindeki artış gümüş talebini yukarı çekiyor. Uzmanlar bu trendin devam edeceğini öngörüyor.',
        'sentiment': 'positive',
        'impact': 1.1,
        'source': 'Emtia Analiz',
        'category': 'Sanayi'
    },
    {
        'title': 'Dolar endeksi güçleniyor, gümüş baskı altında',
        'summary': 'ABD dolarının güçlenmesi emtia fiyatları üzerinde baskı oluşturuyor. Gümüş kısa vadede zayıf seyredebilir.',
        'sentiment': 'negative',
        'impact': -1.2,
        'source': 'Döviz Haberleri',
        'category': 'Döviz'
    },
    {
        'title': 'Türkiye enflasyon verileri açıklandı',
        'summary': 'Yüksek enflasyon ortamında yatırımcılar değer saklama aracı olarak gümüşe yöneliyor. GMSTR işlem hacmi artıyor.',
        'sentiment': 'positive',
        'impact': 1.4,
        'source': 'Ekonomi Haberleri',
        'category': 'Türkiye'
    },
    {
        'title': 'Küresel belirsizlik güvenli liman talebini artırıyor',
        'summary': 'Jeopolitik gerilimler ve ekonomik belirsizlik altın ve gümüş gibi güvenli liman varlıklarına talebi artırıyor.',
        'sentiment': 'positive',
        'impact': 1.2,
        'source': 'Global Finans',
        'category': 'Jeopolitik'
    },
    {
        'title': 'Teknik analiz: Gümüş kritik direnç seviyesinde',
        'summary': 'Gümüş fiyatları önemli bir teknik direnç seviyesine yaklaşıyor. Bu seviyenin kırılması durumunda yükseliş hızlanabilir.',
        'sentiment': 'neutral',
        'impact': 0.5,
        'source': 'Teknik Analiz',
        'category': 'Teknik'
    },
    {
        'title': 'BIST gümüş ETF işlem hacmi rekor kırdı',
        'summary': 'GMSTR işlem hacmi son 3 ayın en yüksek seviyesine ulaştı. Kurumsal yatırımcıların ilgisi artıyor.',
        'sentiment': 'positive',
        'impact': 1.6,
        'source': 'BIST Haberleri',
        'category': 'BIST'
    },
    {
        'title': 'Çin sanayi üretimi beklentilerin altında kaldı',
        'summary': 'Çin\'in sanayi üretim verileri beklentilerin altında geldi. Bu durum gümüş talebini olumsuz etkileyebilir.',
        'sentiment': 'negative',
        'impact': -0.9,
        'source': 'Asya Piyasaları',
        'category': 'Küresel'
    },
    {
        'title': 'Gümüş-altın oranı tarihsel ortalamanın altında',
        'summary': 'Gümüş/altın oranı tarihsel ortalamanın altında seyrediyor. Bu durum gümüşün görece ucuz olduğuna işaret ediyor ve alım fırsatı sunabilir.',
        'sentiment': 'positive',
        'impact': 1.0,
        'source': 'Emtia Araştırma',
        'category': 'Analiz'
    },
    {
        'title': 'Elektrikli araç üretimi artışı gümüş talebini destekliyor',
        'summary': 'Küresel elektrikli araç satışlarındaki rekor artış, gümüş talebini olumlu etkiliyor. Her EV\'de ortalama 25-50 gram gümüş kullanılıyor.',
        'sentiment': 'positive',
        'impact': 1.3,
        'source': 'Enerji & Teknoloji',
        'category': 'Sanayi'
    },
    {
        'title': 'Merkez bankaları değerli metal alımlarını artırıyor',
        'summary': 'Küresel merkez bankaları altın ve gümüş rezervlerini artırıyor. Bu trend değerli metal fiyatlarını destekliyor.',
        'sentiment': 'positive',
        'impact': 1.5,
        'source': 'Merkez Bankası Haberleri',
        'category': 'Makro'
    },
    {
        'title': 'ABD istihdam verileri beklentilerin üzerinde geldi',
        'summary': 'Güçlü istihdam verileri Fed\'in faiz indirimi beklentilerini azaltıyor. Bu durum gümüş üzerinde kısa vadeli baskı oluşturabilir.',
        'sentiment': 'negative',
        'impact': -1.1,
        'source': 'ABD Ekonomi',
        'category': 'Makro'
    },
    {
        'title': 'Gümüş madenciliği üretimi düşüyor',
        'summary': 'Küresel gümüş madenciliği üretimi geçen yıla göre %5 geriledi. Arz kısıtlaması fiyatları destekleyebilir.',
        'sentiment': 'positive',
        'impact': 1.1,
        'source': 'Madencilik Haberleri',
        'category': 'Arz'
    },
    {
        'title': 'Türk lirası değer kaybı GMSTR\'yi cazip kılıyor',
        'summary': 'TL\'nin değer kaybetmesi, dolar bazlı gümüş yatırımlarını Türk yatırımcılar için daha cazip hale getiriyor. GMSTR alımları artıyor.',
        'sentiment': 'positive',
        'impact': 1.4,
        'source': 'Türkiye Piyasaları',
        'category': 'Türkiye'
    },
    {
        'title': 'Küresel resesyon endişeleri emtia piyasalarını baskılıyor',
        'summary': 'Artan resesyon endişeleri sanayi metalleri üzerinde baskı oluşturuyor. Gümüş hem sanayi hem değerli metal özelliği taşıdığından çift yönlü etkilenebilir.',
        'sentiment': 'negative',
        'impact': -0.8,
        'source': 'Küresel Ekonomi',
        'category': 'Makro'
    },
]


class NewsAnalyzer:
    """GMSTR için haber analizi ve AI tahmin sistemi."""
    
    def __init__(self):
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 300  # 5 dakika cache
    
    def fetch_news(self, use_simulation: bool = True, count: int = 12) -> List[Dict]:
        """Haberleri çek (gerçek veya simüle). count: kaç haber döndürülsün (10-20)."""
        now = datetime.now()
        count = max(10, min(20, count))  # 10-20 arası sınırla
        
        # Cache kontrolü
        if (self.cache_time and 
            (now - self.cache_time).seconds < self.cache_duration and
            self.cache.get('news')):
            return self.cache['news']
        
        news = []
        
        if not use_simulation:
            # Gerçek haber çekme denemesi
            news = self._try_fetch_real_news()
        
        # Simüle haberlerle tamamla (10-20 arası)
        sim_news = self._get_simulated_news(count=count)
        
        if not news:
            news = sim_news
        else:
            # Gerçek haberler + simüle haberlerle tamamla
            needed = max(0, count - len(news))
            news = news + sim_news[:needed]
        
        # Maksimum count kadar döndür
        news = news[:count]
        
        self.cache['news'] = news
        self.cache_time = now
        return news
    
    def _try_fetch_real_news(self) -> List[Dict]:
        """Gerçek haber kaynağından çekmeyi dene."""
        news = []
        try:
            # Yahoo Finance RSS
            url = 'https://finance.yahoo.com/rss/headline?s=SI%3DF'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('utf-8')
                
            # Basit RSS parse
            titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', content)
            descriptions = re.findall(r'<description><!\[CDATA\[(.*?)\]\]></description>', content)
            
            for i, title in enumerate(titles[1:8]):  # İlk 7 haber
                desc = descriptions[i+1] if i+1 < len(descriptions) else ''
                sentiment, impact = self._analyze_text_sentiment(title + ' ' + desc)
                news.append({
                    'title': title,
                    'summary': desc[:200] if desc else title,
                    'sentiment': sentiment,
                    'impact': impact,
                    'source': 'Yahoo Finance',
                    'category': 'Gerçek Haber',
                    'time': datetime.now().strftime('%H:%M'),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'is_real': True,
                })
        except Exception as e:
            pass
        
        return news
    
    def _get_simulated_news(self, count: int = 12) -> List[Dict]:
        """Gerçekçi simüle edilmiş haberler üret (10-20 arası)."""
        now = datetime.now()
        count = max(10, min(20, count))
        
        # Tüm şablonlardan seç (15 şablon var, count kadar seç)
        available = SIMULATED_NEWS_TEMPLATES.copy()
        if count <= len(available):
            selected = random.sample(available, count)
        else:
            # Yeterli şablon yoksa tekrar et
            selected = available + random.sample(available, count - len(available))
        
        news = []
        for i, template in enumerate(selected):
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            news_time = now - timedelta(hours=hours_ago, minutes=minutes_ago)
            
            # Küçük varyasyon ekle (aynı haber tekrar görünmesin)
            title = template['title']
            impact = template['impact'] * random.uniform(0.85, 1.15)  # ±15% varyasyon
            
            news.append({
                'title': title,
                'summary': template['summary'],
                'sentiment': template['sentiment'],
                'impact': round(impact, 2),
                'source': template['source'],
                'category': template.get('category', 'Genel'),
                'time': news_time.strftime('%H:%M'),
                'date': news_time.strftime('%Y-%m-%d'),
                'hours_ago': hours_ago,
                'is_real': False,
            })
        
        # Zamana göre sırala (en yeni önce)
        news.sort(key=lambda x: x['hours_ago'])
        return news
    
    def _analyze_text_sentiment(self, text: str) -> tuple:
        """Metin duygu analizi yap."""
        text_lower = text.lower()
        score = 0
        
        for keyword, weight in SILVER_KEYWORDS['positive'].items():
            if keyword in text_lower:
                score += weight
        
        for keyword, weight in SILVER_KEYWORDS['negative'].items():
            if keyword in text_lower:
                score += weight  # Negatif değerler zaten negatif
        
        if score > 0.5:
            return 'positive', min(score, 2.0)
        elif score < -0.5:
            return 'negative', max(score, -2.0)
        else:
            return 'neutral', score
    
    def get_news_sentiment_score(self, news: List[Dict]) -> Dict:
        """Haberlerin genel duygu skorunu hesapla."""
        if not news:
            return {
                'overall_score': 0,
                'direction': 'NÖTR',
                'confidence': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'signal': 'BEKLE',
                'signal_color': 'gray',
                'explanation': 'Haber verisi yok'
            }
        
        positive_news = [n for n in news if n['sentiment'] == 'positive']
        negative_news = [n for n in news if n['sentiment'] == 'negative']
        neutral_news = [n for n in news if n['sentiment'] == 'neutral']
        
        # Ağırlıklı skor
        total_impact = sum(n.get('impact', 0) for n in news)
        avg_impact = total_impact / len(news)
        
        # Son haberlere daha fazla ağırlık ver
        recent_weight = 0
        for i, n in enumerate(news[:5]):  # Son 5 haber
            recent_weight += n.get('impact', 0) * (1.5 - i * 0.1)
        
        combined_score = avg_impact * 0.4 + (recent_weight / 5) * 0.6
        
        # Kategori bazlı ağırlık
        macro_news = [n for n in news if n.get('category') in ['Makro', 'Türkiye', 'Döviz']]
        if macro_news:
            macro_score = sum(n.get('impact', 0) for n in macro_news) / len(macro_news)
            combined_score = combined_score * 0.7 + macro_score * 0.3
        
        # Yön ve güven
        if combined_score > 0.8:
            direction = 'YUKARI ↑'
            signal = 'AL'
            signal_color = 'green'
            confidence = min(combined_score / 2.0 * 100, 90)
        elif combined_score > 0.3:
            direction = 'HAFIF YUKARI ↑'
            signal = 'BEKLE/AL'
            signal_color = 'lightgreen'
            confidence = min(combined_score / 2.0 * 100, 70)
        elif combined_score < -0.8:
            direction = 'AŞAĞI ↓'
            signal = 'SAT'
            signal_color = 'red'
            confidence = min(abs(combined_score) / 2.0 * 100, 90)
        elif combined_score < -0.3:
            direction = 'HAFIF AŞAĞI ↓'
            signal = 'BEKLE/SAT'
            signal_color = 'orange'
            confidence = min(abs(combined_score) / 2.0 * 100, 70)
        else:
            direction = 'YATAY →'
            signal = 'BEKLE'
            signal_color = 'gray'
            confidence = 50
        
        # Açıklama
        explanations = []
        if len(positive_news) > len(negative_news):
            explanations.append(f'{len(positive_news)} pozitif haber gümüşü destekliyor')
        elif len(negative_news) > len(positive_news):
            explanations.append(f'{len(negative_news)} negatif haber baskı oluşturuyor')
        else:
            explanations.append(f'Karma sinyaller: {len(positive_news)} pozitif, {len(negative_news)} negatif')
        
        # En etkili haberi bul
        if news:
            most_impactful = max(news, key=lambda x: abs(x.get('impact', 0)))
            explanations.append(f'En etkili: "{most_impactful["title"][:50]}..."')
        
        return {
            'overall_score': round(combined_score, 3),
            'direction': direction,
            'confidence': round(confidence, 1),
            'positive_count': len(positive_news),
            'negative_count': len(negative_news),
            'neutral_count': len(neutral_news),
            'signal': signal,
            'signal_color': signal_color,
            'explanation': ' | '.join(explanations) if explanations else 'Karma sinyaller',
            'total_news': len(news),
        }
    
    def get_combined_prediction(self, ai_predictions: Dict, news_sentiment: Dict) -> Dict:
        """AI tahminleri + haber duygusu = kombine tahmin."""
        
        # AI sinyali skoru
        ai_score = 0
        ai_confidence = 0
        
        if ai_predictions:
            # 1d tahminini ana sinyal olarak kullan
            pred_1d = ai_predictions.get('1d_daily', {})
            if pred_1d:
                prob_up = pred_1d.get('prob_up', 0.5)
                ai_score = (prob_up - 0.5) * 2  # -1 ile 1 arası
                ai_confidence = pred_1d.get('confidence', 0.5)
        
        # Haber skoru
        news_score = news_sentiment.get('overall_score', 0)
        news_confidence = news_sentiment.get('confidence', 50) / 100
        
        # Kombine skor (AI %60, Haber %40)
        combined_score = ai_score * 0.60 + news_score * 0.40
        combined_confidence = (ai_confidence * 0.60 + news_confidence * 0.40) * 100
        
        # Final karar
        if combined_score > 0.3 and combined_confidence > 55:
            final_signal = 'GÜÇLÜ AL 🟢'
            signal_class = 'strong-buy'
            action = 'Alım yapın'
        elif combined_score > 0.1:
            final_signal = 'AL 🟡'
            signal_class = 'buy'
            action = 'Küçük pozisyon açın'
        elif combined_score < -0.3 and combined_confidence > 55:
            final_signal = 'GÜÇLÜ SAT 🔴'
            signal_class = 'strong-sell'
            action = 'Pozisyon kapatın'
        elif combined_score < -0.1:
            final_signal = 'SAT 🟠'
            signal_class = 'sell'
            action = 'Kısmi satış yapın'
        else:
            final_signal = 'BEKLE ⚪'
            signal_class = 'hold'
            action = 'Pozisyon açmayın'
        
        # AI ve haber uyumu
        ai_dir = 'up' if ai_score > 0 else 'down'
        news_dir = 'up' if news_score > 0 else 'down'
        alignment = 'UYUMLU ✅' if ai_dir == news_dir else 'ÇAKIŞIYOR ⚠️'
        
        return {
            'final_signal': final_signal,
            'signal_class': signal_class,
            'action': action,
            'combined_score': round(combined_score, 3),
            'combined_confidence': round(combined_confidence, 1),
            'ai_contribution': round(ai_score * 0.60, 3),
            'news_contribution': round(news_score * 0.40, 3),
            'ai_news_alignment': alignment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    def get_silver_market_context(self) -> Dict:
        """Gümüş piyasası bağlamı ve makro faktörler."""
        now = datetime.now()
        hour = now.hour
        
        # Piyasa saati durumu
        if 10 <= hour <= 18:
            market_status = 'AÇIK 🟢'
            market_note = 'BIST aktif işlem saatlerinde'
        elif 18 < hour <= 22:
            market_status = 'KAPALI 🔴'
            market_note = 'BIST kapandı, yarın açılış bekleniyor'
        else:
            market_status = 'KAPALI 🔴'
            market_note = 'Piyasa kapalı'
        
        # Haftanın günü
        weekday = now.weekday()
        if weekday >= 5:
            market_status = 'HAFTA SONU 🔵'
            market_note = 'Hafta sonu - Pazartesi açılış bekleniyor'
        
        # Mevsimsel faktörler
        month = now.month
        seasonal_note = ''
        if month in [1, 2]:
            seasonal_note = 'Ocak-Şubat: Tarihsel olarak gümüş için güçlü dönem'
        elif month in [3, 4]:
            seasonal_note = 'Mart-Nisan: Sanayi talebi artış dönemi'
        elif month in [7, 8]:
            seasonal_note = 'Yaz dönemi: Genellikle düşük hacim'
        elif month in [9, 10]:
            seasonal_note = 'Eylül-Ekim: Tarihsel olarak volatil dönem'
        elif month in [11, 12]:
            seasonal_note = 'Kasım-Aralık: Yıl sonu pozisyon kapatma dönemi'
        
        return {
            'market_status': market_status,
            'market_note': market_note,
            'seasonal_note': seasonal_note,
            'current_time': now.strftime('%H:%M'),
            'current_date': now.strftime('%d.%m.%Y'),
            'weekday': ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 
                       'Cuma', 'Cumartesi', 'Pazar'][weekday],
        }


# Global instance
_analyzer = None

def get_analyzer() -> NewsAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = NewsAnalyzer()
    return _analyzer
