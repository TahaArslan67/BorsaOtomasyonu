"""
GMSTR Alım/Satım Veritabanı Yöneticisi
SQLite tabanlı işlem kaydı ve analiz sistemi
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


DB_PATH = Path(__file__).parent / "trades.db"


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Veritabanı tablolarını oluştur."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_type TEXT NOT NULL,          -- 'BUY' veya 'SELL'
            trade_date TEXT NOT NULL,           -- YYYY-MM-DD
            trade_time TEXT NOT NULL,           -- HH:MM
            price REAL NOT NULL,                -- İşlem fiyatı
            quantity REAL NOT NULL,             -- Lot/adet
            total_value REAL NOT NULL,          -- Toplam tutar
            commission REAL DEFAULT 0,          -- Komisyon
            notes TEXT DEFAULT '',              -- Notlar
            bot_signal TEXT DEFAULT '',         -- O anki bot sinyali
            created_at TEXT NOT NULL
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            open_trade_id INTEGER,
            close_trade_id INTEGER,
            open_date TEXT,
            close_date TEXT,
            open_price REAL,
            close_price REAL,
            quantity REAL,
            profit_loss REAL,
            profit_loss_pct REAL,
            status TEXT DEFAULT 'OPEN',        -- 'OPEN' veya 'CLOSED'
            FOREIGN KEY (open_trade_id) REFERENCES trades(id),
            FOREIGN KEY (close_trade_id) REFERENCES trades(id)
        )
    """)
    
    conn.commit()
    conn.close()


def add_trade(trade_type: str, trade_date: str, trade_time: str,
              price: float, quantity: float, commission: float = 0,
              notes: str = '', bot_signal: str = '') -> int:
    """Yeni işlem ekle."""
    conn = get_connection()
    c = conn.cursor()
    
    total_value = price * quantity
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("""
        INSERT INTO trades (trade_type, trade_date, trade_time, price, quantity,
                           total_value, commission, notes, bot_signal, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trade_type.upper(), trade_date, trade_time, price, quantity,
          total_value, commission, notes, bot_signal, created_at))
    
    trade_id = c.lastrowid
    conn.commit()
    
    # Pozisyon güncelle
    _update_positions(conn, trade_id, trade_type.upper(), trade_date, price, quantity)
    
    conn.commit()
    conn.close()
    return trade_id


def _update_positions(conn, trade_id: int, trade_type: str, trade_date: str,
                      price: float, quantity: float):
    """Pozisyon tablosunu güncelle (FIFO mantığı)."""
    c = conn.cursor()
    
    if trade_type == 'BUY':
        # Yeni açık pozisyon oluştur
        c.execute("""
            INSERT INTO positions (open_trade_id, open_date, open_price, quantity, status)
            VALUES (?, ?, ?, ?, 'OPEN')
        """, (trade_id, trade_date, price, quantity))
    
    elif trade_type == 'SELL':
        # Açık pozisyonları FIFO ile kapat
        remaining_qty = quantity
        c.execute("""
            SELECT id, open_price, quantity FROM positions
            WHERE status = 'OPEN'
            ORDER BY open_date ASC
        """)
        open_positions = c.fetchall()
        
        for pos in open_positions:
            if remaining_qty <= 0:
                break
            
            pos_id = pos['id']
            open_price = pos['open_price']
            pos_qty = pos['quantity']
            
            close_qty = min(remaining_qty, pos_qty)
            pl = (price - open_price) * close_qty
            pl_pct = ((price - open_price) / open_price) * 100
            
            if close_qty >= pos_qty:
                # Pozisyonu tamamen kapat
                c.execute("""
                    UPDATE positions SET
                        close_trade_id = ?,
                        close_date = ?,
                        close_price = ?,
                        profit_loss = ?,
                        profit_loss_pct = ?,
                        status = 'CLOSED'
                    WHERE id = ?
                """, (trade_id, trade_date, price, pl, pl_pct, pos_id))
            else:
                # Kısmi kapat - orijinali güncelle, yeni açık pozisyon oluştur
                c.execute("""
                    UPDATE positions SET quantity = ? WHERE id = ?
                """, (pos_qty - close_qty, pos_id))
                
                pl_partial = (price - open_price) * close_qty
                pl_pct_partial = ((price - open_price) / open_price) * 100
                c.execute("""
                    INSERT INTO positions (open_trade_id, close_trade_id, open_date, close_date,
                                         open_price, close_price, quantity, profit_loss,
                                         profit_loss_pct, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED')
                """, (pos['open_trade_id'] if 'open_trade_id' in pos.keys() else pos_id,
                      trade_id, pos['open_date'] if 'open_date' in pos.keys() else trade_date,
                      trade_date, open_price, price, close_qty, pl_partial, pl_pct_partial))
            
            remaining_qty -= close_qty


def get_all_trades() -> List[Dict]:
    """Tüm işlemleri getir."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM trades ORDER BY trade_date DESC, trade_time DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_trade_by_id(trade_id: int) -> Optional[Dict]:
    """ID'ye göre işlem getir."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_trade(trade_id: int) -> bool:
    """İşlemi sil."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    affected = c.rowcount
    # İlgili pozisyonları da temizle
    c.execute("DELETE FROM positions WHERE open_trade_id = ? OR close_trade_id = ?",
              (trade_id, trade_id))
    conn.commit()
    conn.close()
    return affected > 0


def get_analysis() -> Dict:
    """Kapsamlı kar/zarar analizi."""
    conn = get_connection()
    c = conn.cursor()
    
    # Tüm işlemler
    c.execute("SELECT * FROM trades ORDER BY trade_date, trade_time")
    all_trades = [dict(r) for r in c.fetchall()]
    
    # Kapalı pozisyonlar
    c.execute("""
        SELECT p.*, 
               t_open.trade_date as open_date_full,
               t_close.trade_date as close_date_full
        FROM positions p
        LEFT JOIN trades t_open ON p.open_trade_id = t_open.id
        LEFT JOIN trades t_close ON p.close_trade_id = t_close.id
        WHERE p.status = 'CLOSED'
        ORDER BY p.close_date DESC
    """)
    closed_positions = [dict(r) for r in c.fetchall()]
    
    # Açık pozisyonlar
    c.execute("""
        SELECT p.*, t_open.trade_date as open_date_full
        FROM positions p
        LEFT JOIN trades t_open ON p.open_trade_id = t_open.id
        WHERE p.status = 'OPEN'
    """)
    open_positions = [dict(r) for r in c.fetchall()]
    
    conn.close()
    
    # Hesaplamalar
    total_buy = sum(t['total_value'] for t in all_trades if t['trade_type'] == 'BUY')
    total_sell = sum(t['total_value'] for t in all_trades if t['trade_type'] == 'SELL')
    total_commission = sum(t['commission'] for t in all_trades)
    
    realized_pl = sum(p['profit_loss'] for p in closed_positions if p['profit_loss'])
    
    # Aylık performans
    monthly_pl = {}
    for pos in closed_positions:
        if pos['close_date']:
            month_key = pos['close_date'][:7]  # YYYY-MM
            monthly_pl[month_key] = monthly_pl.get(month_key, 0) + (pos['profit_loss'] or 0)
    
    # Kazanan/kaybeden işlemler
    winning_trades = [p for p in closed_positions if (p['profit_loss'] or 0) > 0]
    losing_trades = [p for p in closed_positions if (p['profit_loss'] or 0) < 0]
    
    win_rate = len(winning_trades) / len(closed_positions) * 100 if closed_positions else 0
    
    avg_win = sum(p['profit_loss'] for p in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(p['profit_loss'] for p in losing_trades) / len(losing_trades) if losing_trades else 0
    
    profit_factor = abs(sum(p['profit_loss'] for p in winning_trades) / 
                       sum(p['profit_loss'] for p in losing_trades)) if losing_trades else float('inf')
    
    # En iyi/kötü işlem
    best_trade = max(closed_positions, key=lambda x: x['profit_loss'] or 0) if closed_positions else None
    worst_trade = min(closed_positions, key=lambda x: x['profit_loss'] or 0) if closed_positions else None
    
    # Ortalama tutma süresi
    holding_days = []
    for pos in closed_positions:
        if pos['open_date'] and pos['close_date']:
            try:
                open_dt = datetime.strptime(pos['open_date'], "%Y-%m-%d")
                close_dt = datetime.strptime(pos['close_date'], "%Y-%m-%d")
                holding_days.append((close_dt - open_dt).days)
            except:
                pass
    avg_holding = sum(holding_days) / len(holding_days) if holding_days else 0
    
    # Öneriler
    recommendations = _generate_recommendations(
        win_rate, realized_pl, profit_factor, avg_win, avg_loss,
        monthly_pl, open_positions
    )
    
    return {
        'summary': {
            'total_trades': len(all_trades),
            'buy_trades': len([t for t in all_trades if t['trade_type'] == 'BUY']),
            'sell_trades': len([t for t in all_trades if t['trade_type'] == 'SELL']),
            'total_buy_value': round(total_buy, 2),
            'total_sell_value': round(total_sell, 2),
            'total_commission': round(total_commission, 2),
            'realized_pl': round(realized_pl, 2),
            'realized_pl_net': round(realized_pl - total_commission, 2),
        },
        'performance': {
            'win_rate': round(win_rate, 2),
            'total_closed_positions': len(closed_positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999,
            'avg_holding_days': round(avg_holding, 1),
            'best_trade_pl': round(best_trade['profit_loss'], 2) if best_trade else 0,
            'worst_trade_pl': round(worst_trade['profit_loss'], 2) if worst_trade else 0,
        },
        'monthly_pl': {k: round(v, 2) for k, v in sorted(monthly_pl.items())},
        'open_positions': open_positions,
        'closed_positions': closed_positions[:20],  # Son 20
        'all_trades': all_trades,
        'recommendations': recommendations,
    }


def _generate_recommendations(win_rate, realized_pl, profit_factor,
                               avg_win, avg_loss, monthly_pl, open_positions) -> List[Dict]:
    """Akıllı öneriler üret."""
    recs = []
    
    if win_rate < 40:
        recs.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': 'Düşük Kazanma Oranı',
            'text': f'Kazanma oranınız %{win_rate:.1f}. Bot sinyallerini daha dikkatli takip edin, '
                    f'sadece güçlü sinyal (>%65 güven) olan işlemlere girin.'
        })
    elif win_rate >= 60:
        recs.append({
            'type': 'success',
            'icon': '✅',
            'title': 'İyi Kazanma Oranı',
            'text': f'Kazanma oranınız %{win_rate:.1f} - strateji çalışıyor. Mevcut yaklaşımı sürdürün.'
        })
    
    if profit_factor < 1.0 and profit_factor != 999:
        recs.append({
            'type': 'danger',
            'icon': '🔴',
            'title': 'Negatif Profit Factor',
            'text': f'Profit factor {profit_factor:.2f} < 1.0. Kayıplar kazançları aşıyor. '
                    f'Stop-loss kullanımını artırın.'
        })
    elif profit_factor > 2.0:
        recs.append({
            'type': 'success',
            'icon': '🟢',
            'title': 'Mükemmel Risk/Ödül',
            'text': f'Profit factor {profit_factor:.2f} - kazançlar kayıpların 2 katından fazla!'
        })
    
    if avg_loss != 0 and abs(avg_win / avg_loss) < 1.5:
        recs.append({
            'type': 'warning',
            'icon': '📊',
            'title': 'Risk/Ödül Oranı Düşük',
            'text': f'Ortalama kazanç/kayıp oranı {abs(avg_win/avg_loss):.2f}. '
                    f'Hedef en az 1.5:1 olmalı. Kar hedeflerinizi yükseltin.'
        })
    
    # Aylık performans analizi
    if monthly_pl:
        months = list(monthly_pl.values())
        profitable_months = sum(1 for m in months if m > 0)
        monthly_rate = profitable_months / len(months) * 100
        
        if monthly_rate >= 75:
            recs.append({
                'type': 'success',
                'icon': '📈',
                'title': 'Tutarlı Aylık Performans',
                'text': f'Ayların %{monthly_rate:.0f}\'i karlı. Strateji tutarlı çalışıyor.'
            })
        
        avg_monthly = sum(months) / len(months)
        if avg_monthly > 0:
            recs.append({
                'type': 'info',
                'icon': '💰',
                'title': 'Aylık Ortalama Getiri',
                'text': f'Aylık ortalama kar: {avg_monthly:.2f} TL. '
                        f'{"Hedef %15 üzerinde! 🎯" if avg_monthly > 0 else "Hedef %15 için daha fazla işlem gerekli."}'
            })
    
    if open_positions:
        total_open_qty = sum(p['quantity'] for p in open_positions)
        recs.append({
            'type': 'info',
            'icon': '📌',
            'title': 'Açık Pozisyon',
            'text': f'{len(open_positions)} açık pozisyon var, toplam {total_open_qty:.2f} lot. '
                    f'Bot sinyallerini takip ederek çıkış zamanlaması yapın.'
        })
    
    if not recs:
        recs.append({
            'type': 'info',
            'icon': '📋',
            'title': 'Veri Bekleniyor',
            'text': 'Analiz için işlem kayıtlarınızı ekleyin.'
        })
    
    return recs


# Başlangıçta DB'yi oluştur
init_db()
