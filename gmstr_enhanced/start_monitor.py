# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
GMSTR Gelismis Monitör - Baslatma Scripti
==========================================
Kullanim: python gmstr_enhanced/start_monitor.py
Tarayicida: http://localhost:5050
"""
import sys
import os
import subprocess
from pathlib import Path

# Windows encoding düzeltmesi
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def check_dependencies():
    """Gerekli paketleri kontrol et."""
    missing = []
    
    packages = {
        'flask': 'flask',
        'flask_cors': 'flask-cors',
    }
    
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n[UYARI] Eksik paketler: {', '.join(missing)}")
        print(f"Yuklemek icin: pip install {' '.join(missing)}\n")
        
        answer = input("Otomatik yuklensin mi? (e/h): ").strip().lower()
        if answer == 'e':
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing, check=True)
            print("[OK] Paketler yuklendi!\n")
        else:
            print("[HATA] Paketler yuklenmedi. Manuel yukleyin.")
            sys.exit(1)


def main():
    print("\n" + "="*60)
    print("  GMSTR GELISMIS AI MONITÖR v2.0")
    print("="*60)
    print()
    
    # Bağımlılık kontrolü
    check_dependencies()
    
    # Veritabanını başlat
    try:
        from gmstr_enhanced.trade_db import init_db
        init_db()
        print("[OK] Veritabani hazir")
    except Exception as e:
        print(f"[UYARI] Veritabani hatasi: {e}")
    
    # Model dosyalarını kontrol et
    model_dir = ROOT / 'gmstr_models'
    model_files = list(model_dir.glob('simple_*.pkl')) if model_dir.exists() else []
    pred_file = model_dir / 'latest_predictions.json'
    
    if model_files:
        print(f"[OK] {len(model_files)} model dosyasi bulundu")
    else:
        print("[UYARI] Model dosyasi bulunamadi. Web arayuzunden 'Modeli Egit' butonuna tiklayin.")
    
    if pred_file.exists():
        print("[OK] Tahmin dosyasi mevcut")
    else:
        print("[UYARI] Tahmin dosyasi yok. Model egitimi gerekli.")
    
    print()
    print("[*] Sunucu baslatiliyor...")
    print("[*] Tarayicida acin: http://localhost:5050")
    print("[*] Durdurmak icin: Ctrl+C")
    print()
    
    # Flask uygulamasını başlat
    os.chdir(str(ROOT))
    
    from gmstr_enhanced.app import app, init_db as app_init_db
    app_init_db()
    
    try:
        app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n[OK] Monitör durduruldu.")


if __name__ == '__main__':
    main()
