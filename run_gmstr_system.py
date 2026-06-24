"""
GMSTR Tahmin Sistemi Başlatıcı Script
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Gerekli paketleri kur"""
    print("Gerekli paketler kuruluyor...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Paketler başarıyla kuruldu!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Paket kurulum hatası: {e}")
        return False

def create_directories():
    """Gerekli dizinleri oluştur"""
    directories = ["templates", "static"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("Dizinler oluşturuldu!")

def check_dependencies():
    """Bağımlılıkları kontrol et"""
    required_packages = [
        "flask", "yfinance", "pandas", "numpy", "scikit-learn", 
        "requests", "schedule", "joblib"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Eksik paketler: {missing_packages}")
        return False
    
    print("Tüm paketler mevcut!")
    return True

def main():
    """Ana başlatıcı fonksiyon"""
    print("GMSTR Tahmin Sistemi Başlatılıyor...")
    print("=" * 50)
    
    # Dizinleri oluştur
    create_directories()
    
    # Bağımlılıkları kontrol et
    if not check_dependencies():
        print("Eksik paketler kuruluyor...")
        if not install_requirements():
            print("Paket kurulumu başarısız! Lütfen manuel kurun.")
            return
    
    # Sistemi başlat
    print("\nGMSTR Tahmin Sistemi başlatılıyor...")
    print("Web arayüzü: http://localhost:5000")
    print("Çıkmak için: Ctrl+C")
    print("=" * 50)
    
    try:
        import gmstr_prediction_system
    except ImportError as e:
        print(f"Modül import hatası: {e}")
        return
    
    # Sistemi çalıştır
    try:
        from gmstr_prediction_system import app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nSistem durduruldu.")
    except Exception as e:
        print(f"Sistem çalıştırma hatası: {e}")

if __name__ == "__main__":
    main()
