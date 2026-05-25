"""
APK Build Script
Crypto Monitor için APK oluşturma
"""

import os
import subprocess
import sys

def install_buildozer():
    """Buildozer kur"""
    try:
        print("Buildozer kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "buildozer"])
        print("Buildozer kurulumu tamamlandı.")
    except subprocess.CalledProcessError as e:
        print(f"Buildozer kurulum hatası: {e}")
        return False
    return True

def build_apk():
    """APK oluştur"""
    try:
        print("APK oluşturuluyor...")
        
        # Önce buildozer init çalıştır
        print("Buildozer başlatılıyor...")
        init_cmd = ["buildozer", "init"]
        subprocess.run(init_cmd, check=True)
        
        # APK oluştur
        print("APK derleniyor...")
        cmd = ["buildozer", "android", "debug"]
        
        # Komutu çalıştır
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Çıktıyı göster
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
        if process.returncode == 0:
            print("APK başarıyla oluşturuldu!")
            print("APK dosyası: bin/cryptomonitor-0.1-debug.apk")
        else:
            print("APK oluşturulamadı.")
            
    except Exception as e:
        print(f"APK oluşturma hatası: {e}")

def main():
    print("Crypto Monitor APK Build Script")
    print("=" * 40)
    
    # 1. Buildozer kur
    if not install_buildozer():
        return
    
    print("\n" + "=" * 40)
    
    # 2. APK oluştur
    build_apk()
    
    print("\n" + "=" * 40)
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    main()
