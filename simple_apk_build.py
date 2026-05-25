"""
Basit APK Build Script
"""

import os
import subprocess
import sys

def main():
    print("Crypto Monitor APK Build Script")
    print("=" * 40)
    
    try:
        # 1. Buildozer başlat
        print("1. Buildozer başlatılıyor...")
        subprocess.run(["buildozer", "init"], check=True)
        print("Buildozer başlatıldı.")
        
        # 2. APK oluştur
        print("\n2. APK oluşturuluyor...")
        subprocess.run(["buildozer", "android", "debug"], check=True)
        
        print("\nAPK başarıyla oluşturuldu!")
        print("APK dosyası: bin/cryptomonitor-0.1-debug.apk")
        
    except subprocess.CalledProcessError as e:
        print(f"Hata: {e}")
        print("\nAlternatif çözüm:")
        print("1. Android Studio kurun")
        print("2. Java JDK kurun")
        print("3. Android SDK kurun")
        print("4. NDK kurun")
        print("5. Tekrar deneyin")
    except Exception as e:
        print(f"Beklenmedik hata: {e}")

if __name__ == "__main__":
    main()
