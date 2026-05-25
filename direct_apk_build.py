"""
Doğrudan APK Build Script
"""

import subprocess
import sys

def main():
    print("Crypto Monitor APK Build Script")
    print("=" * 40)
    
    try:
        # Doğrudan APK oluştur
        print("APK oluşturuluyor...")
        subprocess.run(["buildozer", "android", "debug"], check=True)
        
        print("\nAPK başarıyla oluşturuldu!")
        print("APK dosyası: bin/cryptomonitor-0.1-debug.apk")
        
    except subprocess.CalledProcessError as e:
        print(f"Buildozer hatası: {e}")
        print("\nAndroid geliştirme ortamı eksik olabilir.")
        print("Önerilen çözümler:")
        print("1. Python-for-Android kurulumu:")
        print("   pip install python-for-android")
        print("2. Android Studio kurun")
        print("3. Java JDK 11+ kurun")
        print("4. Android SDK kurun")
        print("5. NDK kurun")
        print("6. ANDROID_HOME environment variable ayarlayın")
    except Exception as e:
        print(f"Beklenmedik hata: {e}")

if __name__ == "__main__":
    main()
