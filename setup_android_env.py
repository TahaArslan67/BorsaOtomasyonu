"""
Android Geliştirme Ortamı Kurulum Rehberi
"""

import os
import subprocess

def main():
    print("Android Geliştirme Ortamı Kurulum Rehberi")
    print("=" * 50)
    
    print("\n1. JDK 11 Kurulumu:")
    print("   a) https://adoptium.net/temurin/releases/?version=11")
    print("   b) Windows x64 MSI dosyasını indir")
    print("   c) Kurulumu tamamla")
    
    print("\n2. JDK Kurulum Yeri (Önerilen):")
    print("   C:\\Program Files\\Eclipse Adoptium\\jdk-11.0.x-hotspot\\")
    
    print("\n3. Android Studio Kurulumu:")
    print("   a) https://developer.android.com/studio")
    print("   b) Windows indir")
    print("   c) Kurulumda 'Android SDK' seçeneğini işaretle")
    
    print("\n4. Android SDK Kurulum Yeri:")
    print("   C:\\Users\\KullanıcıAdınız\\AppData\\Local\\Android\\Sdk")
    
    print("\n5. NDK Kurulumu:")
    print("   a) Android Studio aç")
    print("   b) Tools -> SDK Manager")
    print("   c) SDK Tools sekmesi")
    print("   d) 'NDK (Side by side)' ve 'CMake' işaretle")
    print("   e) Apply")
    
    print("\n6. Environment Variables (Ortam Değişkenleri):")
    print("   a) Win + R -> sysdm.cpl")
    print("   b) Advanced -> Environment Variables")
    print("   c) System variables -> New:")
    print("      - Variable name: JAVA_HOME")
    print("      - Variable value: C:\\Program Files\\Eclipse Adoptium\\jdk-11.0.x-hotspot")
    print("   d) System variables -> New:")
    print("      - Variable name: ANDROID_HOME")
    print("      - Variable value: C:\\Users\\KullanıcıAdınız\\AppData\\Local\\Android\\Sdk")
    print("   e) Path'e ekle:")
    print("      - %JAVA_HOME%\\bin")
    print("      - %ANDROID_HOME%\\tools")
    print("      - %ANDROID_HOME%\\platform-tools")
    
    print("\n7. Kontrol Et:")
    print("   Komut satırında çalıştır:")
    print("   java -version")
    print("   echo %JAVA_HOME%")
    print("   echo %ANDROID_HOME%")
    
    print("\n8. Son Adım:")
    print("   Komut satırını kapatıp yeniden aç")
    print("   python direct_apk_build.py")

if __name__ == "__main__":
    main()
