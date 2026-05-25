# APK Build Talimatlari

## Onemli Bilgi

**Buildozer (Python-for-Android) Windows'ta calismaz.** Sadece Linux/macOS uzerinde Android APK uretebilir.

Senin sisteminizde:
- WSL (Windows Subsystem for Linux) kurulu degil
- Docker kurulu degil
- Buildozer Windows'ta "Unknown command/target android" hatasi veriyor

Bu yuzden **3 farkli APK build yontemi** hazirladim:

---

## Yontem 1: GitHub Actions (EN KOLAY - Tavsiye Edilen)

GitHub sunucularinda otomatik build alir, senin PC'n calismaz.

### Adim 1: GitHub Repo Olustur
1. https://github.com/new adresine git
2. Repo adi: `borsabot` yaz
3. Public sec (Actions ucretsiz)
4. "Create repository" butonuna tikla

### Adim 2: Kodu Push Et
```powershell
cd D:\otonomBorsa
git remote add origin https://github.com/KULLANICI_ADIN/borsabot.git
git branch -M main
git push -u origin main
```

### Adim 3: Actions'tan Build Baslat
1. GitHub repo sayfasinda **Actions** sekmesine tikla
2. Sol menude **"Build Android APK"** workflow'unu bul
3. Uzerine tikla, sag ustte **"Run workflow"** butonuna bas
4. Yesil tik (✓) gelene kadar bekle (yaklasik 15-30 dk)

### Adim 4: APK'yi Indir
1. Build tamamlaninca Actions sekmesine geri don
2. En son calisan workflow'a tikla
3. Sayfa altinda **Artifacts** bolumunde `BorsaBot-APK` gorunecek
4. Indir ve telefona yukle

---

## Yontem 2: WSL Kurulumu (Yerel Build)

Windows uzerinde Linux calistirarak build alir.

### Adim 1: WSL Kur
```powershell
wsl --install
```
(Bilgisayar yeniden baslayacak)

### Adim 2: Ubuntu'ya Gerekli Paketleri Kur
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev automake
pip3 install buildozer cython
```

### Adim 3: Build Al
```bash
cd /mnt/d/otonomBorsa
buildozer android debug
```

APK `bin/` klasorunde olusacak.

---

## Yontem 3: Docker ile Build

Docker Desktop kurulu olmali.

```bash
docker run -it --rm -v "${PWD}:/home/user/app" kivy/buildozer bash
cd app
buildozer android debug
```

---

## Hazirlanan Dosyalar

- `buildozer.spec` -> Build konfigurasyonu (optimize edildi)
- `.github/workflows/build_apk.yml` -> GitHub Actions workflow
- `.gitignore` -> Gereksiz dosyalari haric tutar

## Not

GitHub Actions yontemi en kolayi. Kodu push ettikten sonra tek yapman gereken "Run workflow" butonuna basmak. Build GitHub'in Ubuntu sunucularinda calisir, senin PC'n etkilenmez.
