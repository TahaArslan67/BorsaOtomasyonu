#!/usr/bin/env python3
"""
GMSTR Desktop App - PyInstaller Build Script
.exe + gerekli dosyaları dist/GMSTR_Monitor/ altına paketler.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).parent
    os.chdir(root)

    # PyInstaller kurulu mu?
    try:
        import PyInstaller
    except ImportError:
        print("[Build] PyInstaller kurulu değil. Kurulum yapılıyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "-q"])

    spec_name = "gmstr_desktop_app"
    dist_dir = root / "dist" / "GMSTR_Monitor"

    # Önceki build'i temizle (kilitli dosyalari atla)
    if (root / "build").exists():
        shutil.rmtree(root / "build", ignore_errors=True)
    if dist_dir.exists():
        shutil.rmtree(dist_dir, ignore_errors=True)

    # Gizli importlar
    hidden = [
        "sklearn",
        "sklearn.ensemble",
        "sklearn.linear_model",
        "sklearn.preprocessing",
        "sklearn.calibration",
        "sklearn.neural_network",
        "sklearn.model_selection",
        "sklearn.metrics",
        "sklearn.feature_selection",
        "xgboost",
        "lightgbm",
        "numpy",
        "pandas",
        "ta",
    ]
    hidden_args = []
    for h in hidden:
        hidden_args.extend(["--hidden-import", h])

    # Add-data: "src;dest" formatı (Windows'ta noktalı virgül, Linux'ta iki nokta)
    sep = ";" if os.name == "nt" else ":"
    add_data = [
        f"gmstr_system{sep}gmstr_system",
        f"claude{sep}claude",
        f"gmstr_models{sep}gmstr_models",
    ]
    data_args = []
    for d in add_data:
        data_args.extend(["--add-data", d])

    # XGBoost ve LightGBM native DLL'lerini ekle
    try:
        import xgboost
        xgb_path = Path(xgboost.__file__).parent
        xgb_dll = xgb_path / "lib" / "xgboost.dll"
        if xgb_dll.exists():
            data_args.extend(["--add-binary", f"{xgb_dll}{sep}xgboost/lib"])
            print(f"[Build] XGBoost DLL eklendi: {xgb_dll}")
        # VERSION dosyasini ekle (xgboost init tarafindan okunur)
        xgb_version = xgb_path / "VERSION"
        if xgb_version.exists():
            data_args.extend(["--add-data", f"{xgb_version}{sep}xgboost"])
            print(f"[Build] XGBoost VERSION eklendi: {xgb_version}")
    except Exception as e:
        print(f"[Build] XGBoost dosyaları bulunamadı: {e}")

    try:
        import lightgbm
        lgb_dll = Path(lightgbm.__file__).parent / "lib_lightgbm.dll"
        if lgb_dll.exists():
            data_args.extend(["--add-binary", f"{lgb_dll}{sep}lightgbm"])
            print(f"[Build] LightGBM DLL eklendi: {lgb_dll}")
    except Exception as e:
        print(f"[Build] LightGBM DLL bulunamadı: {e}")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "GMSTR_Monitor",
        "--onedir",
        "--windowed",
        "--clean",
        *hidden_args,
        *data_args,
        f"{spec_name}.py",
    ]

    print("[Build] PyInstaller çalıştırılıyor...")
    print(" ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("[Build] HATA! PyInstaller başarısız.")
        sys.exit(1)

    # Kısayol oluştur (Windows)
    if os.name == "nt":
        try:
            import winshell
            from win32com.client import Dispatch
            exe_path = dist_dir / "GMSTR_Monitor.exe"
            desktop = Path(winshell.desktop())
            shortcut = desktop / "GMSTR Monitor.lnk"
            shell = Dispatch("WScript.Shell")
            sc = shell.CreateShortcut(str(shortcut))
            sc.TargetPath = str(exe_path)
            sc.WorkingDirectory = str(dist_dir)
            sc.IconLocation = str(exe_path)
            sc.save()
            print(f"[Build] Masaüstü kısayolu oluşturuldu: {shortcut}")
        except Exception as e:
            print(f"[Build] Kısayol oluşturulamadı ({e}), manuel çalıştırabilirsiniz.")

    print(f"\n[Build] BAŞARILI!")
    print(f"   Çalıştırılabilir: {dist_dir / 'GMSTR_Monitor.exe'}")
    print(f"   Bu dizini ZIP'leyip başka bilgisayarda çalıştırabilirsiniz.")


if __name__ == "__main__":
    main()
