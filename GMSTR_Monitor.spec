# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gmstr_desktop_app.py'],
    pathex=[],
    binaries=[('C:\\Users\\arsla\\AppData\\Roaming\\Python\\Python313\\site-packages\\xgboost\\lib\\xgboost.dll', 'xgboost/lib')],
    datas=[('gmstr_system', 'gmstr_system'), ('claude', 'claude'), ('gmstr_models', 'gmstr_models'), ('C:\\Users\\arsla\\AppData\\Roaming\\Python\\Python313\\site-packages\\xgboost\\VERSION', 'xgboost')],
    hiddenimports=['sklearn', 'sklearn.ensemble', 'sklearn.linear_model', 'sklearn.preprocessing', 'sklearn.calibration', 'sklearn.neural_network', 'sklearn.model_selection', 'sklearn.metrics', 'sklearn.feature_selection', 'xgboost', 'lightgbm', 'numpy', 'pandas', 'ta'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GMSTR_Monitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GMSTR_Monitor',
)
