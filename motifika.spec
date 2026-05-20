# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — MOTİFİKA'yı tek klasörlük (onedir) bir çalıştırılabilire dondurur.

onedir (onefile değil): Raspberry Pi'de açılış hızlıdır; her çalıştırmada /tmp'e
açma maliyeti yoktur. Çıktı: dist/motifika/  (motifika ikilisi + _internal/).

Yapı (CI veya yerel): pip install -r requirements-runtime.txt pyinstaller
                      pyinstaller motifika.spec
"""
from PyInstaller.utils.hooks import collect_all

# OpenCV (opencv-python, GUI/highgui'li tam sürüm) tüm parçalarıyla toplanır:
# paylaşımlı kütüphaneler + Qt eklentileri (cv2.imshow için) + config dosyaları.
cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")

# Çalışma zamanında KULLANILMAYAN ağır bağımlılıkları dışla (YOLO pivotundan kalma;
# requirements.txt'de bulunsalar da app/ içinde import edilmezler). PyInstaller
# zaten import edilmeyeni almaz, ama yanlışlıkla çekilmelerini de engelleriz.
EXCLUDES = [
    "pygame", "albumentations", "skimage", "scipy", "matplotlib",
    "tkinter", "pandas", "IPython", "pytest", "setuptools",
    "torch", "torchvision", "tensorflow", "onnx", "onnxruntime",
]

a = Analysis(
    ["motifika_launcher.py"],
    pathex=["."],
    binaries=cv2_binaries,
    datas=cv2_datas,
    hiddenimports=cv2_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="motifika",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="motifika",
)
