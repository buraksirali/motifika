"""PyInstaller giriş noktası.

`app.main` paketi göreli `from app.calibration import ...` gibi mutlak paket
importları kullandığı için PyInstaller'a tek bir `python -m app.main` yerine bu
küçük başlatıcıyı veriyoruz. main() içinde donmuş modda çalışma dizini exe'nin
yanına alınır (assets/, *.jpg/png, calibration.json göreli yolları için).
"""
from app.main import main

if __name__ == "__main__":
    main()
