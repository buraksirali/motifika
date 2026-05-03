# MOTİFİKA — YOLO'dan AR + Klasik CV'ye Pivot Planı

## Context

PDF rehberindeki orijinal MOTİFİKA mimarisi (kamera → YOLOv8-nano → Türkçe uyarı) gerçek dokuma fotoğrafı toplanamadığı için askıya alındı. Yarışma 7 Haziran 2026; bugün 3 Mayıs — efektif ~5 hafta kaldı (PDF'teki H4 başında).

**Pivot kararı:** Eğitim verisi gerektirmeyen, klasik OpenCV + ekran üstü AR overlay yaklaşımına geçilecek. Sistem hatayı tespit etmek yerine doğru olanı **görsel olarak göstererek** kullanıcıya rehberlik edecek; Tetris'in "next piece" gibi sonraki adımı önerecek. Hailo-8L AI HAT donanımda kalacak ve küçük bir renk sınıflandırma modeli için kullanılacak (yarışma anlatısında NPU'nun yeri korunsun).

**Kullanıcının onayladığı kararlar:**
- Yüzey: kilim dokuma tezgahı (dikey çözgü ipliği yapısı)
- AR yöntemi: ekran üzerinde transparan overlay (projektör yok)
- Izgara boyutu (sıra/sütun sayısı): kullanıcı **manuel girer**, OpenCV ile sayılmaz
- İlerleme takibi: OpenCV ile **otomatik** atkı cephesi tespiti
- Şablon hazırlığı: motif görselini **otomatik piksellestirme** ile NxM ızgaraya çevir
- Hailo-8L: renk sınıflandırma için **kullanılacak**

## Mimari

```
[Önyükleme] kullanıcı sıra×sütun girer + motif seçer (eli belinde / hayat ağacı)
    ↓
[1] pattern.py: motif görselini NxM ızgaraya piksellestir → chart.json
    ↓
[2] calibration.py: kullanıcı çalışma alanının 4 köşesine tıklar → homography matrisi
    ↓
─── ANA DÖNGÜ (her kare) ─────────────────────────────────────────────
    Pi Camera frame
        ↓
[3] progress.py: atkı cephesini tespit et → mevcut sıra indeksi
        ↓
[4] color_check.py: son sıradaki hücre renklerini örnekle
        ↓ Hailo-8L renk sınıflandırma → beklenen renkle karşılaştır
        ↓
[5] overlay.py: homography ile ŞABLONU çalışma alanına yansıt (yapılan kısmı sönük, yapılacak kısmı parlak)
        ↓
[6] ui.py: sol=kamera+overlay, sağ=next-piece paneli + Türkçe uyarı/öneri
─────────────────────────────────────────────────────────────────────
```

## Bileşenler

### 1. `app/pattern.py` — Motif → Chart
- `eli_belinde.jpg` / `hayat_agaci_ornek.png` girdi
- Kullanıcının verdiği `(rows, cols)` boyutuna `cv2.resize(INTER_AREA)` ile küçült
- Her hücre için dominant renk: küçük resim üzerinde k-means (k=palette_size, ör. 6-8)
- Çıktı: `assets/<motif>_chart.json` → `{rows, cols, palette: [(r,g,b),...], grid: [[palette_idx,...]]}`
- `scripts/triage.py` ve `scripts/augment.py` referans olarak kalır (silmek yok)

### 2. `app/calibration.py` — ROI + Homography
- İlk açılışta kullanıcı kameradan kilim çalışma alanının 4 köşesine sırayla tıklar
- `cv2.getPerspectiveTransform` ile chart-pikseli ↔ kamera-pikseli matrisleri kaydedilir
- `assets/calibration.json` olarak persist edilir (her kalibrasyon değişene kadar tekrar tıklamaya gerek yok)

### 3. `app/progress.py` — Atkı Cephesi Tespiti
- Çözgü ipleri açık/sıra dokuması koyu renk yoğun → bölge bazında **dikey yoğunluk profili** (`row_intensity = mean(gray, axis=1)` warped ROI üzerinde)
- Üstten tarama: profilde "düşük varyans + parlak" → henüz dokunmamış; "yüksek varyans + koyu" → dokunmuş
- Geçiş satırı = aktif sıra. Hareketli ortalama + histerezis ile titremeyi azalt
- Manuel `+/-` tuşları ile düzeltme (hız > kesinlik)

### 4. `app/color_check.py` — Hailo-8L Renk Sınıflandırması
- **Eğitim:** 6-8 renk paletinden örnek ip yumakları çekilir (her renk ~30 fotoğraf, augmentation ile 300+). Küçük MobileNetV2/EfficientNet-Lite0 transfer learning (Colab'da 15 dk eğitim)
- **Inference:** son sıranın N hücresinden kırpılan küçük yamalar → Hailo-8L → renk indeksi
- Beklenen renkle uyuşmazsa: ekranda Türkçe **"Sıra X, sütun Y'de mavi yerine kırmızı kullanıldı"**
- **Fallback (Hailo-8L bağlanamazsa):** HSV uzayında merkez paletine en yakın mesafe (klasik). Demo en azından çalışır.

### 5. `app/overlay.py` — AR Şablon
- Chart pikselleri → çalışma alanı pikselleri (homography ile warp)
- İki katmanlı render:
  - Tamamlanmış sıralar: chart rengi düşük opaklıkta (rehber, kapatmasın)
  - Yapılacak sıralar: chart rengi yüksek opaklık + ızgara çizgileri
  - Aktif sıra: sarı vurgu çerçevesi (kullanıcı şu an buradasın)

### 6. `app/ui.py` — Ekran Düzeni
- Sol panel: kamera + AR overlay (640×480 ya da 720p)
- Sağ panel:
  - Üstte "Sonraki Sıra" şeridi (Tetris next-piece): `aktif sıra+1` ve `aktif sıra+2`'nin renk dizisi büyük büyük
  - Altta Türkçe uyarı/durum metni (örn. "Sıra 17 / 40 — sonraki: 8 kırmızı + 4 lacivert")
- Pygame veya `cv2.imshow` + numpy paneli (basit olan tercih)

### 7. `app/main.py` — Orkestrasyon
- argparse: `--motif eli_belinde --rows 40 --cols 40 --camera 0`
- İlk açılış: kalibrasyon ekranı; sonraki açılışlar: kayıtlı kalibrasyonu yükle
- Ana döngü 10+ FPS hedefli; ağır işlemler (renk classify) opsiyonel kareleme (her 5 karede bir)

## Kritik dosyalar

| Yol | Durum | Görev |
|---|---|---|
| `app/main.py` | yeni | giriş noktası, döngü |
| `app/pattern.py` | yeni | motif → chart.json |
| `app/calibration.py` | yeni | ROI 4-köşe + homography |
| `app/progress.py` | yeni | atkı cephesi otomatik tespit + manuel düzeltme |
| `app/overlay.py` | yeni | AR overlay rendering |
| `app/color_check.py` | yeni | Hailo-8L renk sınıflandırma + HSV fallback |
| `app/ui.py` | yeni | sol kamera / sağ panel |
| `assets/eli_belinde_chart.json` | üretilecek | pattern.py çıktısı |
| `assets/hayat_agaci_chart.json` | üretilecek | pattern.py çıktısı |
| `assets/calibration.json` | runtime | kalibrasyon kalıcılığı |
| `eli_belinde.jpg`, `hayat_agaci_ornek.png` | mevcut | kaynak motif |
| `scripts/*` | dokunma | YOLO scriptleri referans olarak kalsın |
| `requirements.txt` | yeni | opencv-python, numpy, pygame, scikit-image (opsiyonel: hailo-platform) |

## Yeniden kullanılacak araçlar (mevcut kod tabanından)

- `scripts/augment.py` içindeki `fit_to_target()` letterbox fonksiyonu → renk classifier eğitim verisi hazırlığı için aynen kullanılabilir
- `scripts/augment.py` içindeki Albumentations pipeline → renk classifier augmentation için aynen kullanılabilir (perspektif, ışık, gürültü)
- `scripts/validate.py` → renk classifier eğitim klasörlerini temizlemek için kullanılır

## Verifikasyon

- **Birim:** `pattern.py` test: bilinen referans görsel için chart.json'u görsel olarak doğrula (matplotlib heatmap)
- **Geliştirme:** Laptop webcam + basit kareli kâğıda boyalı renkli kareler → calibrasyon + progress + overlay end-to-end
- **Donanım:** Pi 5 + Pi Camera Module 3 üzerinde `python -m app.main --motif eli_belinde --rows 40 --cols 40` → ≥10 FPS
- **Hailo-8L:** Renk classifier `.hef` dosyasına dönüştürülmüş halde NPU'da inference → tek kare üzerinde tahmin doğruluğu ≥%85
- **Demo (H5):** Gerçek kilim parçası üzerinde, kasıtlı bir renk hatası + bir sıra atlama. Sistem her ikisini de uyararak ve doğru sonrakini göstererek geçmeli

## Takvim

| Sprint | Tarih | Çıktı |
|---|---|---|
| H4a | 5-7 May | `pattern.py` + `calibration.py` çalışıyor; chart.json üretiliyor; kalibrasyon kaydediliyor |
| H4b | 8-11 May | `progress.py` + `overlay.py` + `ui.py` entegre; laptop webcam ile end-to-end demo |
| H5a | 12-14 May | Renk classifier eğitimi (Colab) + Hailo-8L conversion + entegrasyon. HSV fallback önce devreye girer |
| H5b | 15-18 May | Pi 5 deploy, FPS optimizasyonu, gerçek kilim testi, demo videosu, sunum |

## Riskler ve azaltmalar

- **Atkı cephesi tespiti güvenilmez olabilir** → manuel `+/-` her zaman erişilebilir; demo senaryosunda kontrol edilebilir
- **Hailo-8L pipeline öğrenme eğrisi** → HSV fallback önce çalışır halde olur; NPU bonus
- **Pi 5 CPU FPS yetmezse** → frame skip (overlay her karede, ağır işlem her 5'te bir); 320×240 ROI
- **Kalibrasyon her oturumda gerekirse can sıkar** → sabit kamera montajı varsayımı + persist (`calibration.json`)
