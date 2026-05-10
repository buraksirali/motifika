# MOTİFİKA — Tam Kullanım ve Kod Akışı Rehberi

Bu dosya: uygulamanın açılmasından kapanmasına kadar **hangi kullanıcı eyleminde hangi kod ne yapıyor** sorusunun adım adım, satır referanslı cevabıdır. Her bölüm "kullanıcı ne yapar → kod ne çalışır → ne işe yarar" sırasıyla.

---

## 1. Sistem Mimarisi (kuş bakışı)

| Modül | Sorumluluk |
|---|---|
| [app/main.py](app/main.py) | Orkestrasyon. Argümanlar, kamera, ana döngü, tuşlar. |
| [app/pattern.py](app/pattern.py) | Bir motif görselini → renk paleti + ızgara JSON'una dönüştürür. |
| [app/calibration.py](app/calibration.py) | Kullanıcıdan 4 köşeyi alıp homography matrisi üretir. |
| [app/progress.py](app/progress.py) | Kameradan "şu an hangi sırada örülüyor" otomatik tespit eder. |
| [app/color_check.py](app/color_check.py) | Aktif sıradaki renkleri palette ile karşılaştırır. |
| [app/overlay.py](app/overlay.py) | Chart şablonunu kameranın perspektifine bükerek üstüne bindirir (AR). |
| [app/ui.py](app/ui.py) | Sağdaki bilgi panelini çizer ve kamera ile birleştirir. |

Veri akışı:

```
motif.jpg ──pattern.py──▶ chart.json
                                │
kullanıcı tıklamaları ──calibration.py──▶ calibration.json
                                │
                                ▼
              ┌──── ana döngü (main.py) ────┐
kamera ──▶ frame ──▶ progress.update ──▶ aktif sıra
              │                                 │
              ├──▶ color_check.check_active_row ─▶ uyumsuzluklar
              │
              ├──▶ overlay.render ──▶ AR görüntü
              │
              └──▶ ui.render_panel + compose ──▶ ekran
```

Üç tane "uzay" var:
- **Chart-birim uzayı**: (0..cols, 0..rows). Bir hücre = 1 birim.
- **Chart-piksel uzayı**: chart-birim × `CELL_PX` (overlay'de 16, progress/color_check'te 12).
- **Kamera-piksel uzayı**: kameranın ham karesi.

Bu üç uzay arasında geçiş **3×3 homography matrisleri** ile yapılır. Bu matrisler kalibrasyon dosyasından gelir.

---

## 2. Ön Hazırlık (uygulama açılmadan önce)

Kullanıcının elinde olması gereken iki şey:

**a) Motif görseli** — `eli_belinde.jpg` veya `hayat_agaci_ornek.png` proje kökünde duruyor; [main.py:44-47](app/main.py#L44-L47) `MOTIFS` sözlüğünde tanımlı.

**b) Kamera** — `/dev/video0` (USB webcam veya Pi CSI). `cv2.VideoCapture(0)` ile açılır.

İlk çalıştırmada chart yoksa otomatik üretilir; kalibrasyon yoksa otomatik istenir. Kullanıcının elle yapması gereken hiçbir ön adım yok.

---

## 3. Uygulamanın Açılışı

### 3.1 Komut

```bash
python -m app.main --motif eli_belinde --rows 30 --cols 60 --camera 0
```

**Ne çalışır:** Python `-m app.main` ile [app/main.py:264](app/main.py#L264)'teki `if __name__ == "__main__":` bloğu tetiklenir, `main()` çağrılır.

### 3.2 Argümanların okunması — [main.py:104-122](app/main.py#L104-L122)

```python
ap.add_argument("--motif", choices=list(MOTIFS.keys()), ...)   # 108
ap.add_argument("--rows", type=int, required=True)              # 109
ap.add_argument("--cols", type=int, required=True)              # 110
ap.add_argument("--palette", type=int, default=4)               # 111
ap.add_argument("--direction", choices=["bottom_up", "top_down"], default="bottom_up")  # 112
ap.add_argument("--camera", type=int, default=0)                # 113
ap.add_argument("--image", type=Path, default=None)             # 114
ap.add_argument("--recalibrate", action="store_true")           # 118
ap.add_argument("--no-color-check", action="store_true")        # 121
```

**Ne işe yarar:**
- `--motif`: hangi katalog girdisi kullanılacak.
- `--rows/--cols`: kilim ızgara boyutu (ör. 30×60).
- `--palette`: kaç renkli palet (k-means'in k'sı).
- `--direction`: dokuma yönü.
- `--camera`: aygıt indeksi (`/dev/video0` için `0`).
- `--image`: kamera yerine sabit görsel kullan (test/debug).
- `--recalibrate`: var olan kalibrasyonu yok say, baştan iste.
- `--no-color-check`: renk kontrolünü kapat.

Eksik veya yanlış argümanda argparse hata yazıp çıkar.

### 3.3 Chart hazırlığı — [main.py:124-126](app/main.py#L124-L126) → `ensure_chart`

`ensure_chart(motif, rows, cols, palette)` çağrılır ([main.py:54-66](app/main.py#L54-L66)):

1. `assets/<motif>_chart.json` var mı diye bakar.
2. **Varsa**: hiçbir şey yapmaz, yolu döndürür.
3. **Yoksa**: kaynak görseli ([main.py:45-46](app/main.py#L45-L46)) alır, `build_chart()` ile üretir, JSON'a kaydeder.

`build_chart` neyi nasıl yapıyor (pattern.py):

| Adım | Fonksiyon | Ne yapar |
|---|---|---|
| 1 | [pattern.py:113 `flatten_alpha`](app/pattern.py#L113) | PNG'lerin alpha kanalını beyaz zemine düzleştirir (saydamlık silinir). |
| 2 | [pattern.py:262 `pixelate`](app/pattern.py#L262) | `cv2.resize` ile resmi `(cols, rows)` boyutuna küçültür (`INTER_AREA` = anti-alias). |
| 3 | [pattern.py:310 `quantize_palette`](app/pattern.py#L310) | `cv2.kmeans` ile k renkli palet bulur, her hücreye bir indeks atar. |
| 4 | [pattern.py:546 `save_chart`](app/pattern.py#L546) | JSON yazar, ek olarak `<motif>_chart.preview.png` görsel önizleme üretir. |

**Sonuç:** `assets/eli_belinde_chart.json` içinde `{rows, cols, palette: [[r,g,b],...], grid: [[idx,...]]}` yapısı.

### 3.4 Chart'ın belleğe yüklenmesi — [main.py:125](app/main.py#L125)

```python
chart = Chart.load(chart_path)
```

[overlay.py:99 Chart.load](app/overlay.py#L99) dataclass yapıcısını kullanarak JSON'u `Chart` nesnesine çevirir: `rows`, `cols`, `palette_rgb` (numpy uint8), `grid` (numpy int).

### 3.5 Kameranın açılması — [main.py:128](app/main.py#L128) → `open_camera_or_image`

[main.py:69-93](app/main.py#L69-L93):

1. `--image` verilmişse: `cv2.imread` ile dosyayı yükler, `provider = lambda: img.copy()`. **Kamera açılmaz.**
2. Verilmemişse: `cv2.VideoCapture(args.camera)` → V4L2 sürücüsü → `/dev/video0`.
3. `cap.isOpened()` False ise `RuntimeError`. Sebepleri: aygıt yok, başka uygulama tutuyor, `video` grup yetkisi yok.
4. `cap.set(WIDTH, 1280); cap.set(HEIGHT, 720)` ile çözünürlük ister; kamera desteklemezse en yakını verir (sessizce).
5. **Gerçek** boyutu `cap.get(...)` ile geri okur — homography bu boyuta bağlı, yanlış varsayılırsa eşleşmeler bozulur.
6. `provider` adında bir fonksiyon döndürür: her çağrıldığında `cap.read()` yapıp BGR kare üretir. Diğer modüller "kamera mı sabit görsel mi" detayını bilmek zorunda değil — provider arabirimi onları soyutluyor.

### 3.6 Kalibrasyon kontrolü — [main.py:130-139](app/main.py#L130-L139)

```python
cal_data = load_calibration(CAL_PATH)
cal_match = (cal_data is not None
             and cal_data.get("rows") == args.rows
             and cal_data.get("cols") == args.cols
             and tuple(cal_data.get("frame_size", [])) == frame_size)
```

`assets/calibration.json` okunur. Üç şart birden uyuyorsa (rows, cols, frame_size) eski kalibrasyon kullanılır; biri farklıysa yeniden kalibrasyon başlar. `--recalibrate` verildiyse de baştan başlatılır ([main.py:142](app/main.py#L142)).

---

## 4. Kalibrasyon Akışı (kullanıcı ilk açtığında veya `r` tuşuna bastığında)

### 4.1 Tetiklenme

İki yol:
- **Otomatik**: ilk çalıştırmada veya boyutlar değişmişse.
- **Elle**: ana döngüde `r/R` tuşu ([main.py:223-232](app/main.py#L223-L232)).

Çağrı: [main.py:96-101 `run_calibration_flow`](app/main.py#L96-L101) → [calibration.py:146 `collect_corners_interactive`](app/calibration.py#L146).

### 4.2 Pencere açılışı — [calibration.py:188-192](app/calibration.py#L188-L192)

```python
cv2.namedWindow("MOTIFIKA - Kalibrasyon", cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win, on_mouse)
```

Yeniden boyutlandırılabilir bir pencere açılır. Mouse olayları `on_mouse` fonksiyonuna yönlendirilir.

### 4.3 Sonsuz döngü içinde her kare — [calibration.py:199-320](app/calibration.py#L199-L320)

Her iterasyonda:

1. **Kareyi al**: `frozen` (SPACE ile dondurulmuş) yoksa `provider()` ile canlı; varsa donmuş kareyi kopyala.
2. **Tıklanan noktaları çiz** ([calibration.py:222-242](app/calibration.py#L222-L242)): `cv2.circle` ile sarı dolu daire + `cv2.putText` ile 1/2/3/4 numarası.
3. **2+ nokta varsa** ([calibration.py:245-258](app/calibration.py#L245-L258)): `cv2.polylines` ile noktaları yeşil çizgilerle birleştir.
4. **Talimat metni** ([calibration.py:260-275](app/calibration.py#L260-L275)): "Tikla: SOL UST (1/4)" gibi.
5. `cv2.imshow` + `cv2.waitKey(20)` ile pencereyi güncelle ve tuş bekle.

### 4.4 Kullanıcı eylemleri (kalibrasyon ekranında)

| Eylem | Tetiklenen kod | Sonuç |
|---|---|---|
| **Sol tık** | `on_mouse` ([calibration.py:164-177](app/calibration.py#L164-L177)) | `state["points"]` listesine `(x, y)` eklenir. 4 noktadan fazlası yok sayılır. |
| **`SPACE`** | [calibration.py:310-313](app/calibration.py#L310-L313) | O anki kareyi `state["frozen"]`'a alır. Titreyen kamerada hassas tıklama için. |
| **`R`** | [calibration.py:301-306](app/calibration.py#L301-L306) | Tıklanan noktaları temizler, frozen'ı kaldırır. Yeniden başla. |
| **`ESC`** | [calibration.py:291-296](app/calibration.py#L291-L296) | Pencereyi kapatır, `KeyboardInterrupt` fırlatır. Program çöker. |
| **`ENTER`** (4 nokta tamsa) | [calibration.py:318-320](app/calibration.py#L318-L320) | Döngüden çıkar, noktaları döndürür. |

**Kullanıcının yapması gereken:** `SOL ÜST → SAĞ ÜST → SAĞ ALT → SOL ALT` sırasıyla kilimin 4 köşesine tıkla, sonra ENTER. Sıra ÖNEMLİ; sıra bozulursa homography ters döner.

### 4.5 Homography hesaplama — [calibration.py:108-140 `compute_homography`](app/calibration.py#L108-L140)

`save_calibration` ([calibration.py:333](app/calibration.py#L333)) çağrılır, içinde:

```python
src_chart = np.array([[0,0], [cols,0], [cols,rows], [0,rows]], dtype=np.float32)
dst_cam = camera_corners   # kullanıcının tıkladığı 4 nokta
H_chart_to_cam = cv2.getPerspectiveTransform(src_chart, dst_cam)   # 125
H_cam_to_chart = np.linalg.inv(H_chart_to_cam)                     # 137
```

**Ne işe yarar:**
- `H_chart_to_cam`: chart-birim noktalarını → kamera piksellerine taşır. **Overlay** bu matrisi kullanır (şablonu kamera perspektifine eğmek için).
- `H_cam_to_chart`: kamera piksellerini → chart-birim noktalarına taşır. **Progress** ve **color_check** bu matrisi kullanır (kamerayı düzleştirmek için).

İki matris de JSON'a yazılır.

### 4.6 Kayıt — [calibration.py:343-360](app/calibration.py#L343-L360)

`assets/calibration.json` dosyası oluşturulur:
```json
{"rows": 30, "cols": 60,
 "camera_corners": [[x1,y1],...],
 "frame_size": [w, h],
 "H_chart_to_cam": [[...]],
 "H_cam_to_chart": [[...]]}
```

Kalibrasyon penceresi kapanır ([calibration.py:323](app/calibration.py#L323)), kontrol `main.py`'a döner.

---

## 5. Çalışan Nesnelerin Kurulumu — [main.py:148-167](app/main.py#L148-L167)

```python
H_chart_to_cam = np.array(cal_data["H_chart_to_cam"], ...)
H_cam_to_chart = np.array(cal_data["H_cam_to_chart"], ...)

tracker = ProgressTracker(rows, cols, direction)        # 153
renderer = OverlayRenderer(chart, direction)            # 154
ui = UIRenderer()                                        # 155
backend = HSVBackend(chart.palette_rgb)                  # 156

cv2.namedWindow("MOTIFIKA", cv2.WINDOW_NORMAL)          # 161
```

Burada her sınıfın `__post_init__`'i çalışır:

- **`OverlayRenderer.__post_init__`** ([overlay.py:130-135](app/overlay.py#L130-L135)): `_build_chart_layer()` çağrılır. Tüm chart'ın `(rows*16, cols*16, 3)` boyutlu BGR bitmap'i bir kez hesaplanır. Niye bir kez? Her render'da yapmak israf — ağır bir işlem.

- **`HSVBackend.__post_init__`** ([color_check.py:74-109](app/color_check.py#L74-L109)): Palet `cv2.cvtColor(BGR2LAB)` ile LAB renk uzayına çevrilip cache'lenir. Niye LAB? Çünkü RGB'de Öklid mesafesi gözle görünen renk farkını yansıtmaz, LAB'de yansıtır.

- **`ProgressTracker.__post_init__`** ([progress.py:73-87](app/progress.py#L73-L87)): `_score_ema = None`, `_active_row = 0`, `_manual_delta = 0` — durum değişkenleri sıfırlanır.

---

## 6. Ana Döngü — [main.py:169-245](app/main.py#L169-L245)

Bu döngü saniyede ~25-30 kez döner. Her iterasyon = bir kare = bir "tick".

### 6.1 Frame al — [main.py:170-173](app/main.py#L170-L173)

```python
frame = provider()
if frame is None: break
frame_idx += 1
```

`provider()` `cap.read()` yapar. None dönerse kamera kesildi → çık.

### 6.2 Aktif sıra tespiti — [main.py:176](app/main.py#L176) → `tracker.update`

```python
active_row = tracker.update(frame, H_cam_to_chart)
```

[progress.py:158-226 `update`](app/progress.py#L158-L226) içinde:

1. **`warp` ([progress.py:90-108](app/progress.py#L90-L108))**: `cv2.warpPerspective` ile kameradaki kilimi `(cols*12, rows*12)` boyutlu "kuş bakışı" görüntüye çevirir. Yamuk kilim → tam dik dikdörtgen.

2. **`row_scores` ([progress.py:111-155](app/progress.py#L111-L155))**: Düzleştirilmiş görüntüyü `cv2.cvtColor(BGR2GRAY)` ile gri yapar. Her sıra için bir "dokunmuşluk skoru" hesaplar:
   - **Yatay varyans** (`band.std()`): dokunmuş bölgede iplikler farklı renkte → varyans yüksek.
   - **Koyuluk** (`1 - mean/255`): dokunmuş bölge çözgü ipinden daha koyu.
   - Karma: `0.6 * std/80 + 0.4 * darkness`.

3. **EMA yumuşatma** ([progress.py:163-175](app/progress.py#L163-L175)): `score_ema = 0.4 * yeni + 0.6 * eski`. Anlık skorda titreşim olur (kamera shake, gölge); EMA bunu süzer.

4. **Eşik geçişini bul** ([progress.py:177-220](app/progress.py#L177-L220)):
   - En yüksek skor = `peak`. `peak < 0.05` ise henüz dokuma yok → başlangıç sırası.
   - Aksi halde `threshold = peak * 0.55` ile boolean dizi.
   - **bottom_up**: en yukarıdaki "dokunmuş" satırın **bir üstü** = aktif sıra (henüz dokunmamış, sıradaki).
   - **top_down**: en üstteki "dokunmamış" = aktif.

5. **Manuel offset** ([progress.py:225](app/progress.py#L225)): kullanıcının `+/-` ile ayarladığı `_manual_delta` eklenir.

### 6.3 Renk kontrolü (her 5 frame'de bir) — [main.py:179-188](app/main.py#L179-L188)

```python
if do_color_check and frame_idx % COLOR_CHECK_EVERY_N == 0:
    check_row = last_completed_row(active_row, chart.rows, direction)
    if check_row is not None:
        last_mismatches = check_active_row(frame, H_cam_to_chart, chart, check_row, backend)
```

`COLOR_CHECK_EVERY_N = 5` ([main.py:51](app/main.py#L51)). Her karede yapmak CPU yer; 5'te 1 makul denge.

**Önemli detay:** kontrol AKTİF sırada değil, **bir önceki tamamlanmış sırada** yapılıyor ([main.py:181-182](app/main.py#L181-L182), [color_check.py:259-269](app/color_check.py#L259-L269)). Çünkü aktif sıra henüz yarım, yarım hücreler hatalı sinyal verir.

[color_check.py:275-318 `check_active_row`](app/color_check.py#L275-L318) içinde:

1. **`sample_active_row` ([color_check.py:204-253](app/color_check.py#L204-L253))**: Yine `cv2.warpPerspective` ile kamerayı düzleştir, kontrol edilen sıranın **her hücresinin orta %50 alanından** ortalama BGR rengi al. Niye orta %50? Kenarlarda ızgara çizgisi ve komşu hücre kontaminasyonu var.

2. **`backend.classify` ([color_check.py:112-162](app/color_check.py#L112-L162))**: HSVBackend (aslında LAB kullanıyor, isim eski). Her örneği `cv2.COLOR_BGR2LAB`'a çevirir, broadcasting ile her örnek-her palet rengi için `(N, k)` mesafe matrisi hesaplar, `argmin` ile en yakın paleti bulur.

3. **Karşılaştırma** ([color_check.py:301-318](app/color_check.py#L301-L318)): `chart.grid[check_row]` ile beklenen indeksleri al. Eşleşmeyen hücreleri `(col, expected_idx, observed_idx, distance)` tuple'larıyla listele.

Sonuç `last_mismatches` listesinde tutulur, panele yazılır.

### 6.4 AR Overlay — [main.py:190](app/main.py#L190) → `renderer.render`

[overlay.py:219-294 `render`](app/overlay.py#L219-L294):

1. **Ölçek matrisi** ([overlay.py:237-242](app/overlay.py#L237-L242)): `_chart_layer` chart-piksel boyutunda (cell_px=16); ama `H_chart_to_cam` chart-birim bekliyor. `scale_inv` matrisiyle piksel → birime düşürür.

2. **Birleşik dönüşüm**: `M = H_chart_to_cam @ scale_inv` — tek matriste birleştir, tek warp ile bitir (hızlı).

3. **Chart bitmap'ini warp et** ([overlay.py:267](app/overlay.py#L267)): `cv2.warpPerspective(_chart_layer, M, out_size)` — şablonu kameranın gördüğü perspektife büker.

4. **Alpha haritası** ([overlay.py:177-216 `_alpha_mask`](app/overlay.py#L177-L216)):
   - Tamamlanmış sıralar: `DONE_ALPHA = 0.20` (soluk, dokumayı kapatmasın).
   - Aktif sıra: `ACTIVE_ALPHA = 0.65` (vurgulu).
   - Yapılacak sıralar: `TODO_ALPHA = 0.55` (motif net görünür).

5. **Alpha'yı da aynı M ile warp et** ([overlay.py:272](app/overlay.py#L272)). Yoksa renk ve saydamlık kayar.

6. **Per-pixel alpha blending** ([overlay.py:285-286](app/overlay.py#L285-L286)):
   ```
   blended = chart_renk * alpha + kamera_renk * (1-alpha)
   ```

7. **Aktif sıra çerçevesi** ([overlay.py:297-346](app/overlay.py#L297-L346)): Aktif sıranın 4 köşesini chart-birimde hazırlar, homojen koordinatlarla `H_chart_to_cam` ile çarparak kamera piksellerini bulur, `cv2.polylines` ile **sarı** çerçeve çizer ("buradasın!").

### 6.5 Yeniden boyutlandırma — [main.py:192](app/main.py#L192)

```python
ar_view = _resize_to_height(ar_view, target_h=720)
```

Kamera farklı çözünürlükte gelse bile UI yüksekliği sabit 720 piksel. En-boy oranı korunur.

### 6.6 Bilgi paneli — [main.py:194-198](app/main.py#L194-L198) → `ui.render_panel`

[ui.py:188-339 `render_panel`](app/ui.py#L188-L339) sağdaki paneli baştan çizer:

| Bölüm | Satır | Ne yazar |
|---|---|---|
| Başlık | [ui.py:207](app/ui.py#L207) | "MOTIFIKA" (turkuaz) |
| Sayaç | [ui.py:213-214](app/ui.py#L213-L214) | "Sira: 7 / 30" |
| İlerleme çubuğu | [ui.py:235-239](app/ui.py#L235-L239) | Yeşil dolgulu progress bar |
| Aktif sıra şeridi | [ui.py:243-253](app/ui.py#L243-L253) | O sıranın renk şablonu |
| Sonraki sıra şeridi | [ui.py:256-260](app/ui.py#L256-L260) | Bir sonraki sıranın renkleri (Tetris next-piece tarzı) |
| Sonra | [ui.py:263-267](app/ui.py#L263-L267) | İki sonraki sıra |
| Renk özeti | [ui.py:270-301](app/ui.py#L270-L301) | "Sonraki: 12 Kırmızı, 8 Siyah" gibi |
| Hata uyarıları | [ui.py:304-323](app/ui.py#L304-L323) | "S5.12: Kırmızı yerine Siyah" (en fazla 5 hata) |
| Klavye yardımı | [ui.py:334-338](app/ui.py#L334-L338) | "[+/-] sira ayarla", "[r] kalibrasyon", "[q] cikis" |

Renk adları `_color_name` ile RGB → Türkçe ad'a kabaca çevrilir ([ui.py:55-104](app/ui.py#L55-L104)).

### 6.7 Birleştirme — [main.py:199](app/main.py#L199) → `ui.compose`

[ui.py:342-354](app/ui.py#L342-L354): `np.hstack([camera_view, panel])` ile yan yana yapıştırır. Yükseklikler eşit değilse `cv2.resize` ile uyarlar.

### 6.8 FPS göstergesi — [main.py:201-210](app/main.py#L201-L210)

```python
inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
fps_ema = 0.85 * fps_ema + 0.15 * inst_fps
cv2.putText(composed, f"{fps_ema:.1f} FPS", ...)
```

EMA ile yumuşatılmış FPS sağ üstte sarı yazar.

### 6.9 Görüntüle ve tuş bekle — [main.py:212-214](app/main.py#L212-L214)

```python
cv2.imshow("MOTIFIKA", composed)
key = cv2.waitKey(1) & 0xFF
```

`waitKey(1)` 1 ms bekler. **Kasıtlı düşük** — "olabildiğince hızlı" demek. Bu olmadan pencere donar (OS event loop için zorunlu).

---

## 7. Tuş Tepkileri (ana döngüde)

Her tuşun ne yaptığı [main.py:215-245](app/main.py#L215-L245):

### `q` veya `ESC` — Çıkış
**Kod:** `break` → döngüden çıkar.
**Sonra:** `finally` bloğu ([main.py:246-250](app/main.py#L246-L250)) çalışır, `cap.release()` + `cv2.destroyAllWindows()`.

### `+` veya `=` — Aktif sırayı +1
**Kod:** `tracker.bump(+1)` → [progress.py:229-235](app/progress.py#L229-L235).
**Ne olur:** `_manual_delta` artar, `_active_row` da hemen +1. Sonraki update'te otomatik tahmin yine yapılır ama bu offset KALICI eklenir.
**Niye var:** otomatik tespit yanlış bulduğunda kullanıcı düzeltsin diye.

### `-` veya `_` — Aktif sırayı -1
Aynı mantık, `bump(-1)`.

### `r` veya `R` — Yeniden kalibrasyon
**Kod:** [main.py:223-232](app/main.py#L223-L232).
1. `cv2.destroyWindow("MOTIFIKA")` — ana pencereyi kapat.
2. `run_calibration_flow(...)` — kalibrasyon penceresi açılır, kullanıcı yine 4 köşeye tıklar.
3. Yeni `H_chart_to_cam` ve `H_cam_to_chart` belleğe yüklenir.
4. `cv2.namedWindow("MOTIFIKA", ...)` — ana pencere yeniden açılır.
**Ne işe yarar:** kamera veya kilim kımıldadıysa kalibrasyon eskir; bu tuşla anında düzelt.

### `d` veya `D` — Yön değiştir
**Kod:** [main.py:233-239](app/main.py#L233-L239).
- `direction` `"bottom_up"` ↔ `"top_down"` toggle.
- `tracker.direction = direction` ve `renderer.direction = direction` güncellenir.
- `tracker.reset_manual()` çağrılır ([progress.py:256-259](app/progress.py#L256-L259)) — eski yöne göre biriken `_manual_delta` artık geçersiz.
**Ne işe yarar:** kilim aşağıdan yukarı mı, yukarıdan aşağı mı dokunduğuna göre.

### `c` veya `C` — Renk kontrolü aç/kapat
**Kod:** [main.py:240-245](app/main.py#L240-L245).
- `do_color_check` toggle.
- Kapatınca `last_mismatches = []` — eski uyarılar silinir.
**Ne işe yarar:** CPU darboğazı varsa veya renk uyarıları rahatsız ediyorsa.

---

## 8. Çıkış — [main.py:246-250](app/main.py#L246-L250)

```python
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
```

`try/finally` blokları sayesinde **hata olsa bile** kamera serbest bırakılır. `cap.release()` yapmazsan V4L2 cihazı kilitli kalır, sonraki çalıştırma "kamera açılamadı" der.

---

## 9. Bağımsız Çalıştırılabilir Modüller (yan kullanım)

Bunlar `app.main` dışında ayrıca çalışabilen yardımcı CLI'lar.

### 9.1 `python -m app.pattern <görsel> --rows N --cols M --palette K --out file.json`

Sadece chart üretimi. [pattern.py:601-649](app/pattern.py#L601-L649). Manuel olarak farklı boyutlarda chart denemek için.

### 9.2 `python -m app.calibration --rows N --cols M --camera 0`

Sadece kalibrasyon — ana programdan ayrı. [calibration.py:382-445](app/calibration.py#L382-L445). Sadece `assets/calibration.json` üretir, başka iş yapmaz.

### 9.3 `python -m app.progress <görsel>`

Smoke test. [progress.py:265-284](app/progress.py#L265-L284). Tek bir görselde aktif sırayı bulup ekrana skorları yazar (debug için).

### 9.4 `python -m app.overlay --chart ... --calibration ...`

Smoke test. [overlay.py:352-384](app/overlay.py#L352-L384). Sahte gri kare üretir, üstüne overlay basıp PNG kaydeder.

---

## 10. Hızlı Karar Tablosu (kullanıcı için)

| Sorun / İhtiyaç | Yapılacak |
|---|---|
| İlk açılış | `python -m app.main --motif eli_belinde --rows 30 --cols 60 --camera 0` → 4 köşeye tıkla → ENTER |
| Kalibrasyon kaymış | `r` tuşu |
| Aktif sıra yanlış | `+/-` tuşları |
| FPS düşük | Kamera çözünürlüğünü düşür (kodda `frame_size_hint`) |
| Renk uyarıları rahatsız | `c` tuşu |
| Yön ters | `d` tuşu |
| Yeni motif eklemek | [main.py:44-47](app/main.py#L44-L47) `MOTIFS` sözlüğüne ekle, görseli kök dizine koy |
| Chart boyutu değiştirmek | `--rows / --cols` farklı verince eski chart varsa silinmiyor, manuel sil veya `--out` ile farklı dosya |
| Çıkış | `q` veya `ESC` |

---

## 11. Tek Tweet Özeti

**Açılış:** chart hazırla → kamera aç → kalibrasyon (4 köşe tıkla).
**Döngü:** kareyi al → düzleştir → aktif sırayı bul → renkleri kontrol et → şablonu kameraya bindir → paneli çiz → ekrana bas → tuş dinle.
**Tuşlar:** `q`=çıkış, `r`=yeniden kalibre, `+/-`=sıra düzelt, `d`=yön, `c`=renk kontrolü.
**Çıkış:** kamerayı bırak, pencereleri kapat.
