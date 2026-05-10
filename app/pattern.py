# ============================================================================
# DOSYA HAKKINDA (DOCSTRING)
# ============================================================================
# Üçlü tırnak `"""..."""` Python'da DOCSTRING denir = "dökümantasyon metni".
# Bu metin ÇALIŞMAZ. Sadece insanlar okusun diye var.
# `help(pattern)` yazarsan bu metin çıkar. Profesyonel kodda hep en başta olur.
"""Motif görselini NxM ızgaralı chart'a çevirir.

Bu dosya ne yapıyor (kısaca):
    1. Bir resim al (örn. eli_belinde.jpg)
    2. Onu küçük bir ızgaraya küçült (örn. 30 satır × 60 sütun)
    3. Renkleri "yuvarla" — örneğin 5000 farklı tonu 4 tona indirgele
    4. Sonucu JSON dosyası olarak kaydet (Lego talimatı gibi)

Komut satırından kullanım:
    python -m app.pattern eli_belinde.jpg --rows 40 --cols 40 --palette 4 \
        --out assets/eli_belinde_chart.json

Çıktı JSON şeması:
    {
        "source": "eli_belinde.jpg",
        "rows": 40, "cols": 40,
        "palette": [[r,g,b], ...],   # 0..255 BGR sırasında değil, RGB
        "grid": [[palette_idx, ...], ...]   # rows x cols
    }
"""

# ============================================================================
# İMPORTLAR — başka kütüphanelerden ALETLER getiriyoruz
# ============================================================================
# `import` Python'da "şu kutudaki aletleri buraya getir" demek.
# Olmasaydı bütün kodu sıfırdan yazmamız gerekirdi (resim oku, JSON üret vs).

# Örneğin Python'un geçmiş bir sürümünü kullanıyoruz. Diyelim ki Python 3.7
# Bana Python'un 3.9 sürümünde eklenen birşey lazım
# Bu yeni versiyonda eklenen şeyi versiyon değiştirmeden (3.7'de devam ederek)
# Kullanabilmemizi sağlayan paket.
#
# DAHA TEKNİK: Bu satır olmadan `def f(x: list[int])` veya `x: int | None` gibi
# tip yazımları eski Python'da PATLAR. `annotations` özelliği bunları string olarak
# saklar, çalışma anında değil. Yani tip ipucu sadece yazı, kod akışını etkilemiyor.
# Hep en üste yazılır, modern Python alışkanlığı.
from __future__ import annotations

# `argparse` = "argument parser" = argüman ayrıştırıcı.
# Komut satırında (terminal'de) kullanıcı şunu yazıyor:
#     python pattern.py eli_belinde.jpg --rows 30 --cols 60
# argparse bu kelimeleri ayrıştırıp programa "aha, image=eli_belinde.jpg, rows=30,
# cols=60" diye söyler.
# Olmasaydı `sys.argv` listesine bakıp elle parçalamamız gerekirdi (acı).
# Üstelik argparse `--help` desteği veriyor, hata mesajları üretiyor — bedava.
import argparse

# `json` = JavaScript Object Notation. EVRENSEL bir veri saklama formatı.
# Aslında JavaScript'le ilgisi yok artık, her dilde kullanılır.
# Sözlük gibi: anahtar-değer çiftleri.
# Örnek:
#     {"isim": "Ali", "yas": 25, "renkler": ["kırmızı", "mavi"]}
# Bu modülle Python sözlüğünü METİNE (`json.dumps`), metni sözlüğe (`json.loads`)
# çeviriyoruz. Bu sayede chart'ı dosyaya yazıp sonra geri okuyabiliyoruz.
import json

# `Path` = "Yol". Dosya yollarını temsil eden AKILLI bir sınıf.
# Düz string ile çalışsan: "klasör/" + "dosya.txt" = "klasör/dosya.txt" (Linux OK)
# Ama Windows'ta `\` ayraç. Path bunu otomatik halleder, OS'e göre doğru ayraç koyar.
# Ayrıca yol manipülasyonu için süper metodları var:
#   p.exists()        → dosya var mı?
#   p.read_text()     → dosyayı oku
#   p.with_suffix()   → uzantıyı değiştir
#   p.parent          → üst klasörü al
#   p.name / p.stem   → son parça / uzantısız son parça
# `from X import Y` = "X kütüphanesinden sadece Y aletini al".
# `import pathlib` deseydik her seferinde `pathlib.Path` yazmak gerekirdi.
from pathlib import Path

# `cv2` = OpenCV. "Open Computer Vision" = açık kaynak görüntü işleme.
# Görüntü işlemenin İSVİÇRE ÇAKISI. Resim oku, yaz, küçült, çiz, perspektif çevir,
# k-means, renk uzayı dönüşümleri, video oynatma... hepsi.
# Adındaki "2" tarihsel — eski API "cv" idi, yenisi "cv2" oldu.
# Pip'te paket adı `opencv-python`, importta `cv2`. Kafa karıştırıcı ama öyle.
import cv2

# `numpy` = "Numerical Python". Sayısal hesaplama kütüphanesi.
# Python'un kendi listesi YAVAŞ:
#   - Her eleman ayrı bir Python nesnesi (referans).
#   - Bellek dağınık.
#   - Toplama gibi işlemler tek tek yapılır.
# numpy "ndarray" sunar:
#   - Hepsi aynı tipte sayı (int/float).
#   - Bellekte yan yana (cache-friendly).
#   - C dilinde yazılmış kernel'ler kullanır → 100-1000x hızlı.
# `as np` = "bundan sonra numpy'ı `np` diye kısa yaz" (gelenek, herkes böyle yapar).
# Resimler ASLINDA numpy dizisidir: 3 boyutlu (H × W × 3) sayı tablosu.
import numpy as np


# ============================================================================
# FONKSİYON 1: PNG'lerin ALPHA (saydamlık) kanalını beyaz zemine düzleştir
# ============================================================================
# `def` = "define" = fonksiyon tanımla. Fonksiyon = küçük bir alet.
# Çağrılınca bir iş yapar, sonuç döndürür.
#
# Aletin adı: `flatten_alpha` (alpha'yı düzleştir).
# Parantez içi PARAMETRELER:
#   `img: np.ndarray` → "img" adında bir parametre.
#       `: np.ndarray` = TİP İPUCU (type hint). "Bana numpy dizisi vereceksin" der.
#       Tip ipucu zorunlu DEĞİL ama IDE/linter yardımcı oluyor (yanlış tip uyarısı).
#   `bg=(255, 255, 255)` → "bg" adında ikinci parametre.
#       `=` ile VARSAYILAN değer veriyor → BEYAZ (R=255, G=255, B=255).
#       Çağırırken `bg` vermezsen otomatik beyaz olur.
#       Tuple: parantez içinde virgülle ayrılmış değerler, IMMUTABLE (değişmez).
# `-> np.ndarray` = "bu fonksiyon numpy dizisi DÖNDÜRÜR" (yine tip ipucu).
def flatten_alpha(img: np.ndarray, bg=(255, 255, 255)) -> np.ndarray:
    # Bu da fonksiyonun docstring'i. `help(flatten_alpha)` yazınca çıkar.
    """PNG alpha kanalını beyaz zemine düzleştir."""

    # ----- KOŞUL: bu resmin alpha kanalı var mı? -----
    # Eğer alpha varsa içerdeki kodu çalıştır, yoksa resmi olduğu gibi geri ver.
    #
    # `img.ndim` ne demek?
    #   `ndim` = "number of dimensions" = boyut sayısı = dizinin "derinliği".
    #   ndim=1 → düz vektör:    [5, 3, 8, 1]
    #   ndim=2 → tablo:        [[1,2,3], [4,5,6]]
    #                           Gri resim böyle: yükseklik × genişlik.
    #   ndim=3 → kutu:         [[[r,g,b], [r,g,b]], [[r,g,b], [r,g,b]]]
    #                           Renkli resim böyle: yükseklik × genişlik × kanal.
    #   ndim=4 → kutuların listesi (video, batch).
    #
    # `img.shape` ne demek?
    #   `shape` = "şekil" = her boyutta KAÇ eleman var?
    #   Bir tuple olarak döner. Mesela (480, 640, 3):
    #     480 satır (yükseklik / height)
    #     640 sütun (genişlik / width)
    #     3 kanal (B, G, R)
    #
    # `img.shape[2]` ne demek?
    #   `[2]` = ÜÇÜNCÜ değer. Python 0'dan saymaya başlar:
    #     index 0 = yükseklik (480)
    #     index 1 = genişlik (640)
    #     index 2 = kanal sayısı (3)  ← BU
    #
    # Kanal sayısı ne demek?
    #   1 → gri (sadece parlaklık, 0-255)
    #   3 → renkli (BGR veya RGB — OpenCV BGR)
    #   4 → renkli + saydamlık (BGRA = 4. kanal alpha)
    #   PNG'lerde alpha olabilir (mesela transparan logolar).
    #
    # SENİN VERDİĞİN ÖRNEĞİN CEVABI (3 yapsam ne olur?):
    #   Eğer `img.shape[2] == 3` yazsaydık: alpha'sı OLMAYAN resimleri
    #   düzleştirmeye çalışırdık. Ama o resimde alpha YOK ki!
    #   Aşağıda `img[:, :, 3:4]` ile 4. kanalı okuyacağız → IndexError patlar.
    #
    # `and` = "hem bu hem o doğru olmalı" (mantıksal AND, ikisi true ise true).
    if img.ndim == 3 and img.shape[2] == 4:

        # ----- BGR (ilk 3 kanal) ile ALPHA'yı (4. kanal) AYIR -----

        # `img[:, :, :3]` → SLICING (dilimleme).
        # Slicing = bir dizinin BİR PARÇASINI seçmek.
        # Üç tane "boyut belirteci" var, virgülle ayrılmış:
        #   1. boyut (`:`) = "tüm satırları al"
        #   2. boyut (`:`) = "tüm sütunları al"
        #   3. boyut (`:3`) = "0'dan 3'e kadar (3 dahil değil)" → 0,1,2 = B,G,R
        # SONUÇ: BGR resmi, alpha'sız.
        #
        # SLICING ŞABLONLARI (önemli!):
        #   `:`     tek başına = "hepsi"
        #   `:3`    = "baştan 3'e kadar (3 hariç)"     → 0, 1, 2
        #   `3:`    = "3'ten sona kadar"
        #   `1:5`   = "1'den 5'e kadar (5 hariç)"      → 1, 2, 3, 4
        #   `::2`   = "her 2'sinden 1'ini al"           → atlama
        #   `::-1`  = "ters sırayla al"                 → mesela palette ters çevirme
        #   `1:5:2` = "1'den 5'e, 2 atlayarak"          → 1, 3
        #
        # `.astype(np.float32)` = TİP DÖNÜŞTÜR.
        # uint8 = "unsigned 8-bit integer" = 0-255 arası TAM SAYI (1 byte).
        # float32 = 32-bit ondalık sayı (4 byte). 0.5, 3.14, 200.7 gibi.
        # NEDEN dönüştürüyoruz?
        #   Aşağıda ÇARPIM yapacağız: bgr * alpha.
        #   alpha 0-1 arası float, ama uint8 sadece tam sayı tutar (0.5'i 0 yapar).
        #   Ayrıca 200 * 1.5 = 300, uint8 max 255 → TAŞAR (256 → 0 olur, kötü).
        #   Float'ta sınır yok pratikte (32 bit ile çok büyük sayılar tutar).
        bgr = img[:, :, :3].astype(np.float32)

        # `img[:, :, 3:4]` = sadece 4. kanal (alpha).
        # NEDEN `3:4` (3'ten 4'e kadar) ve sadece `3` değil?
        #   `[:, :, 3]` deseydik shape (H, W) olurdu — boyut DÜŞER (rank reduction).
        #   `[:, :, 3:4]` ise shape (H, W, 1) → boyut KORUNUR.
        # Boyut korumamızın sebebi: AŞAĞIDA broadcasting yapacağız.
        #   bgr (H, W, 3) ile alpha'yı çarparken:
        #     (H, W, 1) ile çarpınca son axis "yayılır" → her kanala uygulanır.
        #     (H, W) ile çarpsaydık şekil uyumsuzluğu hatası alırdık.
        #
        # `/ 255.0` = 0-255 aralığını 0-1 aralığına ÖLÇEKLE.
        # Alpha uint8 olarak 0-255 arası geliyor (0=tamamen şeffaf, 255=tamamen opak).
        # Biz 0-1 istiyoruz çünkü çarpım için mantıklı:
        #   "%30 opak" = 0.3 ile çarp.
        # NEDEN 255.0 (sondaki .0)?
        #   Python 3'te `255` int, `255.0` float. int'le böler isen sonuç da float olur
        #   (Python 3 sayesinde) ama açıkça yazmak gelenek + kod niyetini belirtir.
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0

        # ----- Arka plan rengini hazırla (broadcasting için) -----
        #
        # `bg` parametresi = (255, 255, 255) tuple'ı = BEYAZ.
        # `np.array(bg, dtype=np.float32)` = bunu numpy dizisi yap, float olarak.
        # Şu an shape: (3,) → düz 3 elemanlı dizi: [255, 255, 255].
        #
        # `.reshape(1, 1, 3)` = ŞEKLİNİ DEĞİŞTİR.
        # (3,) → (1, 1, 3). Yani [255, 255, 255] → [[[255, 255, 255]]].
        # Toplam eleman sayısı AYNI (3 tane), ama "kabuğu" değişti (paketleme).
        #
        # NEDEN? BROADCASTING için.
        # bgr shape: (H, W, 3) → mesela (480, 640, 3)
        # bg_arr shape: (1, 1, 3)
        # numpy "1 olan eksenleri tekrarla" mantığıyla:
        #   bg_arr aslında (480, 640, 3) gibi davranır → tüm pikseller beyaz.
        # Bellek kullanmadan! "Sanal tekrarlama" — gerçek kopya yapmaz.
        #
        # BROADCASTING KURALI (önemli!):
        #   İki dizinin şekli karşılaştırılır SAĞDAN SOLA.
        #   Eşitse: tamam.
        #   Birinin o eksende 1 ise: tekrarlanır.
        #   İkisi farklı ve hiçbiri 1 değilse: HATA.
        bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)

        # ----- KLASİK ALPHA BLENDING formülü -----
        # YENİ_RENK = ÖNDEKİ * alpha + ARKADAKİ * (1 - alpha)
        #
        # Mantık: alpha = saydamlık.
        #   alpha = 1 (tamamen opak):
        #     YENİ = ÖN * 1 + ARKA * 0 = ÖN  (sadece önü görüyoruz)
        #   alpha = 0 (tamamen şeffaf):
        #     YENİ = ÖN * 0 + ARKA * 1 = ARKA  (sadece arkayı görüyoruz)
        #   alpha = 0.5:
        #     YENİ = ÖN * 0.5 + ARKA * 0.5  (yarı yarıya karışım)
        #
        # ÇARPIM ŞEKİLLERİ:
        #   bgr: (H, W, 3) — front (renkli resim)
        #   alpha: (H, W, 1) — saydamlık değerleri
        #   bg_arr: (1, 1, 3) — beyaz arka plan
        # Hepsi (H, W, 3) gibi davranır broadcasting ile.
        # Sonuç: HER PİKSEL için ayrı ayrı blend hesabı, ama tek satırda + hızlı.
        out = bgr * alpha + bg_arr * (1.0 - alpha)

        # ----- Float'tan tekrar uint8'e dön -----
        # Resim kaydetme/gösterme uint8 ister (0-255).
        # `.astype(np.uint8)` float → uint8.
        # Eğer sayılar 0-255 dışına taşmışsa "wrap around" olur (256 → 0 olur, kötü).
        # Ama bizim formül 0-255 içinde kalır (alpha 0-1, bgr 0-255), endişe yok.
        return out.astype(np.uint8)

    # ----- Yukarıdaki `if` false ise -----
    # Yani alpha yoksa, resmi OLDUĞU GİBİ geri ver. Hiçbir şey yapma.
    # Bu "early exit" pattern'ı: kodun mantığı sade, gereksiz işlem yok.
    return img


# ============================================================================
# FONKSİYON 2: Resmi belirli ızgara boyutuna küçült
# ============================================================================
def pixelate(img_bgr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Görseli rows × cols ızgaraya küçült (her hücre = bir piksel)."""
    # `cv2.resize(src, dsize, interpolation)`:
    #   src: kaynak resim
    #   dsize: hedef boyut, (genişlik, yükseklik) → DİKKAT! (cols, rows) sıralaması!
    #   interpolation: hangi yöntemle küçültsün/büyültsün?
    #
    # ⚠️ TUZAK: cv2 her zaman (W, H) ister. Ama numpy shape (H, W) verir.
    # Bu kafa karıştırıcı. Hep akılda tut: "cv2 = (genişlik, yükseklik)".
    # Hatalı yazsan resim 90 derece dönmüş gibi gözükür ve sebebi anlaşılmaz.
    #
    # interpolation seçenekleri (her birinin kullanım alanı var):
    #   INTER_NEAREST   = "en yakın komşu". Hedefteki piksel = kaynaktaki en yakın.
    #                     HIZLI ama TIRTIKLI (alias).
    #                     Pixel art / mask resize için iyi.
    #   INTER_LINEAR    = "doğrusal". Komşu 2x2 pikseli interpolate eder.
    #                     Hızlı + yumuşak. Genel amaçlı varsayılan.
    #   INTER_AREA      = "alan ortalaması". Hedef piksel = kaynaktaki alanın ortalaması.
    #                     KÜÇÜLTME için EN İYİ. Anti-alias yapar (dişli kenar yok).
    #                     Bizim kullandığımız bu.
    #   INTER_CUBIC     = "kübik". 4x4 komşu kullanarak yumuşak büyütme.
    #                     BÜYÜTME için iyi, küçültmede gereksiz yavaş.
    #   INTER_LANCZOS4  = "lanczos". En kaliteli ama en yavaş büyütme.
    #
    # Biz büyük resmi (örn. 800×800) küçük ızgaraya (30×60) düşürdüğümüz için
    # INTER_AREA mantıklı.
    return cv2.resize(img_bgr, (cols, rows), interpolation=cv2.INTER_AREA)


# ============================================================================
# FONKSİYON 3: Renkleri "yuvarla" (k-means ile palet bul)
# ============================================================================
# K-MEANS NEDİR? (TEMEL FİKİR)
# Diyelim resimde 5000 farklı renk tonu var: koyu kırmızı, açık kırmızı,
# açık siyah, koyu siyah, krem, açık krem... ama sen 4 renge sıkıştırmak istiyorsun
# (çünkü dokumada sadece 4 renk ipliğin var).
#
# K-means şöyle çalışır:
#   1. RASTGELE 4 nokta seç (3 boyutlu RGB uzayında "merkezler").
#   2. Her piksele bak: "en yakın merkez hangisi?" (öklid mesafe) → o merkeze ata.
#   3. Her grubun ORTASINI hesapla → yeni merkezler.
#   4. 2-3'ü tekrarla, ta ki merkezler artık değişmesin.
#   5. Sonuç: 4 ana renk + her piksel hangi gruba ait olduğu.
#
# 3 boyutlu uzayı düşün: x=R, y=G, z=B. Her piksel bu uzayda bir nokta.
# K-means bu noktaları 4 KÜMEYE ayırıyor.
#
# `tuple[A, B]` = "iki şey döndürür: önce A, sonra B" (Python 3.9+ yazımı).
def quantize_palette(small_bgr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means ile k renkli palet bul. Dönüş: (palette_rgb [k,3], grid [rows,cols] int)."""

    # Resmin yüksekliği ve genişliğini al.
    # `small_bgr.shape` = mesela (30, 60, 3).
    # `[:2]` = "ilk 2 elemanı al" → (30, 60).
    # `rows, cols = ...` = TUPLE UNPACKING. İki değişkene aynı anda atama:
    #   rows = 30, cols = 60.
    # Tuple unpacking şöyle de olur:
    #   a, b, c = (1, 2, 3)
    #   ilk, *orta, son = [1, 2, 3, 4, 5]   → ilk=1, orta=[2,3,4], son=5
    rows, cols = small_bgr.shape[:2]

    # `.reshape(-1, 3)` = "şeklini (?, 3) yap, ?'i sen hesapla".
    # `-1` = "sen bul, ne çıkarsa". numpy hesaplar: 30*60 = 1800.
    # Yani (30, 60, 3) → (1800, 3). 1800 piksel, her biri 3 sayı.
    # K-means TABLO bekler: N örnek × D özellik.
    # Burada N=1800 piksel, D=3 (R, G, B üç boyut).
    #
    # `.astype(np.float32)` = float'a çevir. K-means float ister, ZORUNLU.
    pixels = small_bgr.reshape(-1, 3).astype(np.float32)

    # K-means'i ne zaman durduralım? (3 elemanlı tuple)
    #
    # 1. parametre: hangi şart?
    #    `cv2.TERM_CRITERIA_EPS` = "merkezlerin değişimi epsilon'dan küçükse dur".
    #    `cv2.TERM_CRITERIA_MAX_ITER` = "iterasyon sayısı sınıra gelince dur".
    #    `+` ile birleştirince: "ikisinden BİRİ olursa dur" (önce hangisi gelirse).
    #    Aslında bu bit-OR ama OpenCV bunu öyle tasarlamış.
    #
    # 2. parametre: max iterasyon = 20.
    #    20 yapsan: makul. 5 yapsan: çok kaba sonuç ama hızlı.
    #    100 yapsan: hassas ama yavaş (genelde gereksiz, 20'de zaten oturuyor).
    #
    # 3. parametre: epsilon = 0.5.
    #    Merkezler 0.5 birimden az değiştiyse dur.
    #    0.01 yapsan: çok hassas, çok iter koşar.
    #    1.0 yapsan: erken durur, biraz kaba ama hızlı.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)

    # `cv2.kmeans` 3 şey döndürür:
    #   compactness: kümelerin ne kadar iyi (sayı, biz umurumuzda değil).
    #   labels: her pikselin atandığı küme numarası → shape (N, 1).
    #   centers: her kümenin merkez noktası (renk) → shape (k, 3).
    #
    # `_` = "BU DEĞİŞKENİ ALMIYORUM" geleneği.
    # Python'da `_` geçerli bir değişken adı ama isim olarak kullanmak
    # "umursamadığımı" söyler. Linter de uyarmaz.
    #
    # ARGÜMANLAR:
    #   pixels: N × 3 örnek tablosu
    #   k: kaç küme (renk) olsun
    #   None: önceden etiket var mı? Yok, baştan bul.
    #         (Önceden etiketler verseydin "incremental clustering" yapardı.)
    #   criteria: yukarıda hazırladığımız durdurma şartları
    #   attempts=5: 5 farklı RASTGELE başlangıçla dene, en iyiyi al.
    #               K-means başlangıca duyarlıdır; aynı veri farklı başlangıçla
    #               farklı sonuç verebilir (lokal minimum tuzakları).
    #               5 deneme = makul tutarlılık.
    #               1 = hızlı ama bazen kötü. 20 = yavaş ama tutarlı.
    #   flags=cv2.KMEANS_PP_CENTERS: "k-means++" akıllı başlangıç.
    #               Birbirine UZAK başlangıç merkezleri seçer (akıllıca).
    #               Alternatif: KMEANS_RANDOM_CENTERS = tamamen rastgele (genelde kötü).
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )

    # `np.clip(x, low, high)` = "x'i low ile high arasına SIKIŞTIR".
    # x < low ise low yap, x > high ise high yap. Aradakiler değişmez.
    # K-means merkezleri float, bazen taşar:
    #   255.7 → 255 yap (geçerli renk).
    #   -0.3 → 0 yap.
    # Renk olarak 0-255 dışı geçersiz, kırpıyoruz.
    #
    # `.astype(np.uint8)` = float → uint8 (renk tipi).
    palette_bgr = np.clip(centers, 0, 255).astype(np.uint8)

    # `[:, ::-1]` = SLICING tekniği.
    # `:` = "tüm satırları al" (her renk için)
    # `::-1` = "sütunları TERS sırada al" (step=-1, geriye doğru git).
    # BGR → RGB çeviriyor: [B, G, R] olan sıralamayı [R, G, B] yapıyor.
    #
    # NEDEN OpenCV BGR ama biz RGB yapıyoruz?
    # Tarihi: OpenCV başlangıçta Windows/BMP uyumlu olsun diye BGR seçilmiş.
    # AMA herkes RGB ile düşünüyor (web, mobil, screen, çoğu kütüphane).
    # JSON'a yazarken insan RGB beklediği için RGB tercih ettik.
    palette_rgb = palette_bgr[:, ::-1]

    # `labels` şu an düz dizi: [0, 1, 0, 2, 1, 3, 0, ...] uzunluk N=1800.
    # `.reshape(rows, cols)` = bunu 2D tabloya çevir → (30, 60).
    # Yani her hücrede o pikselin atandığı küme indeksi (0..k-1).
    #
    # `.astype(int)` = tam sayı yap. K-means bazen float döndürür, biz int istiyoruz.
    grid = labels.reshape(rows, cols).astype(int)

    # İki şey geri veriyoruz: palet ve grid.
    # Çağıran taraf hem renkleri hem hangi pikselin hangi renk olduğunu bilecek.
    # Python tuple döndürürken parantez şart değil, virgül yeterli:
    #   `return a, b` = `return (a, b)`.
    return palette_rgb, grid


# ============================================================================
# FONKSİYON 4: Chart'ı görsel olarak kaydetmek için PNG önizleme yap
# ============================================================================
# Bu fonksiyon "sonuç doğru mu?" diye gözle bakabilmen için var.
# JSON'a sayılar yazıyoruz ya, gözle anlamsız.
# Bu PNG'yi açıp "evet kilim deseni gibi görünüyor" diyebiliyorsun.
def render_preview(palette_rgb: np.ndarray, grid: np.ndarray, cell_px: int = 16) -> np.ndarray:
    """Chart'ı görsel olarak doğrulamak için BGR önizleme oluştur."""
    # OpenCV BGR çalıştığı için palette'i geri çeviriyoruz.
    palette_bgr = palette_rgb[:, ::-1]

    # FANCY INDEXING — numpy'ın ÇOK GÜÇLÜ bir özelliği.
    # `palette_bgr` shape: (k, 3) → mesela 4 renk: [[B,G,R], [B,G,R], [B,G,R], [B,G,R]].
    # `grid` shape: (rows, cols) → her hücrede 0..k-1 arası sayı.
    # `palette_bgr[grid]` yapınca:
    #   numpy "her grid hücresine git, palette'den o indeksi al" diyor.
    #   Sonuç shape: (rows, cols, 3) — RENKLI RESİM!
    #
    # FOR DÖNGÜSÜYLE yapsak:
    #   for r in range(rows):
    #     for c in range(cols):
    #       result[r, c] = palette_bgr[grid[r, c]]
    #   1800 iterasyon, Python yavaş.
    # Fancy indexing ile TEK SATIRDA, C kodunda, çok hızlı.
    preview = palette_bgr[grid]

    # `np.repeat(arr, n, axis)` = "arr'ı axis yönünde n kere tekrarla".
    # Şu an preview shape: (30, 60, 3) — küçük, gözle zor görünür.
    # Her pikseli 16×16 bloğa şişirip 480×960 yapacağız.
    #
    # İçten dışa okuyoruz:
    #   İç: np.repeat(preview, cell_px, axis=0)
    #     axis=0 = SATIR yönü (dikey).
    #     Her satırı 16 kere tekrarla → satır sayısı 16x oldu → (480, 60, 3).
    #   Dış: np.repeat(..., cell_px, axis=1)
    #     axis=1 = SÜTUN yönü (yatay).
    #     Her sütunu 16 kere tekrarla → sütun sayısı 16x oldu → (480, 960, 3).
    # Sonuç: her hücre 16×16 piksel bloğa şişti.
    #
    # ALTERNATİF: cv2.resize ile INTER_NEAREST de yapılabilirdi:
    #   cv2.resize(preview, (cols*16, rows*16), interpolation=cv2.INTER_NEAREST)
    # Ama np.repeat daha açık niyetli (neyi tekrarladığımız belli).
    preview = np.repeat(np.repeat(preview, cell_px, axis=0), cell_px, axis=1)

    # Önizlemenin yeni yükseklik ve genişliği.
    h, w = preview.shape[:2]

    # ----- IZGARA çizgileri -----
    # `range(N)` = 0, 1, 2, ..., N-1 üretir (lazy iterator).
    # `grid.shape[0]` = satır sayısı, mesela 30.
    # `+ 1` çünkü 30 satır için 31 yatay çizgi gerekir (üst kenar + her satırın altı).
    for r in range(grid.shape[0] + 1):
        # `cv2.line(img, başlangıç, bitiş, renk, kalınlık)`.
        #
        # ⚠️ KOORDİNAT TUZAKLARI:
        #   cv2 koordinat → (x, y) sırasında!
        #   numpy index → [satır, sütun] = [y, x] sırasında!
        #   Yani cv2 ve numpy farklı! Bu tuzağa milyonlarca developer düşmüştür.
        #
        # Yatay çizgi: x değişiyor (0'dan w'ye), y SABİT (r * cell_px).
        # `(0, r * cell_px)` = sol kenardan başla, satır seviyesinde.
        # `(w, r * cell_px)` = sağ kenara git, aynı satır seviyesinde.
        #
        # `(40, 40, 40)` = BGR. Hepsi 40 olunca → KOYU GRİ.
        # 1 = kalınlık, 1 piksel.
        cv2.line(preview, (0, r * cell_px), (w, r * cell_px), (40, 40, 40), 1)

    # Sütun yönünde de aynısını yap.
    for c in range(grid.shape[1] + 1):
        # Dikey çizgi: x sabit (c * cell_px), y değişiyor (0'dan h'ye).
        cv2.line(preview, (c * cell_px, 0), (c * cell_px, h), (40, 40, 40), 1)

    return preview


# ============================================================================
# FONKSİYON 5: Yukarıdaki tüm aletleri kullanarak chart üret
# ============================================================================
# `-> dict` = Python sözlüğü (key-value çiftleri) döndürür.
def build_chart(image_path: Path, rows: int, cols: int, palette_size: int) -> dict:

    # `cv2.imread(yol, bayrak)` = resim oku.
    # `str(image_path)` = Path → string. cv2 string yolu ister, Path nesnesi anlamaz.
    #   (cv2 C++ kütüphanesi, Python Path'ini tanımıyor.)
    #
    # İkinci parametre = "OKUMA MODU":
    #   IMREAD_UNCHANGED = "olduğu gibi oku, alpha varsa onu da getir" (4 kanal mümkün).
    #   IMREAD_COLOR     = "renkli oku, alpha YOK SAY" (hep 3 kanal). VARSAYILAN.
    #   IMREAD_GRAYSCALE = "gri oku" (1 kanal).
    # PNG'lerin alpha'sı olabilir, kaybetmemek için UNCHANGED.
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    # ⚠️ ÖNEMLİ: cv2.imread dosya yoksa veya bozuksa `None` döner ama EXCEPTION FIRLATMAZ.
    # Bu Python'a göre garip; biz elle kontrol edip kendimiz fırlatıyoruz.
    if raw is None:
        # `raise X(...)` = ÖZEL DURUM (exception) FIRLAT.
        # Program durur, hata yukarıya kadar gider; bir yerde yakalanmazsa çöker.
        # `f"..."` = F-STRING (formatted string).
        # Süslü parantez `{x}` içine değişken yazınca yerine değeri konur.
        # f olmadan {image_path} olarak literal yazılırdı.
        # Diğer string formatları:
        #   "%s" % image_path           (eski stil)
        #   "{}".format(image_path)     (orta stil)
        #   f"{image_path}"             (yeni stil, en hızlı, en okunaklı)
        raise FileNotFoundError(f"Görsel okunamadı: {image_path}")

    # Sırayla 3 fonksiyon çağrısı (PIPELINE):
    bgr = flatten_alpha(raw)                            # alpha varsa beyaza düzleştir
    small = pixelate(bgr, rows, cols)                   # küçük ızgaraya küçült
    palette_rgb, grid = quantize_palette(small, palette_size)  # k renkli palet bul

    # ----- Sonucu sözlük olarak hazırla (JSON'a yazılabilir) -----
    # Python `dict` literal: `{"key": value, ...}`
    return {
        # `image_path.name` = "eli_belinde.jpg" (path'in son parçası).
        # `image_path.stem` = "eli_belinde" (uzantısız).
        # `image_path.suffix` = ".jpg" (uzantı).
        # `image_path.parent` = path'in üst klasörü.
        "source": image_path.name,
        "rows": rows,
        "cols": cols,
        # `.tolist()` = numpy dizisi → Python listesi.
        # `json.dumps` numpy bilmiyor (3rd party kütüphane), sadece liste/dict/sayı/string biliyor.
        # `.tolist()` olmadan TypeError alırız: "Object of type ndarray is not JSON serializable".
        "palette": palette_rgb.tolist(),
        "grid": grid.tolist(),
    }


# ============================================================================
# FONKSİYON 6: Chart sözlüğünü diske JSON+PNG olarak yaz
# ============================================================================
# `-> None` = "hiçbir şey döndürmez". Sadece YAN ETKİ için var (dosya yazıyor).
# `preview: bool = True` = varsayılanı açık.
def save_chart(chart: dict, out_path: Path, preview: bool = True) -> None:

    # `out_path.parent` = "out_path'in içinde olduğu klasör".
    # Path("a/b/c.json").parent = Path("a/b")
    # Path("a/b/c.json").name   = "c.json"
    # Path("a/b/c.json").stem   = "c"
    #
    # `.mkdir(parents=True)` = "klasörü oluştur, üst klasörler de yoksa onları da yap".
    #   parents=False olsaydı sadece son seviyeyi yaratırdı, üst yoksa hata verirdi.
    #   `mkdir -p` (Linux) ile aynı mantık.
    # `exist_ok=True` = "klasör zaten varsa hata verme, sus".
    #   exist_ok=False olsaydı klasör varsa FileExistsError alırdık.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # `json.dumps(obj, ...)` = Python sözlüğü → JSON STRING'i.
    # `json.loads(s)` = JSON string → Python sözlüğü. (zıt yön)
    # `json.dump(obj, file)` ve `json.load(file)` = doğrudan dosyaya yazar/okur.
    #
    # `ensure_ascii=False` = Türkçe "ş", "ğ" karakterlerini DÜZ yazsın.
    #   True (varsayılan) olsa: "Şıralı" → "Şıralı" (escape, çirkin).
    #   False: "Şıralı" → "Şıralı" (okunabilir, UTF-8 ile sorunsuz).
    #
    # `indent=2` = girinti seviyesi.
    #   None: tek satıra basar, küçük dosya ama okunaksız.
    #   2: 2 boşlukla girinti, okunaklı format.
    #   4: 4 boşluk, daha geniş.
    #
    # `out_path.write_text(string)` = string'i dosyaya yaz (varsa üzerine yaz).
    # Bytes için `.write_bytes()` var.
    out_path.write_text(json.dumps(chart, ensure_ascii=False, indent=2))

    # Preview istenmişse PNG önizlemeyi de oluştur.
    if preview:
        # JSON'dan okunmuş listeleri tekrar numpy dizisine çevir.
        # `np.array(liste, dtype=...)` = listeden numpy dizisi yap.
        palette_rgb = np.array(chart["palette"], dtype=np.uint8)
        grid = np.array(chart["grid"], dtype=int)

        prev = render_preview(palette_rgb, grid)

        # `with_suffix(".yeni")` = uzantıyı değiştir.
        # Path("a/b/c.json").with_suffix(".preview.png") = Path("a/b/c.preview.png")
        # Önce .json'ı SİLER, sonra ".preview.png" ekler.
        # (c.json.preview.png OLMAZ, c.preview.png olur.)
        prev_path = out_path.with_suffix(".preview.png")

        # `cv2.imwrite(yol, resim)` = resmi dosyaya yaz.
        # Uzantıdan format anlar (.png → PNG, .jpg → JPEG).
        # PNG kayıpsız (chart için ideal), JPEG kayıplı (renk bozulabilir).
        cv2.imwrite(str(prev_path), prev)


# ============================================================================
# CLI GİRİŞ NOKTASI — terminal'den çalıştırılınca burası çağrılır
# ============================================================================
def main():
    # ArgumentParser nesnesi oluştur. Argümanları buna ekleyeceğiz.
    ap = argparse.ArgumentParser()

    # ----- 1. argüman: ZORUNLU pozisyonel "image" -----
    # Başında `--` YOK → POZİSYONEL argüman (sırasına göre yakalanır).
    # `python pattern.py eli_belinde.jpg` dediğinde "eli_belinde.jpg" buna gider.
    # `type=Path` = otomatik Path nesnesine çevirir (string'i Path yapar).
    # `help` = `--help` yazılınca bu açıklama görünür.
    ap.add_argument("image", type=Path, help="kaynak motif görseli")

    # ----- 2. argüman: --rows -----
    # `--rows` (başında --) → İSİMLİ argüman (named, optional ama biz zorunlu yaptık).
    # `--rows 30` diye yazılır.
    # `required=True` = vermezsen argparse hata verip çıkar.
    # `type=int` = "30" string'ini 30 int'ine çevirir (cast).
    ap.add_argument("--rows", type=int, required=True)

    ap.add_argument("--cols", type=int, required=True)

    # `default=4` = vermezsen 4 say. `required=True` ile DEFAULT birlikte kullanılmaz
    # (zorunlu olan bir şeyin "varsayılanı" mantıksız).
    ap.add_argument("--palette", type=int, default=4, help="renk sayısı (k-means)")

    ap.add_argument("--out", type=Path, required=True, help="çıktı chart.json yolu")

    # `parse_args()` = gerçek argümanları alıp parse et.
    # Eksik veya hatalı argüman varsa burada hata verir ve program durur.
    # Sonuç bir Namespace nesnesi: `args.image`, `args.rows`, ... gibi erişebilirsin.
    # Tireli isimler (`--no-color-check`) `_` ile parse edilir (`args.no_color_check`).
    args = ap.parse_args()

    # ----- Random'a SABİT TOHUM ver -----
    # K-means biraz random içerir (başlangıç merkezleri rastgele).
    # Eğer tohumu sabitlemezsen her çalıştırmada palet biraz farklı çıkabilir.
    # Tohumu sabitlersen DETERMİNİSTİK olur (her sefer aynı sonuç).
    # 0 yerine 42, 1, 999 yazsan da olur — önemli olan SABİT olması.
    # NOT: cv2.kmeans aslında numpy random'ı kullanmıyor olabilir; bu satır
    # opensource'ta tartışmalı. Yine de zarar vermiyor.
    np.random.seed(0)

    # Üç adım:
    chart = build_chart(args.image, args.rows, args.cols, args.palette)
    save_chart(chart, args.out)

    # `print(...)` = ekrana yaz. f-string ile değişken araya konur.
    # `print` aslında dosyaya da yazabilir: `print(..., file=sys.stderr)`.
    print(f"chart kaydedildi: {args.out}")
    print(f"önizleme: {args.out.with_suffix('.preview.png')}")
    print(f"palet: {chart['palette']}")


# ============================================================================
# `if __name__ == "__main__":` — Python'ın en sık görülen deyimi
# ============================================================================
# `__name__` özel bir değişken. Python her modüle bunu otomatik atar:
#   - Dosya DOĞRUDAN çalıştırılıyorsa:        __name__ = "__main__"
#   - Dosya başka bir dosya tarafından IMPORT edildiyse:  __name__ = "pattern"
#
# Yani:
#   $ python pattern.py            →  if doğru, main() koşar
#   $ python -m app.pattern ...    →  if doğru, main() koşar
#   from app import pattern         →  if YANLIŞ, main() koşmaz
#
# Bu olmasaydı: import ettiğin anda main() çalışırdı (kötü, istemezsin).
# Mesela main.py içinde "from app.pattern import build_chart" deyince
# import sırasında pattern.main() koşardı, kullanıcıya "image yolu ver"
# diye sorardı, çöker.
# O yüzden bu kalıbı her dosyaya ekliyoruz.
if __name__ == "__main__":
    main()
