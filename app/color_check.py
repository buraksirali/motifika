# ============================================================================
# DOSYA HAKKINDA
# ============================================================================
"""Aktif sıradaki hücre renklerini çıkar ve beklenen palette ile karşılaştır.

Bu dosya ne yapıyor (kısaca):
  Kullanıcı bir sırayı dokuduktan sonra, "doğru renkleri mi kullandın?" diye
  kontrol eder. Tabloda kırmızı diyor ama kullanıcı yanlışlıkla siyah kullandıysa
  bu modül "S5.12: Kırmızı yerine Siyah" diye uyarır.

İki backend (renk sınıflandırıcı):
  - HSVBackend (varsayılan): klasik OpenCV, eğitim gerektirmez. Demo bunu kullanır.
  - HailoBackend (placeholder): Hailo-8L .hef ile inference. H5'te eklenecek.

API:
    backend = HSVBackend(palette_rgb)
    mismatches = check_active_row(frame, H_cam_to_chart, chart, active_row, backend)
    # mismatches: [(col, expected_idx, observed_idx, distance), ...]
"""

# ============================================================================
# İMPORTLAR
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass
# `Protocol` = "duck typing"in tipli versiyonu.
# Duck typing = "Eğer ördek gibi yürüyorsa ve ördek gibi vakvaklıyorsa, ördektir."
# Yani sınıf isim olarak Protocol'a inherit etmesi gerekmez, sadece imzası uysun.
# Java/C# interface'ine benzer ama daha gevşek (structural typing).
from typing import Protocol

import cv2
import numpy as np


# ============================================================================
# MODÜL SEVİYESİ SABİTLER
# ============================================================================
# Hücre boyutu (sample alırken).
CELL_PX = 12

# CIE-LAB uzayında ~25 birim üstü = göze çarpan fark.
# Just Noticeable Difference (JND) yaklaşık 2-3 birim, 25 oldukça yüksek.
# Pratikte BU SABIT KULLANILMIYOR (kontrolde sadece indeks farkına bakıyor).
# İleride threshold-based filtre eklenebilir.
LAB_DISTANCE_THRESHOLD = 25.0


# ============================================================================
# Protocol — "renk sınıflandırıcı" arabirimi
# ============================================================================
# Hem HSVBackend hem HailoBackend bu protokole uyar.
# Yeni backend eklemek istersen: bu metoda sahip ol, başka şart yok.
class ColorBackend(Protocol):
    def classify(self, samples_bgr: np.ndarray) -> np.ndarray:
        """Verilen N×3 örneği için palet indeksi (N,) döndür."""
        # `...` = Ellipsis. "İmplementasyon yok, sadece tip imzası" demek.
        # Bu protokol sınıfında metod gövdesi olmamalı, sadece imza.
        # Bu Python'da "abstract method" gibi davranır ama enforce edilmez.
        ...


# ============================================================================
# HSVBackend — klasik renk sınıflandırma (LAB uzayında en yakın komşu)
# ============================================================================
@dataclass
class HSVBackend:
    """LAB uzayında en yakın palet rengini bul (klasik, hızlı, eğitim gerektirmez)."""
    # NOT: Adı HSV ama aslında LAB kullanıyor (HSV erken denemelerden kaldı).
    # İsim değişikliği = bütün kodu kırmak demek, yorum yetiyor.
    palette_rgb: np.ndarray  # (k, 3) uint8

    def __post_init__(self):
        # ----- palette_rgb → palette_lab (önişlem) -----
        # palette_rgb'yi BGR'a çevir + cv2 uyumlu boyut yap (1, k, 3).
        #
        # Adım adım:
        #   self.palette_rgb shape: (k, 3)            — RGB renkler
        #   [:, ::-1]           → (k, 3) BGR'a çevir   (son ekseni tersle)
        #   .astype(np.uint8)   → uint8 (cv2 için)
        #   [None, :, :]        → ÖNE yeni eksen ekle. (k, 3) → (1, k, 3).
        #
        # NEDEN [None, :, :]?
        # cv2.cvtColor "resim" bekler (en az 2D, kanal axis ile 3D).
        # 1 satırlık 1 piksel'lik fake resim gibi yorumluyor.
        # `None` numpy'da `np.newaxis`'in kısaltması (yeni eksen ekler).
        bgr = self.palette_rgb[:, ::-1].astype(np.uint8)[None, :, :]

        # `cv2.COLOR_BGR2LAB` = BGR'den LAB'a renk uzayı dönüşümü.
        #
        # LAB renk uzayı NEDİR?
        # CIE LAB (1976) = algısal (perceptual) renk uzayı.
        #   L = lightness (parlaklık, 0-100 normalde, cv2'de 0-255)
        #   a = yeşil↔kırmızı eksen (negatif=yeşil, pozitif=kırmızı)
        #   b = mavi↔sarı eksen (negatif=mavi, pozitif=sarı)
        #
        # NEDEN LAB? (RGB yerine)
        # RGB'de iki rengin Öklid mesafesi (uzaklığı) gözle görünen farkı
        # YANSITMAZ. Mesela RGB'de 50 birim fark, kırmızı tonlarda büyük fark
        # gibi görünür ama mavi tonlarda küçük fark.
        # LAB'de Öklid mesafesi ≈ göze görünen renk farkı (perceptual uniformity).
        # Yani 5 birim LAB ≈ 50 birim LAB'ın 1/10'u kadar fark eder gözle.
        #
        # `[0]` = (1, k, 3) → (k, 3). İlk satırı al, fake boyutu kaldır.
        # `.astype(np.float32)` = float (mesafe hesabı için).
        #
        # `_palette_lab` _ ile başlıyor → "private" gelenek (dışarıdan dokunma).
        self._palette_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0].astype(np.float32)

    # ----- classify: ASIL İŞ -----
    def classify(self, samples_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Boş input için boş output. Hata fırlatma yerine sade davranış.
        if samples_bgr.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        # Samples (N, 3) → (N, 1, 3) → cvtColor için fake resim.
        # Yine boyut numarası: cvtColor 2D+ ister.
        lab = cv2.cvtColor(samples_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        # Sonra tekrar (N, 3) düzleştir (mesafe hesabı için).
        lab = lab.reshape(-1, 3).astype(np.float32)

        # ----- (N, k) MESAFE MATRİSİ — broadcasting ile vektörize -----
        #
        # BROADCASTING NUMARASI! ÇOK ÖNEMLİ TEKNİK.
        # Detay:
        #   lab shape: (N, 3)
        #   lab[:, None, :] shape: (N, 1, 3)  ← araya 1 boyutu eklendi
        #   self._palette_lab shape: (k, 3)
        #   self._palette_lab[None, :, :] shape: (1, k, 3)
        #
        # Çıkarma: (N, 1, 3) - (1, k, 3)
        # numpy "1 olan eksenleri tekrarla" mantığıyla → her ikisi de (N, k, 3) olur.
        # Sonuç: her örnek-her renk çifti için (3,) fark vektörü → toplam (N, k, 3).
        #
        # FOR DÖNGÜSÜZ HER ÖRNEĞİ HER RENGE KARŞILAŞTIRMAK = vektörize.
        # 1000 örnek × 4 renk = 4000 fark, hepsi tek satırda, C kodunda, çok hızlı.
        #
        # FOR İLE YAPSAK:
        #   for n in range(N):
        #     for k in range(K):
        #       diff[n, k] = lab[n] - palette[k]
        #   N=1000, K=4 → 4000 iterasyon, Python yavaş.
        diff = lab[:, None, :] - self._palette_lab[None, :, :]

        # `np.linalg.norm(arr, axis=...)` = vektör uzunluğu (öklid).
        # `axis=2` = son eksen üzerinden hesapla (3 LAB değerini birleştir).
        # Sonuç (N, k) — her örnek için her renge mesafe.
        # FORMÜL: sqrt(L² + a² + b²) for each (n, k).
        dist = np.linalg.norm(diff, axis=2)

        # `dist.argmin(axis=1)` = "her satırın MIN değerinin İNDEKSİ".
        # axis=1 = sütunlar üzerinde min.
        # Sonuç (N,) — her örnek için en yakın paletin indeksi.
        idx = dist.argmin(axis=1)

        # Mesafenin kendisi. Shape (N,).
        # min ve argmin ayrı çağrı — biraz tekrar ama okunaklı.
        # Optimize: `dist.min(axis=1)` yerine `np.take_along_axis(dist, idx[:, None], axis=1)`.
        min_dist = dist.min(axis=1)

        return idx, min_dist


# ============================================================================
# HailoBackend — yapay zeka çipiyle hızlandırılmış (placeholder)
# ============================================================================
@dataclass
class HailoBackend:
    """Hailo-8L .hef wrapper. H5'te implementasyon yapılacak; şimdilik HSV'ye düşer."""
    palette_rgb: np.ndarray
    # `str | None = None` = "ya string ya None, varsayılan None".
    hef_path: str | None = None

    def __post_init__(self):
        # Hailo kullanılamazsa HSV'yi yedek tut (graceful degradation).
        self._fallback = HSVBackend(self.palette_rgb)
        self._available = False
        try:
            # import hailo_platform  # noqa: F401
            # gerçek implementasyon: VDevice.create_from_hef(self.hef_path)
            #
            # PLACEHOLDER. Hailo SDK ileride buraya gelecek.
            # Şu an `_available = False`, hep HSV'ye düşüyor.
            if self.hef_path:
                pass
        except ImportError:
            # `pass` = "hiçbir şey yapma" placeholder.
            # ImportError yakalanmazsa modül import edemez → genel crash.
            # Yakaladığımız için sadece sessizce HSV'ye düşüyoruz.
            pass

    def classify(self, samples_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._available:
            return self._fallback.classify(samples_bgr)
        # `raise NotImplementedError(...)` = "bu metod henüz yazılmadı" deyimi.
        # Geliştiricilere "implement etmeden çağırma" sinyali.
        raise NotImplementedError("Hailo inference H5'te eklenecek")


# ============================================================================
# FONKSİYON: aktif sıranın renk örneklerini al
# ============================================================================
def sample_active_row(
    frame_bgr: np.ndarray,
    H_cam_to_chart: np.ndarray,
    rows: int,
    cols: int,
    active_row: int,
    cell_px: int = CELL_PX,
) -> np.ndarray:
    """Aktif sıranın N hücresinin merkez renklerini örnekle. (cols, 3) BGR döndür."""
    # Sınır dışıysa boş array. Caller bunu len(samples)==0 ile yakalıyor.
    # `np.empty((0, 3), ...)` = 0 satır × 3 sütun = boş ama doğru şekilde array.
    if not (0 <= active_row < rows):
        return np.empty((0, 3), dtype=np.uint8)

    # progress.warp ile aynı: kamerayı chart-piksele warp et.
    # Bu kod tekrarı bilinçli — sample_active_row bağımsız çalışabilsin diye.
    scale = np.array(
        [[cell_px, 0, 0], [0, cell_px, 0], [0, 0, 1]], dtype=np.float64,
    )
    M = scale @ np.asarray(H_cam_to_chart, dtype=np.float64)
    out_size = (cols * cell_px, rows * cell_px)
    warped = cv2.warpPerspective(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR)

    # Sadece aktif sıranın yatay şeridi.
    # `warped[start:end]` = ilk eksen (satır) için slicing.
    band = warped[active_row * cell_px:(active_row + 1) * cell_px]

    # Her sütun bloğunun merkez 50% alanının ortalama rengini al (kenar kontaminasyonunu azalt)
    # `//` = tam sayı bölme. 12 // 4 = 3 (5/2 = 2 tam sayı bölmesi).
    # Her hücreden 3 piksel kenardan boşluk:
    #   Kenarlarda ızgara çizgisi var (40,40,40 koyu gri), bulaşık.
    #   Komşu hücrelerin rengi taşmış olabilir warp'tan.
    # Orta %50 daha temiz örnek verir.
    pad = cell_px // 4

    samples = np.empty((cols, 3), dtype=np.uint8)
    for c in range(cols):
        # band[satır_aralığı, sütun_aralığı] dilim.
        # satır: pad'ten cell_px-pad'e (üst alt 3'er piksel kırpıldı).
        # sütun: hücrenin pad sağından, hücre sonu - pad'ine.
        # Yani hücrenin orta %50 kare kısmı.
        block = band[pad:cell_px - pad, c * cell_px + pad:(c + 1) * cell_px - pad]

        # block (h, w, 3) → reshape (-1, 3) → (h*w, 3).
        # `mean(axis=0)` → axis 0 üzerinden ortalama → (3,) — RGB ortalama.
        # Tek bir renk değeri: hücrenin "ana rengi".
        # `.astype(np.uint8)` = float → uint8 (renk tipi).
        samples[c] = block.reshape(-1, 3).mean(axis=0).astype(np.uint8)

    return samples


# ============================================================================
# FONKSİYON: hangi sıra "az önce tamamlandı"?
# ============================================================================
def last_completed_row(active_row: int, rows: int, direction: str) -> int | None:
    """Aktif sıranın bir önceki (yön bağımlı) tamamlanmış sırasını döndür."""
    if direction == "bottom_up":
        # bottom_up'ta aktif yukarı, yapılmış olan altta = +1.
        r = active_row + 1
    else:
        # top_down'da aktif aşağı, yapılmış olan üstte = -1.
        r = active_row - 1
    # Sınır içindeyse döndür, dışındaysa None.
    # ÖZET: aktif sıra dokumanın en başındaysa "az önce tamamlanmış" sıra YOK.
    return r if 0 <= r < rows else None


# ============================================================================
# FONKSİYON: aktif sıradaki hataları bul
# ============================================================================
def check_active_row(
    frame_bgr: np.ndarray,
    H_cam_to_chart: np.ndarray,
    chart,
    active_row: int,
    backend: ColorBackend,
    distance_threshold: float = LAB_DISTANCE_THRESHOLD,
) -> list[tuple[int, int, int, float]]:
    """
    Verilen sıranın renklerini palette ile karşılaştır.
    Döndürür: uyuşmayan hücrelerin listesi (col, expected_idx, observed_idx, lab_distance).
    """
    if not (0 <= active_row < chart.rows):
        return []

    samples = sample_active_row(
        frame_bgr, H_cam_to_chart, chart.rows, chart.cols, active_row,
    )
    if len(samples) == 0:
        return []

    # Backend ne dönerse: gözlemlenen palet indeksi + mesafe.
    # `observed_idx` shape (cols,), her hücrenin tahmin edilen renk indeksi.
    # `observed_dist` shape (cols,), o tahminin LAB mesafesi.
    observed_idx, observed_dist = backend.classify(samples)

    # Chart'taki o sıranın beklenen indeksleri (cols uzunluğunda dizi).
    expected_idx = chart.grid[active_row]

    # ----- Karşılaştır ve uyuşmayanları topla -----
    mismatches = []
    for c in range(chart.cols):
        if observed_idx[c] != expected_idx[c]:
            # 4'lü tuple: (sütun, beklenen, gözlemlenen, mesafe).
            # `int(...)` ve `float(...)` numpy scalar'larından Python primitivelerine çevirir.
            # Neden? JSON'a yazılabilir / düzgün print() olur / typing problemleri.
            #
            # NOT: distance_threshold parametresi tanımlı ama kullanılmamış!
            # İyileştirme alanı: çok düşük mesafede bile farklı indeks varsa
            # belki kabul edilebilir, eşikle filtreleme yapmıyoruz.
            mismatches.append(
                (c, int(expected_idx[c]), int(observed_idx[c]), float(observed_dist[c]))
            )
    return mismatches
