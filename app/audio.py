"""Motife bağlı podcast oynatıcı — pygame.mixer.music sarmalayıcı.

Her motifin kendi mp3'ü vardır (eli_belinde / hayat_agaci). Kullanıcı program
çalışırken oynat/duraklat, ses +/− ve 30 sn geri/ileri yapabilir; motif değişince
yeni podcast'e geçilir ve o podcast KALDIĞI YERDEN sürer (her podcast kendi konumunu
hatırlar, çalma/durma durumu korunur).

Konum modeli (ölçümle doğrulandı):
    play(start=offset) → base = offset.
    get_pos() o play()'den beri geçen ms'i verir; pause'da DONAR, yeni play (seek)
    ile SIFIRLANIR, stop/bitişte -1 döner.
    → konum = base + max(0, get_pos())/1000.  Bu, ±30 sn ve mm:ss için yeterli.

Sağlamlık: pygame/mutagen yoksa ya da ses cihazı açılamazsa oynatıcı SESSİZCE devre
dışı kalır (enabled=False) — kontroller no-op olur, AR uygulaması çökmez. `--help`
ve ses cihazsız Raspberry Pi için kritik.
"""
from __future__ import annotations

import os
from pathlib import Path

# pygame import'undan ÖNCE: "Hello from the pygame community" banner'ını sustur.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except Exception:  # pygame yoksa/yüklenemezse ses tümden devre dışı
    pygame = None

try:
    from mutagen.mp3 import MP3
except Exception:  # mutagen yoksa süre bilinmez (mm:ss yerine yalnız geçen süre)
    MP3 = None


SEEK_SECONDS = 30.0   # ±atlama miktarı (kullanıcı isteği: 30 sn)
VOL_STEP = 0.1        # her ses +/− basışında değişim
VOL_DEFAULT = 0.7     # açılış ses düzeyi (0..1)


def fmt_time(seconds: float) -> str:
    """Saniyeyi mm:ss biçimine çevir (negatifi 0 say)."""
    s = max(0, int(round(seconds)))
    return f"{s // 60:d}:{s % 60:02d}"


class PodcastPlayer:
    """pygame.mixer.music üzerine motife bağlı podcast oynatıcı.

    tracks: {motif_adı: mp3 yolu}. autoplay: ilk motif yüklenince hemen çalsın mı.
    enabled: False ise (ya da mixer açılamazsa) tüm metotlar güvenli no-op'tur.
    """

    def __init__(self, tracks: dict, autoplay: bool = True,
                 volume: float = VOL_DEFAULT, enabled: bool = True):
        self.tracks = {k: Path(v) for k, v in tracks.items()}
        self.autoplay = autoplay
        self.volume = min(1.0, max(0.0, volume))
        self.enabled = False  # mixer açılırsa True olur

        self.motif: str | None = None
        self.loaded = False       # geçerli motifin mp3'ü gerçekten yüklendi mi
        self.base = 0.0           # son play(start=) ofseti (sn)
        self.paused = False       # kullanıcı duraklattı mı
        self.ended = False        # parça doğal olarak bitti mi
        self._positions: dict[str, float] = {}   # motif → bırakılan konum (sn)
        self._durations: dict[str, float | None] = {}  # motif → süre (sn) | None

        if enabled:
            self._init_mixer()

    # --- kurulum -----------------------------------------------------------
    def _init_mixer(self) -> None:
        """Mixer'ı aç; ses cihazı yoksa sessizce devre dışı kal."""
        if pygame is None:
            return
        try:
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            self.enabled = True
        except Exception:
            self.enabled = False

    # --- süre --------------------------------------------------------------
    def duration(self, motif: str | None = None) -> float | None:
        """Motifin mp3 süresi (sn). mutagen yoksa/okunamazsa None (lazy + cache)."""
        m = motif if motif is not None else self.motif
        if m is None:
            return None
        if m not in self._durations:
            d = None
            p = self.tracks.get(m)
            if MP3 is not None and p is not None and p.exists():
                try:
                    d = float(MP3(str(p)).info.length)
                except Exception:
                    d = None
            self._durations[m] = d
        return self._durations[m]

    # --- iç yardımcı -------------------------------------------------------
    def _start_at(self, offset: float, play: bool) -> None:
        """offset saniyesinden başlat; play=False ise hemen duraklatılmış kal.

        play(start=) parçayı ÇALMAYA başlatır; play=False istenince hemen pause()
        ederiz (ölçümle doğrulandı: play→pause çalışır, konum offset'te donar).
        """
        self.base = max(0.0, offset)
        self.ended = False
        # play() de (bozuk akış / sürücü hatası) nadiren fırlatabilir → sessizce bu
        # motifi devre dışı bırak, çökme. load() ve seek() bu yolu paylaşır.
        try:
            pygame.mixer.music.play(start=self.base)
        except Exception:
            self.loaded = False
            self.paused = True
            return
        if play:
            self.paused = False
        else:
            pygame.mixer.music.pause()
            self.paused = True

    # --- motif yükleme -----------------------------------------------------
    def set_motif(self, motif: str) -> None:
        """Aktif motifi değiştir: eski konumu sakla, yeni mp3'ü kaldığı yerden yükle.

        İlk çağrıda çalma durumu autoplay'e göre; sonraki çağrılarda ÖNCEKİ
        çalma/durma durumu korunur. Track dosyası yoksa sessizce durur.
        """
        if not self.enabled or motif == self.motif:
            return

        first = self.motif is None
        if first:
            play = self.autoplay
        else:
            # Eski konumu hatırla. Parça BİTTİYSE position() süreyi döndürür; o yüzden
            # 0.0 sakla → geri dönülünce baştan başlasın (sona sıkışıp kalmasın).
            self._positions[self.motif] = 0.0 if self.ended else self.position()
            play = self.is_playing()                        # durumu koru

        self.motif = motif
        p = self.tracks.get(motif)
        if p is None or not p.exists():
            # Bu motifin podcast'i yok → yüklenmedi; tüm kontroller no-op olur.
            self.loaded = False
            pygame.mixer.music.stop()
            self.base, self.paused, self.ended = 0.0, True, False
            return

        # Dosya var ama çözülemeyebilir (bozuk mp3 / donmuş ikilide kod çözücü yok).
        # Bu motifi sessiz bırak; uygulamayı ÇÖKERTME, sonraki sağlam motif yine çalar.
        try:
            pygame.mixer.music.load(str(p))
        except Exception:
            self.loaded = False
            pygame.mixer.music.stop()
            self.base, self.paused, self.ended = 0.0, True, False
            return
        self.loaded = True
        self._start_at(self._positions.get(motif, 0.0), play=play)

    # --- kullanıcı kontrolleri --------------------------------------------
    def toggle(self) -> None:
        """Oynat/Duraklat. Parça bittiyse baştan çalar."""
        if not self.enabled or not self.loaded:
            return
        if self.ended:
            self._start_at(0.0, play=True)
        elif self.paused:
            pygame.mixer.music.unpause()
            self.paused = False
        else:
            pygame.mixer.music.pause()
            self.paused = True

    def seek(self, delta: float) -> None:
        """Geçerli konumdan delta sn atla (−geri / +ileri). Durumu korur."""
        if not self.enabled or not self.loaded:
            return
        new = self.position() + delta
        new = max(0.0, new)
        d = self.duration()
        if d is not None:
            new = min(new, max(0.0, d - 1.0))  # bitime 1 sn kala sınırla
        keep_playing = not (self.paused and not self.ended)
        self._start_at(new, play=keep_playing)

    def set_volume(self, vol: float) -> None:
        """Ses düzeyini 0..1 aralığına sıkıştırıp uygula."""
        self.volume = min(1.0, max(0.0, round(vol, 2)))
        if self.enabled:
            pygame.mixer.music.set_volume(self.volume)

    def volume_up(self) -> None:
        self.set_volume(self.volume + VOL_STEP)

    def volume_down(self) -> None:
        self.set_volume(self.volume - VOL_STEP)

    # --- durum sorguları ---------------------------------------------------
    def position(self) -> float:
        """Geçerli çalma konumu (sn). Pause'da donar, bitişte süreye/baz'a sabitlenir."""
        if not self.enabled or not self.loaded:
            return 0.0
        pos_ms = pygame.mixer.music.get_pos()
        if pos_ms < 0:  # stop/bitiş
            d = self.duration()
            return d if (self.ended and d is not None) else self.base
        return self.base + pos_ms / 1000.0

    def is_playing(self) -> bool:
        """Şu an aktif çalıyor mu (duraklatılmamış ve bitmemiş)."""
        return self.enabled and self.loaded and not self.paused and not self.ended

    def state_icon(self) -> str:
        """Panelde gösterilecek durum simgesi."""
        if not self.enabled or not self.loaded:
            return "⏹"
        if self.ended:
            return "⏹"
        return "▶" if not self.paused else "⏸"

    def update(self) -> None:
        """Her karede çağrılır: parçanın doğal bitişini yakala."""
        if not self.enabled or not self.loaded or self.paused or self.ended:
            return
        # get_busy() False VE get_pos() -1 → akış bitti (anlık yanlış-negatif olmasın diye ikisi de).
        if not pygame.mixer.music.get_busy() and pygame.mixer.music.get_pos() < 0:
            self.ended = True
            d = self.duration()
            if d is not None:
                self.base = d
            self._positions[self.motif] = 0.0  # bitti → sonraki sefere baştan

    def close(self) -> None:
        """Çıkışta sesi durdur ve mixer'ı kapat."""
        if not self.enabled or pygame is None:
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass
