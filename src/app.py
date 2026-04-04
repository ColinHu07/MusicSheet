"""
MP3 → Sheet Music  —  PySide6 desktop application.

Supports:
  - Local audio file transcription
  - YouTube search and audio fetch
  - Multiple monophonic instruments (violin, flute, viola, cello)
  - New melody-path resolver with configurable settings

Run with:  python -m src.app
"""

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.pipeline import run_pipeline
from src.pipeline.utils import TranscriptionConfig, TranscriptionResult, setup_logging

logger = logging.getLogger("mp3_to_sheet")


# =========================================================================== #
# Worker threads
# =========================================================================== #

class PipelineWorker(QThread):
    """Executes the transcription pipeline in a background thread."""

    log_signal = Signal(str)
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, config: TranscriptionConfig):
        super().__init__()
        self.config = config

    def run(self):
        handler = _SignalLogHandler(self.log_signal)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger("mp3_to_sheet")
        root.addHandler(handler)

        try:
            result = run_pipeline(self.config, progress_cb=self.log_signal.emit)
            self.finished_signal.emit(result)
        except Exception as exc:
            self.error_signal.emit(str(exc))
        finally:
            root.removeHandler(handler)


class YouTubeSearchWorker(QThread):
    """Searches YouTube in a background thread."""

    results_signal = Signal(list)
    error_signal = Signal(str)

    def __init__(self, query: str, max_results: int = 8):
        super().__init__()
        self.query = query
        self.max_results = max_results

    def run(self):
        try:
            from src.providers.youtube_provider import YouTubeProvider
            provider = YouTubeProvider()
            if not provider.is_available():
                self.error_signal.emit(
                    "yt-dlp not found. Install it:\n  pip install yt-dlp"
                )
                return
            results = provider.search(self.query, self.max_results)
            self.results_signal.emit(results)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class YouTubeFetchWorker(QThread):
    """Downloads YouTube audio in a background thread."""

    log_signal = Signal(str)
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, url: str, output_dir: str = "out/cache/youtube"):
        super().__init__()
        self.url = url
        self.output_dir = output_dir

    def run(self):
        try:
            from src.providers.youtube_provider import YouTubeProvider
            provider = YouTubeProvider()
            self.log_signal.emit(f"Downloading audio from: {self.url}")
            result = provider.fetch_audio(self.url, self.output_dir)
            self.log_signal.emit(f"Download complete: {result.title}")
            self.finished_signal.emit(result)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class _SignalLogHandler(logging.Handler):
    """Routes log records into a Qt signal."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def emit(self, record):
        try:
            msg = self.format(record)
            self._signal.emit(msg)
        except RuntimeError:
            pass


# =========================================================================== #
# Main window
# =========================================================================== #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MP3 → Sheet Music")
        self.setMinimumSize(780, 800)
        self._result: Optional[TranscriptionResult] = None
        self._worker: Optional[PipelineWorker] = None
        self._yt_search_worker: Optional[YouTubeSearchWorker] = None
        self._yt_fetch_worker: Optional[YouTubeFetchWorker] = None
        self._yt_results: List = []
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Source tabs (Local File / YouTube) ----------------------------
        self.source_tabs = QTabWidget()

        # Tab 1: Local file
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Audio file:"))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select MP3 / WAV / FLAC / OGG / M4A …")
        file_row.addWidget(self.file_edit, stretch=1)
        browse_btn = QPushButton("Browse …")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(browse_btn)
        local_layout.addLayout(file_row)
        self.source_tabs.addTab(local_tab, "Local File")

        # Tab 2: YouTube
        yt_tab = QWidget()
        yt_layout = QVBoxLayout(yt_tab)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.yt_search_edit = QLineEdit()
        self.yt_search_edit.setPlaceholderText("Song name, artist, or paste YouTube URL …")
        self.yt_search_edit.returnPressed.connect(self._on_yt_search)
        search_row.addWidget(self.yt_search_edit, stretch=1)
        self.yt_search_btn = QPushButton("Search")
        self.yt_search_btn.clicked.connect(self._on_yt_search)
        search_row.addWidget(self.yt_search_btn)
        yt_layout.addLayout(search_row)

        self.yt_results_list = QListWidget()
        self.yt_results_list.setMaximumHeight(160)
        yt_layout.addWidget(self.yt_results_list)

        self.yt_status_label = QLabel("")
        yt_layout.addWidget(self.yt_status_label)

        self.source_tabs.addTab(yt_tab, "YouTube")
        root.addWidget(self.source_tabs)

        # --- Instrument ----------------------------------------------------
        inst_row = QHBoxLayout()
        inst_row.addWidget(QLabel("Instrument:"))
        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems([
            "Violin",
            "Flute",
            "Viola",
            "Cello",
            "Piano (coming soon)",
            "Drums (coming soon)",
        ])
        for i in range(4, self.instrument_combo.count()):
            self.instrument_combo.model().item(i).setEnabled(False)
        inst_row.addWidget(self.instrument_combo)
        inst_row.addStretch()
        root.addLayout(inst_row)

        # --- Melody mode / Time sig / Key ---------------------------------
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Melody mode:"))
        self.melody_mode_combo = QComboBox()
        self.melody_mode_combo.addItems([
            "Predominant melody",
            "Upper melody bias",
            "Conservative (smoother)",
        ])
        grid_row.addWidget(self.melody_mode_combo)

        grid_row.addSpacing(15)
        grid_row.addWidget(QLabel("Meter:"))
        self.ts_combo = QComboBox()
        self.ts_combo.addItems(["Auto (4/4)", "4/4", "3/4", "6/8", "2/4"])
        grid_row.addWidget(self.ts_combo)

        grid_row.addSpacing(15)
        grid_row.addWidget(QLabel("Key:"))
        self.key_combo = QComboBox()
        self.key_combo.addItems([
            "Auto", "C", "G", "D", "A", "E", "B", "F#",
            "F", "Bb", "Eb", "Ab", "Db",
            "A minor", "E minor", "B minor", "D minor", "G minor",
        ])
        grid_row.addWidget(self.key_combo)
        grid_row.addStretch()
        root.addLayout(grid_row)

        # --- Checkboxes / duration -----------------------------------------
        chk_row = QHBoxLayout()
        self.simplify_chk = QCheckBox("Simplify rhythm (1/8)")
        chk_row.addWidget(self.simplify_chk)
        self.triplets_chk = QCheckBox("Allow triplets")
        chk_row.addWidget(self.triplets_chk)
        self.slurs_chk = QCheckBox("Add phrase slurs")
        chk_row.addWidget(self.slurs_chk)
        chk_row.addStretch()
        root.addLayout(chk_row)

        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Max duration (sec):"))
        self.max_dur_spin = QSpinBox()
        self.max_dur_spin.setRange(5, 600)
        self.max_dur_spin.setValue(45)
        dur_row.addWidget(self.max_dur_spin)

        dur_row.addSpacing(15)
        dur_row.addWidget(QLabel("Start at (sec):"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 3600)
        self.start_spin.setValue(0)
        dur_row.addWidget(self.start_spin)

        dur_row.addStretch()
        root.addLayout(dur_row)

        # --- Advanced settings group ---------------------------------------
        adv_group = QGroupBox("Advanced settings")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_layout = QVBoxLayout(adv_group)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Min note (ms):"))
        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(20, 500)
        self.min_dur_spin.setValue(100)
        r1.addWidget(self.min_dur_spin)

        r1.addSpacing(15)
        r1.addWidget(QLabel("Min confidence:"))
        self.min_conf_spin = QDoubleSpinBox()
        self.min_conf_spin.setRange(0.0, 1.0)
        self.min_conf_spin.setSingleStep(0.05)
        self.min_conf_spin.setValue(0.3)
        r1.addWidget(self.min_conf_spin)
        r1.addStretch()
        adv_layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Onset threshold:"))
        self.onset_spin = QDoubleSpinBox()
        self.onset_spin.setRange(0.0, 1.0)
        self.onset_spin.setSingleStep(0.05)
        self.onset_spin.setValue(0.5)
        r2.addWidget(self.onset_spin)
        r2.addWidget(QLabel("Frame threshold:"))
        self.frame_spin = QDoubleSpinBox()
        self.frame_spin.setRange(0.0, 1.0)
        self.frame_spin.setSingleStep(0.05)
        self.frame_spin.setValue(0.3)
        r2.addWidget(self.frame_spin)
        r2.addStretch()
        adv_layout.addLayout(r2)

        # Advanced toggles
        tog_row = QHBoxLayout()
        self.octave_smooth_chk = QCheckBox("Octave smoothing")
        self.octave_smooth_chk.setChecked(True)
        tog_row.addWidget(self.octave_smooth_chk)

        self.debug_chk = QCheckBox("Save debug artifacts")
        self.debug_chk.setChecked(True)
        tog_row.addWidget(self.debug_chk)

        self.legacy_chk = QCheckBox("Use legacy pipeline")
        tog_row.addWidget(self.legacy_chk)
        tog_row.addStretch()
        adv_layout.addLayout(tog_row)

        r4 = QHBoxLayout()
        r4.addWidget(QLabel("MuseScore path:"))
        self.mscore_edit = QLineEdit()
        self.mscore_edit.setPlaceholderText("(auto-detect)")
        r4.addWidget(self.mscore_edit, stretch=1)
        mscore_btn = QPushButton("Browse …")
        mscore_btn.clicked.connect(self._browse_musescore)
        r4.addWidget(mscore_btn)
        adv_layout.addLayout(r4)

        root.addWidget(adv_group)

        # --- Generate button ------------------------------------------------
        self.gen_btn = QPushButton("  ▶  Generate Sheet Music  ")
        self.gen_btn.setStyleSheet(
            "QPushButton { font-size: 14px; padding: 8px; }"
        )
        self.gen_btn.clicked.connect(self._on_generate)
        root.addWidget(self.gen_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Log area -------------------------------------------------------
        root.addWidget(QLabel("Log:"))
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        root.addWidget(self.log_edit, stretch=1)

        # --- Output buttons -------------------------------------------------
        out_row = QHBoxLayout()
        self.open_xml_btn = QPushButton("Open MusicXML")
        self.open_xml_btn.setEnabled(False)
        self.open_xml_btn.clicked.connect(self._open_musicxml)
        out_row.addWidget(self.open_xml_btn)

        self.open_pdf_btn = QPushButton("Open PDF")
        self.open_pdf_btn.setEnabled(False)
        self.open_pdf_btn.clicked.connect(self._open_pdf)
        out_row.addWidget(self.open_pdf_btn)

        self.open_folder_btn = QPushButton("Open output folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_folder)
        out_row.addWidget(self.open_folder_btn)
        root.addLayout(out_row)

    # ------------------------------------------------------------------ #
    # Slots — file browsing
    # ------------------------------------------------------------------ #
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio files (*.mp3 *.wav *.flac *.ogg *.m4a);;All files (*)",
        )
        if path:
            self.file_edit.setText(path)

    def _browse_musescore(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate MuseScore executable", "",
            "Executables (*.exe);;All files (*)",
        )
        if path:
            self.mscore_edit.setText(path)

    # ------------------------------------------------------------------ #
    # Slots — YouTube
    # ------------------------------------------------------------------ #
    def _on_yt_search(self):
        query = self.yt_search_edit.text().strip()
        if not query:
            return

        self.yt_search_btn.setEnabled(False)
        self.yt_status_label.setText("Searching …")
        self.yt_results_list.clear()
        self._yt_results = []

        self._yt_search_worker = YouTubeSearchWorker(query)
        self._yt_search_worker.results_signal.connect(self._on_yt_search_done)
        self._yt_search_worker.error_signal.connect(self._on_yt_search_error)
        self._yt_search_worker.start()

    def _on_yt_search_done(self, results):
        self.yt_search_btn.setEnabled(True)
        self._yt_results = results

        if not results:
            self.yt_status_label.setText("No results found.")
            return

        self.yt_status_label.setText(f"{len(results)} results")
        for r in results:
            dur = f" [{r.duration_display}]" if r.duration_display else ""
            channel = f" — {r.channel}" if r.channel else ""
            text = f"{r.title}{channel}{dur}"
            self.yt_results_list.addItem(text)

    def _on_yt_search_error(self, msg):
        self.yt_search_btn.setEnabled(True)
        self.yt_status_label.setText(f"Error: {msg}")

    # ------------------------------------------------------------------ #
    # Slots — generate
    # ------------------------------------------------------------------ #
    def _on_generate(self):
        input_path = None
        source_url = ""

        # Check which tab is active
        if self.source_tabs.currentIndex() == 0:
            # Local file tab
            path = self.file_edit.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "No file", "Please select a valid audio file.")
                return
            input_path = path
        else:
            # YouTube tab
            selected = self.yt_results_list.currentRow()
            query = self.yt_search_edit.text().strip()

            if query.startswith("http"):
                self._start_yt_fetch_and_transcribe(query)
                return
            elif selected >= 0 and selected < len(self._yt_results):
                url = self._yt_results[selected].url
                self._start_yt_fetch_and_transcribe(url)
                return
            else:
                QMessageBox.warning(
                    self, "No selection",
                    "Please search and select a YouTube result, or paste a URL."
                )
                return

        self._start_pipeline(input_path)

    def _start_yt_fetch_and_transcribe(self, url: str):
        self.log_edit.clear()
        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("  ⏳  Downloading …  ")
        self._append_log(f"Fetching audio from YouTube: {url}")

        self._yt_fetch_worker = YouTubeFetchWorker(url)
        self._yt_fetch_worker.log_signal.connect(self._append_log)
        self._yt_fetch_worker.finished_signal.connect(self._on_yt_fetch_done)
        self._yt_fetch_worker.error_signal.connect(self._on_yt_fetch_error)
        self._yt_fetch_worker.start()

    def _on_yt_fetch_done(self, fetch_result):
        self._append_log(f"Audio ready: {fetch_result.audio_path}")
        self._start_pipeline(fetch_result.audio_path)

    def _on_yt_fetch_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("  ▶  Generate Sheet Music  ")
        self._append_log(f"\n❌ YouTube download error: {msg}")
        QMessageBox.critical(self, "Download error", msg)

    def _start_pipeline(self, input_path: str):
        ts_text = self.ts_combo.currentText()
        ts_val = "auto" if ts_text.startswith("Auto") else ts_text

        key_text = self.key_combo.currentText()
        key_val = "auto" if key_text == "Auto" else key_text

        instrument = self.instrument_combo.currentText().lower().split()[0]

        config = TranscriptionConfig(
            input_path=input_path,
            max_duration_s=self.max_dur_spin.value(),
            time_signature=ts_val,
            key_signature=key_val,
            instrument=instrument,
            allow_triplets=self.triplets_chk.isChecked(),
            simplify_rhythm=self.simplify_chk.isChecked(),
            add_phrase_slurs=self.slurs_chk.isChecked(),
            min_note_duration_ms=self.min_dur_spin.value(),
            min_confidence=self.min_conf_spin.value(),
            onset_threshold=self.onset_spin.value(),
            frame_threshold=self.frame_spin.value(),
            use_melody_resolver=not self.legacy_chk.isChecked(),
            save_debug_artifacts=self.debug_chk.isChecked(),
            musescore_path=self.mscore_edit.text().strip(),
        )

        self.log_edit.clear()
        self.open_xml_btn.setEnabled(False)
        self.open_pdf_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("  ⏳  Processing …  ")
        self._result = None

        self._worker = PipelineWorker(config)
        self._worker.log_signal.connect(self._append_log)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _append_log(self, msg: str):
        self.log_edit.append(msg)
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, result: TranscriptionResult):
        self._result = result
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("  ▶  Generate Sheet Music  ")

        if result.musicxml_path and os.path.isfile(result.musicxml_path):
            self.open_xml_btn.setEnabled(True)

        if result.pdf_path and os.path.isfile(result.pdf_path):
            self.open_pdf_btn.setEnabled(True)

        self.open_folder_btn.setEnabled(True)

        self._append_log("")
        self._append_log(f"═══ Summary ═══")
        self._append_log(f"  Tempo : {result.tempo_bpm} BPM")
        self._append_log(f"  Key   : {result.detected_key}")
        self._append_log(f"  Notes : {result.num_notes}")
        self._append_log(f"  Measures : {result.num_measures}")
        if result.melody_stats:
            self._append_log(f"  Melody candidates : {result.melody_stats.get('total_candidates', '?')}")
            self._append_log(f"  After cleanup : {result.melody_stats.get('after_cleanup', '?')}")
        if result.warnings:
            for w in result.warnings:
                self._append_log(f"  ⚠ {w}")

    def _on_error(self, msg: str):
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("  ▶  Generate Sheet Music  ")
        self._append_log(f"\n❌ ERROR: {msg}")
        QMessageBox.critical(self, "Pipeline error", msg)

    def _open_musicxml(self):
        if self._result and self._result.musicxml_path:
            _open_file(self._result.musicxml_path)

    def _open_pdf(self):
        if self._result and self._result.pdf_path:
            _open_file(self._result.pdf_path)

    def _open_folder(self):
        _open_file("out")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _open_file(path: str):
    """Open a file or folder with the OS default handler."""
    path = os.path.abspath(path)
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        logger.error("Cannot open %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    setup_logging(logging.DEBUG)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
