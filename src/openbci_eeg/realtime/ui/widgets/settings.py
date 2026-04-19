"""Settings tab — audio device selection and other configuration.

Stores preferences in a QSettings instance so they persist across sessions.
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets, QtMultimedia

_BG = QtGui.QColor("#0a0e14")
_TEXT_PRIMARY = QtGui.QColor("#c8d8e8")
_TEXT_SECONDARY = QtGui.QColor("#889aaa")

# QSettings key
_SETTINGS_ORG = "QDNU"
_SETTINGS_APP = "openbci-lattice"
_KEY_AUDIO_DEVICE = "audio/output_device_id"


def load_preferred_audio_device() -> Optional[QtMultimedia.QAudioDevice]:
    """Load the saved audio device preference, or return None for system default."""
    settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    saved_id = settings.value(_KEY_AUDIO_DEVICE, None)
    if saved_id is None:
        return None

    if isinstance(saved_id, str):
        saved_id = saved_id.encode()

    for dev in QtMultimedia.QMediaDevices.audioOutputs():
        if dev.id() == saved_id:
            return dev
    return None


def get_audio_device() -> QtMultimedia.QAudioDevice:
    """Get the preferred audio device, falling back to system default."""
    preferred = load_preferred_audio_device()
    if preferred is not None:
        return preferred
    return QtMultimedia.QMediaDevices.defaultAudioOutput()


class SettingsWidget(QtWidgets.QWidget):
    """Settings tab with audio device picker."""

    audio_device_changed = QtCore.Signal(object)  # emits QAudioDevice

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, _BG)
        self.setPalette(pal)

        self._settings = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Title
        title = QtWidgets.QLabel("Settings")
        title.setStyleSheet("color: #c8d8e8; font-size: 18pt; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        main_layout.addSpacing(20)

        # --- Audio section ---
        audio_header = QtWidgets.QLabel("Audio Output")
        audio_header.setStyleSheet(
            "color: #c8d8e8; font-size: 13pt; font-weight: bold;"
        )
        main_layout.addWidget(audio_header)

        audio_desc = QtWidgets.QLabel(
            "Select the audio device for paradigm cues (eyes open/closed tones)."
        )
        audio_desc.setStyleSheet("color: #889aaa; font-size: 10pt;")
        audio_desc.setWordWrap(True)
        main_layout.addWidget(audio_desc)

        # Device dropdown
        device_layout = QtWidgets.QHBoxLayout()
        device_label = QtWidgets.QLabel("Device:")
        device_label.setStyleSheet("color: #c8d8e8; font-size: 11pt;")
        device_label.setFixedWidth(60)
        device_layout.addWidget(device_label)

        self._device_combo = QtWidgets.QComboBox()
        self._device_combo.setStyleSheet("""
            QComboBox {
                background: #12161e; color: #c8d8e8; border: 1px solid #333;
                border-radius: 4px; padding: 6px 12px; font-size: 11pt;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #12161e; color: #c8d8e8; border: 1px solid #333;
                selection-background-color: #4a9eff;
            }
        """)
        device_layout.addWidget(self._device_combo, stretch=1)
        main_layout.addLayout(device_layout)

        # Test button
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self._test_btn = QtWidgets.QPushButton("Test Sound")
        self._test_btn.setFixedSize(140, 40)
        self._test_btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff; color: white; font-size: 11pt;
                font-weight: bold; border: none; border-radius: 6px;
            }
            QPushButton:hover { background: #3a8eef; }
        """)
        self._test_btn.clicked.connect(self._test_sound)
        btn_layout.addWidget(self._test_btn)

        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # Status label
        self._status = QtWidgets.QLabel("")
        self._status.setStyleSheet("color: #34C759; font-size: 10pt;")
        self._status.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self._status)

        main_layout.addStretch()

        # Populate devices
        self._devices: list[QtMultimedia.QAudioDevice] = []
        self._populate_devices()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)

    def _populate_devices(self) -> None:
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._devices = []

        saved_id = self._settings.value(_KEY_AUDIO_DEVICE, None)
        if isinstance(saved_id, str):
            saved_id = saved_id.encode()

        default_dev = QtMultimedia.QMediaDevices.defaultAudioOutput()

        # Add system default as first option
        self._device_combo.addItem(f"System Default ({default_dev.description()})")
        self._devices.append(default_dev)
        selected_idx = 0

        for dev in QtMultimedia.QMediaDevices.audioOutputs():
            self._device_combo.addItem(dev.description())
            self._devices.append(dev)
            if saved_id is not None and dev.id() == saved_id:
                selected_idx = len(self._devices) - 1

        self._device_combo.setCurrentIndex(selected_idx)
        self._device_combo.blockSignals(False)

    def _on_device_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._devices):
            return
        dev = self._devices[index]
        if index == 0:
            # System default — clear saved preference
            self._settings.remove(_KEY_AUDIO_DEVICE)
        else:
            self._settings.setValue(_KEY_AUDIO_DEVICE, dev.id().data())
        self._status.setText(f"Selected: {dev.description()}")
        self.audio_device_changed.emit(dev)

    def _test_sound(self) -> None:
        """Play a short test tone on the selected device."""
        import math
        import struct

        idx = self._device_combo.currentIndex()
        if idx < 0 or idx >= len(self._devices):
            return
        dev = self._devices[idx]

        fmt = QtMultimedia.QAudioFormat()
        fmt.setSampleRate(44100)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.Int16)

        # Generate a short chirp (440 -> 880 Hz, 300ms)
        sr = 44100
        duration = 0.3
        n = int(sr * duration)
        samples = []
        for i in range(n):
            t = i / sr
            freq = 440 + (880 - 440) * (t / duration)
            fade = 1.0
            fade_n = int(sr * 0.02)
            if i < fade_n:
                fade = i / fade_n
            elif i > n - fade_n:
                fade = (n - i) / fade_n
            val = 0.5 * fade * math.sin(2 * math.pi * freq * t)
            samples.append(int(val * 32767))
        pcm = struct.pack(f"<{len(samples)}h", *samples)

        try:
            self._audio_sink = QtMultimedia.QAudioSink(dev, fmt)
            buf = QtCore.QBuffer()
            buf.setData(pcm)
            buf.open(QtCore.QIODevice.ReadOnly)
            buf.setParent(self._audio_sink)
            self._audio_sink.start(buf)
            self._status.setText(f"Playing on: {dev.description()}")
            self._status.setStyleSheet("color: #34C759; font-size: 10pt;")
        except Exception as e:
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet("color: #FF453A; font-size: 10pt;")

    def get_selected_device(self) -> QtMultimedia.QAudioDevice:
        """Return the currently selected audio device."""
        idx = self._device_combo.currentIndex()
        if 0 <= idx < len(self._devices):
            return self._devices[idx]
        return QtMultimedia.QMediaDevices.defaultAudioOutput()
