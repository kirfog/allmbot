#!/usr/bin/env python3
import logging
import sys
import time

import numpy as np
import psutil
import sounddevice as sd
from llmbot import BotActor
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QProgressBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger()


class BotWorker(QThread):
    chat_signal = Signal(str, str)

    def __init__(self, bot_instance):
        super().__init__()
        self.bot = bot_instance

    def run(self):
        while not self.isInterruptionRequested():
            # Process incoming audio from the stream
            while self.bot.recognizer.is_ready(self.bot.stt_stream):
                self.bot.recognizer.decode_stream(self.bot.stt_stream)

            if self.bot.recognizer.is_endpoint(self.bot.stt_stream):
                result_obj = self.bot.recognizer.get_result(self.bot.stt_stream)
                result = (
                    result_obj.text.strip()
                    if hasattr(result_obj, "text")
                    else str(result_obj).strip()
                )

                if result:
                    self.chat_signal.emit("User", result)
                    # This blocks THIS thread, but NOT the MonitorWorker
                    thought = self.bot.think(result)
                    self.chat_signal.emit("Bot", thought)
                    self.bot.speak(thought)

                self.bot.recognizer.reset(self.bot.stt_stream)
            self.msleep(10)


class MonitorWorker(QThread):
    volume_signal = Signal(float)
    cpu_signal = Signal(float)

    def __init__(self, bot_instance):
        super().__init__()
        self.bot = bot_instance

    def run(self):
        psutil.cpu_percent(interval=None)
        last_cpu_check = time.time()

        def audio_callback(indata, frames, time_info, status):
            rms = np.sqrt(np.mean(indata**2))
            self.volume_signal.emit(rms * 100)

            self.bot.stt_stream.accept_waveform(
                self.bot.sample_rate, indata.copy().flatten()
            )

        with sd.InputStream(
            channels=1,
            samplerate=self.bot.sample_rate,
            callback=audio_callback,
            dtype="float32",
        ):
            while not self.isInterruptionRequested():
                current_time = time.time()

                if current_time - last_cpu_check >= 0.5:
                    cpu_val = psutil.cpu_percent(interval=None)
                    self.cpu_signal.emit(cpu_val)
                    last_cpu_check = current_time

                self.msleep(100)


class BotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dialogue")
        self.resize(500, 600)

        # 1. Chat History Widget
        self.chat_history = QTextBrowser()
        self.chat_history.setAcceptRichText(True)
        self.chat_history.setPlaceholderText("Listening for speech...")

        # 2. Progress Bar
        self.mic_label = QLabel("Mic level: 0%")
        self.mic_bar = QProgressBar()
        self.mic_bar.setRange(0, 100)
        self.mic_bar.setTextVisible(False)
        self.mic_bar.setStyleSheet("QProgressBar::chunk { background-color: #05B8CC; }")

        container = QWidget()
        self.setCentralWidget(container)

        layout = QVBoxLayout(container)
        layout.addWidget(self.chat_history)
        layout.addWidget(self.mic_label)
        layout.addWidget(self.mic_bar)

        # CPU Progress Bar Setup
        self.cpu_label = QLabel("CPU Load: 0%")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #f39c12; }")

        layout.addWidget(self.cpu_label)
        layout.addWidget(self.cpu_bar)

        self.bot = BotActor()

        self.monitor = MonitorWorker(self.bot)
        self.processor = BotWorker(self.bot)
        self.monitor.volume_signal.connect(self.update_volume)
        self.monitor.cpu_signal.connect(self.update_cpu)
        self.processor.chat_signal.connect(self.update_chat)

        self.monitor.start()
        self.processor.start()

    @Slot(float)
    def update_volume(self, value):
        self.mic_bar.setValue(min(100, int(value * 5)))
        self.mic_label.setText(f"Mic level: {value:.1f}%")

    @Slot(str, str)
    def update_chat(self, role, message):
        color = "#0078D7" if role == "User" else "#28A745"
        formatted_msg = f'<p><b style="color: {color};">{role}:</b> {message}</p>'
        self.chat_history.append(formatted_msg)
        # Ensure scroll stays at the bottom
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        )

    @Slot(float)
    def update_cpu(self, value):
        self.cpu_bar.setValue(int(value))
        self.cpu_label.setText(f"CPU Load: {value:.1f}%")

    def closeEvent(self, event):
        self.processor.requestInterruption()
        self.monitor.requestInterruption()
        self.processor.quit()
        self.monitor.quit()

        if not self.processor.wait(2000):
            print("Processor thread timed out, forcing termination.")
            self.processor.terminate()

        if not self.monitor.wait(2000):
            print("Monitor thread timed out, forcing termination.")
            self.monitor.terminate()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BotWindow()
    window.show()
    sys.exit(app.exec())
