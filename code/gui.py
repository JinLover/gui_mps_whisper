import sounddevice
import numpy as np
import speech_recognition as sr
import whisper
import wave

from queue import Queue
from time import sleep

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QComboBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer

class TranscriptionThread(QThread):
    transcription_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()  # Thread finished signal

    def __init__(self, audio_model):
        super().__init__()
        self.audio_model = audio_model
        self.source = sr.Microphone(sample_rate=16000)
        self.running = True  # Flag to control loop state

    def run(self):
        data_queue = Queue()
        recorder = sr.Recognizer()
        recorder.energy_threshold = 1000
        recorder.dynamic_energy_threshold = False

        record_timeout = 2
        sample_rate = 16000

        audio_file_path = "live_audio.wav"
        wf = wave.open(audio_file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        full_transcription = ""

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            data_queue.put(data)
            wf.writeframes(data)

        with self.source:
            recorder.adjust_for_ambient_noise(self.source)
        stop_listening = recorder.listen_in_background(self.source, record_callback, phrase_time_limit=record_timeout)

        try:
            while self.running:
                if not data_queue.empty():
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()

                    if audio_data:
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                        result = self.audio_model.transcribe(audio_np, fp16=True)
                        text = result['text'].strip()

                        full_transcription += text
                        self.transcription_signal.emit(text)
                self.msleep(100)  # Prevent UI thread blocking

        except Exception as e:
            self.transcription_signal.emit(f"Error: {e}")
        finally:
            wf.close()
            stop_listening(wait_for_stop=False)  # Safely stop listener
            self.transcription_signal.emit("\nFinal Transcription:\n" + full_transcription)
            self.finished_signal.emit()  # Signal thread finished

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Transcription")
        self.setGeometry(100, 100, 600, 400)  # Set window size

        # Main layout
        main_layout = QVBoxLayout()

        # 1. Text output area (top)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText("Transcription will appear here...")
        main_layout.addWidget(self.text_edit, stretch=3)

        # 2. Start/Stop button in the center
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.toggle_transcription)
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout, stretch=1)

        # 3. Recording icon (right)
        icon_layout = QHBoxLayout()
        icon_layout.addStretch()
        self.recording_icon = QLabel(self)
        self.recording_icon.setPixmap(QPixmap())  # Initial state is empty
        self.recording_icon.setFixedSize(24, 24)
        self.recording_icon.setStyleSheet("""
            QLabel {
                background-color: red;
                border-radius: 12px;
            }
        """)
        self.recording_icon.setVisible(False)  # Initially hidden
        icon_layout.addWidget(self.recording_icon)
        main_layout.addLayout(icon_layout)

        # 4. Recording timer (bottom)
        self.timer_label = QLabel("Recording Time: 0s", self)
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.timer_label)

        # 5. Model selection dropdown
        self.model_label = QLabel("Select Model", self)
        self.model_combo_box = QComboBox(self)
        self.model_combo_box.addItem("base.en")
        self.model_combo_box.addItem("small.en")
        self.model_combo_box.addItem("medium.en")
        self.model_combo_box.addItem("large.en")  # Add more model options as needed
        self.model_combo_box.currentIndexChanged.connect(self.update_model)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combo_box)
        main_layout.addLayout(model_layout)

        # Set main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Timer settings
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.recording_time = 0

        # Transcription settings
        self.transcription_active = False
        self.model = whisper.load_model("base.en", device="mps")

    def update_model(self):
        model_name = self.model_combo_box.currentText()
        self.model = whisper.load_model(model_name, device="mps")

    def toggle_transcription(self):
        if self.transcription_active:
            self.stop_transcription()
        else:
            self.start_transcription()

    def start_transcription(self):
        self.transcription_active = True
        self.start_button.setText("Stop")
        self.recording_icon.setVisible(True)
        self.timer.start(1000)
        self.recording_time = 0

        self.transcription_thread = TranscriptionThread(self.model)
        self.transcription_thread.transcription_signal.connect(self.update_transcription)
        self.transcription_thread.finished_signal.connect(self.stop_transcription)
        self.transcription_thread.start()

    def stop_transcription(self):
        self.transcription_active = False
        self.start_button.setText("Start")
        self.recording_icon.setVisible(False)
        self.timer.stop()
        self.timer_label.setText("Recording Time: 0s")

        if hasattr(self, 'transcription_thread') and self.transcription_thread.isRunning():
            self.transcription_thread.terminate()

    def update_timer(self):
        self.recording_time += 1
        self.timer_label.setText(f"Recording Time: {self.recording_time}s")

    def update_transcription(self, text):
        self.text_edit.append(text)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()