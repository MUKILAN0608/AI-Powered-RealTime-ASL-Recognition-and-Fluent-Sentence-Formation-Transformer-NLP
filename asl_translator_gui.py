#!/usr/bin/env python3
"""
Enhanced ASL Translator GUI with Ollama Integration
Features: Real-time translation, loading states, error handling, and beautiful UI
"""

import sys
import time
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                              QProgressBar, QFrame, QScrollArea, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon
import ollama

class TranslationWorker(QObject):
    """Worker thread for Ollama API calls"""
    translation_complete = pyqtSignal(str)
    translation_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        
    def translate_phrase(self, asl_input):
        """Translate ASL phrase using Ollama"""
        if self.is_running:
            return
            
        self.is_running = True
        
        try:
            # System instruction for consistent style
            system_prompt = (
                "You are an advanced ASL predictor and sentence stylist. "
                "For each input phrase, produce one short, elegant, and grammatically perfect English sentence "
                "that faithfully represents the intended meaning. "
                "Prefer natural phrasing, fluid word order, and polished sentence-level punctuation. "
                "Keep each output concise (one sentence) and beautiful."
            )
            
            # Call Ollama API
            response = ollama.chat(
                model="llama3.2:3b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": asl_input}
                ]
            )
            
            # Extract translation
            translation = response["message"]["content"].strip()
            self.translation_complete.emit(translation)
            
        except Exception as e:
            error_msg = f"Translation error: {str(e)}"
            self.translation_error.emit(error_msg)
        finally:
            self.is_running = False

class ASLTranslatorGUI(QMainWindow):
    """Main ASL Translator GUI Window"""
    
    def __init__(self):
        super().__init__()
        self.translation_worker = TranslationWorker()
        self.worker_thread = QThread()
        self.translation_worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.translation_worker.translation_complete.connect(self.on_translation_complete)
        self.translation_worker.translation_error.connect(self.on_translation_error)
        
        # Start worker thread
        self.worker_thread.start()
        
        self.init_ui()
        self.setup_styles()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Enhanced ASL Translator with Ollama")
        self.setGeometry(100, 100, 900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_frame = self.create_header()
        main_layout.addWidget(header_frame)
        
        # Input section
        input_frame = self.create_input_section()
        main_layout.addWidget(input_frame)
        
        # Translation section
        translation_frame = self.create_translation_section()
        main_layout.addWidget(translation_frame)
        
        # History section
        history_frame = self.create_history_section()
        main_layout.addWidget(history_frame)
        
        # Control buttons
        control_frame = self.create_control_section()
        main_layout.addWidget(control_frame)
        
        # Status bar
        self.status_label = QLabel("Ready to translate ASL phrases")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Initialize variables
        self.translation_history = []
        self.is_translating = False
        
    def create_header(self):
        """Create the header section"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        header_layout = QVBoxLayout(header_frame)
        
        # Title
        title_label = QLabel("🎯 Enhanced ASL Translator")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        
        # Subtitle
        subtitle_label = QLabel("Powered by Ollama LLM - Transform ASL to Perfect English")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #7f8c8d; margin: 5px;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        
        return header_frame
        
    def create_input_section(self):
        """Create the input section"""
        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        input_layout = QVBoxLayout(input_frame)
        
        # Input label
        input_label = QLabel("📝 Enter ASL Phrase:")
        input_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        input_label.setStyleSheet("color: #2c3e50; margin: 5px;")
        
        # Input text area
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Type your ASL phrase here... (e.g., 'I want to learn sign language')")
        self.input_text.setMaximumHeight(100)
        self.input_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                background-color: #ecf0f1;
            }
            QTextEdit:focus {
                border-color: #3498db;
                background-color: #ffffff;
            }
        """)
        
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_text)
        
        return input_frame
        
    def create_translation_section(self):
        """Create the translation section"""
        translation_frame = QFrame()
        translation_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        translation_layout = QVBoxLayout(translation_frame)
        
        # Translation label
        translation_label = QLabel("✨ Polished Translation:")
        translation_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        translation_label.setStyleSheet("color: #2c3e50; margin: 5px;")
        
        # Translation display
        self.translation_display = QTextEdit()
        self.translation_display.setReadOnly(True)
        self.translation_display.setMaximumHeight(120)
        self.translation_display.setStyleSheet("""
            QTextEdit {
                border: 2px solid #27ae60;
                border-radius: 8px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                background-color: #d5f4e6;
                color: #2c3e50;
            }
        """)
        
        # Loading indicator
        self.loading_frame = QFrame()
        loading_layout = QHBoxLayout(self.loading_frame)
        
        self.loading_label = QLabel("🔄 Translating...")
        self.loading_label.setFont(QFont("Arial", 12))
        self.loading_label.setStyleSheet("color: #e67e22;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e67e22;
                border-radius: 10px;
                text-align: center;
                background-color: #fdf2e9;
            }
            QProgressBar::chunk {
                background-color: #e67e22;
                border-radius: 10px;
            }
        """)
        
        loading_layout.addWidget(self.loading_label)
        loading_layout.addWidget(self.progress_bar)
        
        # Hide loading initially
        self.loading_frame.hide()
        
        translation_layout.addWidget(translation_label)
        translation_layout.addWidget(self.translation_display)
        translation_layout.addWidget(self.loading_frame)
        
        return translation_frame
        
    def create_history_section(self):
        """Create the translation history section"""
        history_frame = QFrame()
        history_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        history_layout = QVBoxLayout(history_frame)
        
        # History label
        history_label = QLabel("📚 Translation History:")
        history_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        history_label.setStyleSheet("color: #2c3e50; margin: 5px;")
        
        # History display
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        self.history_display.setMaximumHeight(200)
        self.history_display.setStyleSheet("""
            QTextEdit {
                border: 2px solid #9b59b6;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                background-color: #f8f4fd;
                color: #2c3e50;
            }
        """)
        
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.history_display)
        
        return history_frame
        
    def create_control_section(self):
        """Create the control buttons section"""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QHBoxLayout(control_frame)
        
        # Translate button
        self.translate_button = QPushButton("🚀 Translate")
        self.translate_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.translate_button.setMinimumHeight(50)
        self.translate_button.clicked.connect(self.translate_phrase)
        self.translate_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 15px 30px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f4e79;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        
        # Clear button
        self.clear_button = QPushButton("🗑️ Clear All")
        self.clear_button.setFont(QFont("Arial", 12))
        self.clear_button.setMinimumHeight(50)
        self.clear_button.clicked.connect(self.clear_all)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 15px 30px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #922b21;
            }
        """)
        
        # Copy button
        self.copy_button = QPushButton("📋 Copy Translation")
        self.copy_button.setFont(QFont("Arial", 12))
        self.copy_button.setMinimumHeight(50)
        self.copy_button.clicked.connect(self.copy_translation)
        self.copy_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 25px;
                padding: 15px 30px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        
        control_layout.addWidget(self.translate_button)
        control_layout.addWidget(self.copy_button)
        control_layout.addWidget(self.clear_button)
        
        return control_frame
        
    def setup_styles(self):
        """Setup application-wide styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QFrame {
                background-color: white;
                border-radius: 10px;
            }
        """)
        
    def translate_phrase(self):
        """Handle translation request"""
        asl_input = self.input_text.toPlainText().strip()
        
        if not asl_input:
            QMessageBox.warning(self, "Input Required", "Please enter an ASL phrase to translate.")
            return
            
        if self.is_translating:
            QMessageBox.information(self, "Translation in Progress", "Please wait for the current translation to complete.")
            return
            
        # Show loading state
        self.show_loading_state()
        
        # Start translation in worker thread
        QTimer.singleShot(100, lambda: self.start_translation(asl_input))
        
    def start_translation(self, asl_input):
        """Start translation process"""
        try:
            # Use QTimer to call the worker method
            QTimer.singleShot(0, lambda: self.translation_worker.translate_phrase(asl_input))
        except Exception as e:
            self.hide_loading_state()
            QMessageBox.critical(self, "Error", f"Failed to start translation: {str(e)}")
            
    def show_loading_state(self):
        """Show loading state"""
        self.is_translating = True
        self.translate_button.setEnabled(False)
        self.loading_frame.show()
        self.status_label.setText("🔄 Translating with Ollama...")
        self.translation_display.setText("Processing your ASL phrase...")
        
    def hide_loading_state(self):
        """Hide loading state"""
        self.is_translating = False
        self.translate_button.setEnabled(True)
        self.loading_frame.hide()
        self.status_label.setText("✅ Translation complete!")
        
    def on_translation_complete(self, translation):
        """Handle successful translation"""
        self.hide_loading_state()
        
        # Display translation
        self.translation_display.setText(translation)
        
        # Add to history
        asl_input = self.input_text.toPlainText().strip()
        history_entry = f"ASL: {asl_input}\nEN: {translation}\n{'='*50}\n"
        self.translation_history.append(history_entry)
        
        # Update history display
        self.history_display.setText("".join(self.translation_history))
        
        # Scroll to bottom of history
        scrollbar = self.history_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        self.status_label.setText("✨ Translation completed successfully!")
        
    def on_translation_error(self, error_msg):
        """Handle translation error"""
        self.hide_loading_state()
        self.translation_display.setText("❌ Translation failed")
        self.status_label.setText("❌ Error occurred during translation")
        QMessageBox.critical(self, "Translation Error", error_msg)
        
    def copy_translation(self):
        """Copy translation to clipboard"""
        translation = self.translation_display.toPlainText()
        if translation and translation != "Processing your ASL phrase..." and translation != "❌ Translation failed":
            clipboard = QApplication.clipboard()
            clipboard.setText(translation)
            self.status_label.setText("📋 Translation copied to clipboard!")
        else:
            QMessageBox.information(self, "No Translation", "No translation available to copy.")
            
    def clear_all(self):
        """Clear all inputs and history"""
        reply = QMessageBox.question(self, "Clear All", 
                                   "Are you sure you want to clear all inputs and history?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.input_text.clear()
            self.translation_display.clear()
            self.history_display.clear()
            self.translation_history.clear()
            self.status_label.setText("🧹 All cleared and ready for new translations")
            
    def closeEvent(self, event):
        """Handle application closure"""
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

def main():
    """Main application entry point"""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Enhanced ASL Translator")
        app.setApplicationVersion("2.0")
        
        # Set application icon (if available)
        try:
            app.setWindowIcon(QIcon("icon.png"))
        except:
            pass
            
        # Create and show main window
        window = ASLTranslatorGUI()
        window.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
