"""
PDFScraper v0.01 - Main Application Window
Enhanced PDF Annotation Tool with ML Pipeline Integration and Single File Processing
Integrates all components and provides the primary user interface with advanced ML capabilities.
"""
import pdf_annotator_main
import sys
import os
import json
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLabel, QPushButton, QFileDialog, QMessageBox,
    QComboBox, QToolBar, QStatusBar, QProgressBar, QSplitter,
    QGroupBox, QSpinBox, QCheckBox, QTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QMutex,
    QWaitCondition, QThreadPool, QRunnable, QObject, QStringListModel
)
from PyQt6.QtGui import (
    QAction, QKeySequence, QPixmap, QIcon, QFont, QPalette,
    QColor, QBrush, QPen, QActionGroup
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local imports (these would be your existing modules)
try:
    from pdf_annotator_main import AnnotationCanvas, Annotation
    HAS_ANNOTATION_CANVAS = True
except ImportError:
    logger.warning("pdf_annotator_main module not found - using mock classes")
    HAS_ANNOTATION_CANVAS = False

try:
    from keyword_manager import KeywordManagerDialog
    HAS_KEYWORD_MANAGER = True
except ImportError:
    logger.warning("keyword_manager module not found - using basic keyword management")
    HAS_KEYWORD_MANAGER = False

# Multi-File Trainer import
try:
    from multi_file_trainer import MultiFileTrainer, MultiFileTrainerDialog

    HAS_MULTI_FILE_TRAINER = True
    logger.info("Multi-file trainer loaded successfully")
except ImportError as e:
    logger.warning(f"Multi-file trainer not found: {e}")
    HAS_MULTI_FILE_TRAINER = False

from ml_batch_processor import SingleFileProcessorDialog
HAS_SINGLE_FILE_PROCESSOR = True

# PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Excel export
try:
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ML/OCR processing
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Theme support
try:
    import qdarktheme
    HAS_QT_DARKTHEME = True
except ImportError:
    HAS_QT_DARKTHEME = False

# ML Pipeline imports
try:
    from ml_batch_processor import ThreadedBatchProcessor, BatchMLProcessor, ExtractionResult

    HAS_ML_PIPELINE = True
    logger.info("ML batch processor loaded successfully")
except ImportError as e:
    logger.warning(f"ML batch processor not found: {e} - using basic OCR only")
    HAS_ML_PIPELINE = False


    # Create mock classes
    class ThreadedBatchProcessor:
        def __init__(self, *args, **kwargs): pass

        def start_processing(self, *args, **kwargs): return False

        def stop_processing(self): pass

        def is_processing(self): return False


    class ExtractionResult:
        def __init__(self, *args, **kwargs): pass



# Mock classes for missing dependencies
if not HAS_ANNOTATION_CANVAS:
    class Annotation:
        def __init__(self, **kwargs):
            self.id = kwargs.get('id', '')
            self.keyword = kwargs.get('keyword', '')
            self.color = kwargs.get('color', '#FF0000')
            self.normalized_coords = kwargs.get('normalized_coords', [0, 0, 0, 0])
            self.page_num = kwargs.get('page_num', 0)
            self.text_content = kwargs.get('text_content', '')
            self.confidence = kwargs.get('confidence', 0.0)
            self.created_date = kwargs.get('created_date', '')
            self.modified_date = kwargs.get('modified_date', '')

        def to_dict(self):
            return asdict(self)

        @classmethod
        def from_dict(cls, data):
            return cls(**data)

    class AnnotationCanvas(QWidget):
        annotation_added = pyqtSignal(object)
        annotation_deleted = pyqtSignal(str)
        annotation_modified = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.annotations = {}
            self.selected_annotations = []
            self.keywords = []
            self.keyword_colors = {}
            self.current_keyword = ""
            self.interaction_mode = "draw"

        def set_pdf_page(self, pixmap, page_size):
            if hasattr(self, 'setPixmap'):
                self.setPixmap(pixmap)

        def delete_selected_annotations(self):
            pass

        def copy_annotations(self, annotation_ids):
            pass

        def paste_annotations(self):
            pass

if not HAS_KEYWORD_MANAGER:
    class KeywordManagerDialog(QDialog):
        keywords_updated = pyqtSignal(list, dict)

        def __init__(self, keywords, colors, parent=None):
            super().__init__(parent)
            self.keywords = keywords
            self.colors = colors

if not HAS_SINGLE_FILE_PROCESSOR:
    class SingleFileProcessorDialog(QDialog):
        def __init__(self, keywords, colors, parent=None):
            super().__init__(parent)
            self.keywords = keywords
            self.colors = colors

if not HAS_MULTI_FILE_TRAINER:
    class MultiFileTrainingDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)


class EnhancedBatchProcessingWorker(QObject):
    """Enhanced worker for batch processing PDFs using ML pipeline."""

    progress_updated = pyqtSignal(int, str)  # progress percentage, current file
    processing_complete = pyqtSignal(str, int, dict)  # output file, total processed, statistics
    error_occurred = pyqtSignal(str)
    extraction_result = pyqtSignal(object)  # Individual extraction result

    def __init__(self, pdf_directory: str, output_file: str,
                 keywords: List[str], keyword_colors: Dict[str, str],
                 model_path: str = None, use_ml_pipeline: bool = True):
        super().__init__()

        self.pdf_directory = pdf_directory
        self.output_file = output_file
        self.keywords = keywords
        self.keyword_colors = keyword_colors
        self.model_path = model_path
        self.use_ml_pipeline = use_ml_pipeline
        self.should_stop = False

        # Initialize ML processor if requested and available
        self.ml_processor = None
        if self.use_ml_pipeline and HAS_ML_PIPELINE:
            try:
                self.ml_processor = ThreadedBatchProcessor(model_path=model_path)
            except Exception as e:
                logger.warning(f"Failed to initialize ML processor: {e}")
                self.use_ml_pipeline = False

    def process(self):
        """Process all PDFs using the appropriate method."""
        try:
            if self.use_ml_pipeline and self.ml_processor:
                self._process_with_ml_pipeline()
            else:
                self._process_with_basic_ocr()
        except Exception as e:
            self.error_occurred.emit(f"Processing failed: {str(e)}")

    def _process_with_ml_pipeline(self):
        """Process using the ML pipeline."""
        try:
            # Set up callbacks for ML processor
            def progress_callback(percentage, message):
                self.progress_updated.emit(percentage, message)

            def error_callback(error):
                self.error_occurred.emit(error)

            def complete_callback(results, stats):
                self.processing_complete.emit(self.output_file, len(results), stats)

                # Emit individual results for real-time updates
                for result in results:
                    self.extraction_result.emit(result)

            # Start ML processing
            success = self.ml_processor.start_processing(
                self.pdf_directory,
                self.output_file,
                self.keywords,
                self.keyword_colors,
                progress_callback,
                error_callback,
                complete_callback
            )

            if not success:
                self.error_occurred.emit("Failed to start ML batch processing")
                return

            # Wait for completion or stop signal
            while self.ml_processor.is_processing():
                if self.should_stop:
                    self.ml_processor.stop_processing()
                    break
                QThread.msleep(100)

        except Exception as e:
            self.error_occurred.emit(f"ML processing error: {str(e)}")

    def _process_with_basic_ocr(self):
        """Fallback processing using basic OCR."""
        try:
            pdf_files = list(Path(self.pdf_directory).rglob("*.pdf"))
            total_files = len(pdf_files)

            if total_files == 0:
                self.error_occurred.emit("No PDF files found in the specified directory.")
                return

            results = []
            processed_count = 0

            for i, pdf_file in enumerate(pdf_files):
                if self.should_stop:
                    break

                self.progress_updated.emit(
                    int((i / total_files) * 100),
                    f"Processing: {pdf_file.name}"
                )

                # Process individual PDF
                pdf_results = self.process_single_pdf(pdf_file)
                if pdf_results:
                    results.extend(pdf_results)
                    processed_count += 1

                    # Emit results for real-time display
                    for result in pdf_results:
                        if HAS_ML_PIPELINE:
                            extraction_result = ExtractionResult(
                                pdf_file=result['pdf_file'],
                                page_number=result['page_number'],
                                keyword=result['keyword'],
                                extracted_text=result['extracted_text'],
                                confidence=result['confidence'],
                                normalized_coords=(
                                    result['normalized_x'],
                                    result['normalized_y'],
                                    result['normalized_width'],
                                    result['normalized_height']
                                ),
                                pixel_coords=(
                                    result['pixel_x'],
                                    result['pixel_y'],
                                    result['pixel_width'],
                                    result['pixel_height']
                                ),
                                annotation_color=result['annotation_color'],
                                extraction_method='basic_ocr',
                                processing_time=0.0,
                                metadata={'method': 'basic_ocr'}
                            )
                            self.extraction_result.emit(extraction_result)
                        else:
                            self.extraction_result.emit(result)

            # Save results to Excel
            if results and HAS_PANDAS:
                df = pd.DataFrame(results)

                # Check if output file exists for appending
                if Path(self.output_file).exists():
                    try:
                        existing_df = pd.read_excel(self.output_file)
                        df = pd.concat([existing_df, df], ignore_index=True)
                    except Exception:
                        pass

                df.to_excel(self.output_file, index=False)

                # Create basic statistics
                stats = {
                    'total_files': total_files,
                    'processed_files': processed_count,
                    'total_annotations': len(results),
                    'successful_extractions': len([r for r in results if r.get('confidence', 0) > 30]),
                    'failed_extractions': len([r for r in results if r.get('confidence', 0) <= 30]),
                    'average_confidence': sum(r.get('confidence', 0) for r in results) / len(results) if results else 0,
                    'processing_time': 0.0
                }

                self.processing_complete.emit(self.output_file, processed_count, stats)
            else:
                self.error_occurred.emit("No results to save or pandas not available.")

        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")

    def process_single_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Process a single PDF file using basic OCR."""
        results = []

        try:
            # Check for existing annotation file
            annotation_file = pdf_path.with_suffix('.json')
            if not annotation_file.exists():
                return results

            # Load annotations
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)

            annotations = annotation_data.get('annotations', {})

            # Open PDF
            if not HAS_PYMUPDF:
                return results

            pdf_doc = fitz.open(str(pdf_path))

            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_annotations = {k: v for k, v in annotations.items()
                                    if v.get('page_num', 0) == page_num}

                if not page_annotations:
                    continue

                # Get page as image for OCR
                if HAS_CV2 and HAS_TESSERACT:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Process each annotation
                    for ann_id, ann_data in page_annotations.items():
                        result = self.extract_annotation_text(
                            img, ann_data, pdf_path, page_num
                        )
                        if result:
                            results.append(result)

            pdf_doc.close()

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")

        return results

    def extract_annotation_text(self, img: np.ndarray, annotation: Dict[str, Any],
                                pdf_path: Path, page_num: int) -> Optional[Dict[str, Any]]:
        """Extract text from annotation region using basic OCR."""
        if not HAS_CV2 or not HAS_TESSERACT:
            return None

        try:
            # Convert normalized coordinates to pixel coordinates
            coords = annotation.get('normalized_coords', [0, 0, 0, 0])
            h, w = img.shape[:2]

            x = int(coords[0] * w)
            y = int(coords[1] * h)
            width = int(coords[2] * w)
            height = int(coords[3] * h)

            # Extract region of interest
            roi = img[y:y + height, x:x + width]

            if roi.size == 0:
                return None

            # Preprocess for better OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR extraction
            text = pytesseract.image_to_string(thresh, config='--psm 8').strip()
            confidence_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

            # Calculate average confidence
            confidences = [int(conf) for conf in confidence_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                'pdf_file': str(pdf_path),
                'page_number': page_num + 1,
                'keyword': annotation.get('keyword', ''),
                'extracted_text': text,
                'confidence': round(avg_confidence, 2),
                'normalized_x': coords[0],
                'normalized_y': coords[1],
                'normalized_width': coords[2],
                'normalized_height': coords[3],
                'pixel_x': x,
                'pixel_y': y,
                'pixel_width': width,
                'pixel_height': height,
                'annotation_color': annotation.get('color', '#FF0000')
            }

        except Exception as e:
            logger.error(f"Error extracting text from annotation: {e}")
            return None

    def stop(self):
        """Stop the processing."""
        self.should_stop = True
        if self.ml_processor:
            self.ml_processor.stop_processing()


class BatchProcessingDialog(QDialog):
    """Dialog for configuring and monitoring batch processing."""

    def __init__(self, keywords: List[str], keyword_colors: Dict[str, str], parent=None):
        super().__init__(parent)
        self.keywords = keywords
        self.keyword_colors = keyword_colors
        self.worker = None
        self.thread = None

        self.setWindowTitle("Batch Processing Configuration")
        self.setMinimumWidth(850)
        self.setMaximumHeight(850)
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)

        # Configuration group
        config_group = QGroupBox("Processing Configuration")
        config_layout = QVBoxLayout(config_group)

        # Input directory selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("PDF Directory:"))
        self.input_path_label = QLabel("No directory selected")
        self.input_path_label.setStyleSheet(
            "QLabel { padding: 5px; border: 1px solid #ccc; }")
        input_layout.addWidget(self.input_path_label, 1)

        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.clicked.connect(self.browse_input_directory)
        input_layout.addWidget(self.browse_input_btn)
        config_layout.addLayout(input_layout)

        # Output file selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output File:"))
        self.output_path_label = QLabel("batch_results.xlsx")
        self.output_path_label.setStyleSheet(
            "QLabel { padding: 5px; border: 1px solid #ccc; }")
        output_layout.addWidget(self.output_path_label, 1)

        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_file)
        output_layout.addWidget(self.browse_output_btn)
        config_layout.addLayout(output_layout)

        # Processing options
        options_layout = QVBoxLayout()
        self.use_ml_checkbox = QCheckBox("Use ML Pipeline (Advanced)")
        self.use_ml_checkbox.setChecked(HAS_ML_PIPELINE)
        self.use_ml_checkbox.setEnabled(HAS_ML_PIPELINE)

        tooltip_text = "Use advanced ML pipeline with multiple OCR engines and text classification"
        if not HAS_ML_PIPELINE:
            tooltip_text += " (Unavailable - ML pipeline not installed)"
        self.use_ml_checkbox.setToolTip(tooltip_text)
        options_layout.addWidget(self.use_ml_checkbox)

        # Model path selection
        self.model_path_layout = QHBoxLayout()
        self.model_path_layout.addWidget(QLabel("Model Path (optional):"))
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setStyleSheet(
            "QLabel { padding: 5px; border: 1px solid #ccc; }")
        self.model_path_layout.addWidget(self.model_path_label, 1)

        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        self.browse_model_btn.setEnabled(HAS_ML_PIPELINE)
        self.model_path_layout.addWidget(self.browse_model_btn)
        options_layout.addLayout(self.model_path_layout)

        config_layout.addLayout(options_layout)
        layout.addWidget(config_group)

        # Progress group
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to start processing")
        progress_layout.addWidget(self.status_label)

        # Results display
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["PDF", "Page", "Keyword", "Extracted Text", "Confidence"])

        # Auto-fit header width by stretching all columns
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)

        progress_layout.addWidget(self.results_table)
        layout.addWidget(progress_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def browse_input_directory(self):
        """Browse for input directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select PDF Directory")
        if directory:
            self.input_path_label.setText(directory)

    def browse_output_file(self):
        """Browse for output file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "batch_results.xlsx",
            "Excel files (*.xlsx);;All files (*)"
        )
        if file_path:
            self.output_path_label.setText(file_path)

    def browse_model_file(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "PyTorch models (*.pth *.pt);;All files (*)"
        )
        if file_path:
            self.model_path_label.setText(file_path)

    def start_processing(self):
        """Start the batch processing."""
        input_dir = self.input_path_label.text()
        output_file = self.output_path_label.text()

        if input_dir == "No directory selected":
            QMessageBox.warning(self, "Warning", "Please select an input directory.")
            return

        if not Path(input_dir).exists():
            QMessageBox.warning(self, "Warning", "Selected input directory does not exist.")
            return

        # Get model path if specified
        model_path = None
        if self.model_path_label.text() != "No model selected":
            model_path = self.model_path_label.text()
            if not Path(model_path).exists():
                QMessageBox.warning(self, "Warning", "Selected model file does not exist.")
                return

        # Create and configure worker
        self.worker = EnhancedBatchProcessingWorker(
            input_dir, output_file, self.keywords, self.keyword_colors,
            model_path, self.use_ml_checkbox.isChecked()
        )

        # Create thread
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.processing_complete.connect(self.on_processing_complete)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.extraction_result.connect(self.add_extraction_result)

        self.thread.started.connect(self.worker.process)
        self.thread.finished.connect(self.thread.deleteLater)

        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting processing...")
        self.results_table.setRowCount(0)

        # Start thread
        self.thread.start()

    def stop_processing(self):
        """Stop the processing."""
        if self.worker:
            self.worker.stop()
        self.status_label.setText("Stopping...")

    def update_progress(self, percentage: int, message: str):
        """Update progress display."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def add_extraction_result(self, result):
        """Add extraction result to the table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        # Get result data based on type
        if hasattr(result, 'pdf_file'):  # ExtractionResult object
            pdf_name = Path(result.pdf_file).name
            page_num = str(result.page_number)
            keyword = result.keyword
            text = result.extracted_text[:50] + "..." if len(result.extracted_text) > 50 else result.extracted_text
            confidence = f"{result.confidence:.1f}%"
        else:  # Dictionary format
            pdf_name = Path(result['pdf_file']).name
            page_num = str(result['page_number'])
            keyword = result['keyword']
            text = result['extracted_text'][:50] + "..." if len(result['extracted_text']) > 50 else result['extracted_text']
            confidence = f"{result.get('confidence', 0):.1f}%"

        # Add items to table
        self.results_table.setItem(row, 0, QTableWidgetItem(pdf_name))
        self.results_table.setItem(row, 1, QTableWidgetItem(page_num))
        self.results_table.item(row, 1).setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.results_table.setItem(row, 2, QTableWidgetItem(keyword))
        self.results_table.setItem(row, 3, QTableWidgetItem(text))
        self.results_table.setItem(row, 4, QTableWidgetItem(confidence))
        self.results_table.item(row, 4).setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)

        # Scroll to bottom
        self.results_table.scrollToBottom()

    def on_processing_complete(self, output_file: str, processed_count: int, stats: dict):
        """Handle processing completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Complete! Processed {processed_count} files.")

        # Show completion message
        msg = f"Processing Complete!\n\n"
        msg += f"Files processed: {processed_count}\n"
        msg += f"Total extractions: {stats.get('total_annotations', 0)}\n"
        msg += f"Successful extractions: {stats.get('successful_extractions', 0)}\n"
        msg += f"Average confidence: {stats.get('average_confidence', 0):.1f}%\n\n"
        msg += f"Results saved to: {output_file}"

        QMessageBox.information(self, "Processing Complete", msg)

    def on_error(self, error_message: str):
        """Handle processing error."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)


class PDFAnnotatorMainWindow(QMainWindow):
    """Main application window for PDF annotation with ML pipeline integration."""
    def __init__(self):
        super().__init__()

        self.ui_default_height = 500
        self.ui_default_width = 1400

        self.setWindowTitle("PDFScraper v0.01 - Enhanced PDF Annotation Tool with ML Pipeline")
        self.setGeometry(100, 100, self.ui_default_width, self.ui_default_height)

        # Apply a default theme on startup
        self.current_theme = "dark_cyan.xml"

        # Application state
        self.current_pdf_path: Optional[Path] = None
        self.current_page = 0
        self.total_pages = 0
        self.pdf_document = None
        self.annotations: Dict[str, Annotation] = {}
        self.auto_save_timer = QTimer()

        # Keywords management
        self.keywords = []
        self.keyword_colors = {}

        # UI components
        self.canvas = None
        self.keyword_combo = None
        self.page_label = None
        self.progress_bar = None
        self.status_bar = None

        # Threading
        self.thread_pool = QThreadPool()
        self.processing_thread = None
        self.worker = None

        # Initialize application
        self.load_application_settings()
        self.init_ui()
        self.init_keywords()
        self.setup_auto_save()
        self.setup_shortcuts()
        self.add_ml_menu_items()

        # Apply theme after UI is initialized
        if HAS_QT_DARKTHEME:
            try:
                qdarktheme.setup_theme("auto")
            except Exception as e:
                logger.warning(f"Failed to apply theme: {e}")

    def init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Canvas
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([300, 1100])

        # Create menu bar
        self.create_menu_bar()

        # Create tool bar
        # Create tool bar
        self.create_tool_bar()

        # Create status bar
        self.create_status_bar()

    def create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(250)

        layout = QVBoxLayout(panel)

        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)

        self.open_btn = QPushButton("Open PDF")
        self.open_btn.clicked.connect(self.open_pdf)
        file_layout.addWidget(self.open_btn)

        self.save_btn = QPushButton("Save Annotations")
        self.save_btn.clicked.connect(self.save_annotations)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)

        self.load_annotations_btn = QPushButton("Load Annotations")
        self.load_annotations_btn.clicked.connect(self.load_annotations_dialog)
        self.load_annotations_btn.setEnabled(False)
        file_layout.addWidget(self.load_annotations_btn)

        layout.addWidget(file_group)

        # Navigation group
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)

        nav_buttons_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_page)
        self.prev_btn.setEnabled(False)
        nav_buttons_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        nav_buttons_layout.addWidget(self.next_btn)

        nav_layout.addLayout(nav_buttons_layout)

        self.page_label = QLabel("Page: 0 / 0")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.page_label)

        nav_layout.addStretch()
        layout.addWidget(nav_group)

        # Annotation tools group
        tools_group = QGroupBox("Annotation Tools")
        tools_layout = QVBoxLayout(tools_group)

        # Keyword selection
        tools_layout.addWidget(QLabel("Select Keyword:"))
        self.keyword_combo = QComboBox()
        self.keyword_combo.currentTextChanged.connect(self.update_current_keyword)
        tools_layout.addWidget(self.keyword_combo)

        # Manage keywords button
        self.manage_keywords_btn = QPushButton("Manage Keywords")
        self.manage_keywords_btn.clicked.connect(self.open_keyword_manager)
        tools_layout.addWidget(self.manage_keywords_btn)

        # Interaction mode
        tools_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Draw", "Select", "Resize"])
        self.mode_combo.currentTextChanged.connect(self.update_interaction_mode)
        tools_layout.addWidget(self.mode_combo)

        # Delete selected button
        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_annotations)
        self.delete_selected_btn.setEnabled(False)
        tools_layout.addWidget(self.delete_selected_btn)

        layout.addWidget(tools_group)

        # ML Processing group
        ml_group = QGroupBox("ML Processing")
        ml_layout = QVBoxLayout(ml_group)

        # Enhanced batch processing button
        self.batch_process_btn = QPushButton("Enhanced Batch Process")
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        self.batch_process_btn.setToolTip("Process PDFs with ML pipeline for advanced text extraction")
        ml_layout.addWidget(self.batch_process_btn)

        # Single file processing button - PDFScraper v0.01 enhancement
        self.single_file_btn = QPushButton("Single File Processor")
        self.single_file_btn.clicked.connect(self.open_single_file_processor)
        self.single_file_btn.setToolTip("Process single PDF with real-time feedback and verification")
        self.single_file_btn.setEnabled(HAS_SINGLE_FILE_PROCESSOR)
        ml_layout.addWidget(self.single_file_btn)

        # Multi-file trainer button - PDFScraper v0.01 enhancement
        self.multi_trainer_btn = QPushButton("Multi-File ML Trainer")
        self.multi_trainer_btn.clicked.connect(self.open_multi_file_trainer)
        self.multi_trainer_btn.setToolTip("Train ML models using multiple annotation files")
        self.multi_trainer_btn.setEnabled(HAS_MULTI_FILE_TRAINER)
        ml_layout.addWidget(self.multi_trainer_btn)

        # Training data generation
        self.training_data_btn = QPushButton("Generate Training Data")
        self.training_data_btn.clicked.connect(self.create_training_data)
        self.training_data_btn.setToolTip("Create training dataset from current annotations")
        self.training_data_btn.setEnabled(HAS_ML_PIPELINE)
        ml_layout.addWidget(self.training_data_btn)

        # Export to Excel
        self.export_excel_btn = QPushButton("Export to Excel")
        self.export_excel_btn.clicked.connect(self.export_to_excel)
        ml_layout.addWidget(self.export_excel_btn)

        layout.addWidget(ml_group)

        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        layout.addWidget(stats_group)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right panel with canvas."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create scroll area for canvas
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create canvas
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_added.connect(self.on_annotation_added)
        self.canvas.annotation_deleted.connect(self.on_annotation_deleted)
        self.canvas.annotation_modified.connect(self.on_annotation_modified)

        scroll_area.setWidget(self.canvas)
        layout.addWidget(scroll_area)

        return panel

    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        open_action = QAction('Open PDF', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_pdf)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction('Save Annotations', self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)

        load_action = QAction('Load Annotations', self)
        load_action.triggered.connect(self.load_annotations_dialog)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        export_action = QAction('Export to Excel', self)
        export_action.triggered.connect(self.export_to_excel)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction('Quit', self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu('Edit')

        copy_action = QAction('Copy', self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_selected)
        edit_menu.addAction(copy_action)

        paste_action = QAction('Paste', self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(self.paste_annotations)
        edit_menu.addAction(paste_action)

        delete_action = QAction('Delete', self)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self.delete_selected_annotations)
        edit_menu.addAction(delete_action)

        edit_menu.addSeparator()

        select_all_action = QAction('Select All', self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.select_all_annotations)
        edit_menu.addAction(select_all_action)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        keywords_action = QAction('Manage Keywords', self)
        keywords_action.triggered.connect(self.open_keyword_manager)
        tools_menu.addAction(keywords_action)

        batch_action = QAction('Enhanced Batch Process', self)
        batch_action.triggered.connect(self.start_batch_processing)
        batch_action.setToolTip("Advanced batch processing with ML pipeline")
        tools_menu.addAction(batch_action)

        # PDFScraper v0.01 - New menu items
        tools_menu.addSeparator()

        single_action = QAction('Single File Processor', self)
        single_action.triggered.connect(self.open_single_file_processor)
        single_action.setEnabled(HAS_SINGLE_FILE_PROCESSOR)
        tools_menu.addAction(single_action)

        trainer_action = QAction('Multi-File ML Trainer', self)
        trainer_action.triggered.connect(self.open_multi_file_trainer)
        trainer_action.setEnabled(HAS_MULTI_FILE_TRAINER)
        tools_menu.addAction(trainer_action)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Theme menu (if qdarktheme is available)
        if HAS_QT_DARKTHEME:
            theme_menu = menubar.addMenu("Theme")
            theme_action_group = QActionGroup(self)
            theme_action_group.setExclusive(True)

            # Get available themes
            try:
                themes = ["auto", "dark", "light"]
                for theme_name in themes:
                    # Create the action without the 'checkable' argument
                    action = QAction(theme_name.title(), self, checkable=True)
                    # Set the action to be checkable after creation
#                    action.setCheckable(True)
                    # Add the action to the menu
                    theme_action_group.addAction(action)
                    # Add the action to the menu
                    theme_menu.addAction(action)

                    # Connect the signal
                    action.triggered.connect(lambda checked, t=theme_name: self.apply_theme(t))

                    # Check the default theme
                    if theme_name == "auto":
                        action.setChecked(True)
            except Exception as e:
                logger.warning(f"Failed to load themes: {e}")

    def apply_theme(self, theme_name):
        """Apply a theme to the application."""
        if HAS_QT_DARKTHEME:
            try:
                self.current_theme = theme_name
                qdarktheme.setup_theme(theme_name)
            except Exception as e:
                logger.warning(f"Failed to apply theme {theme_name}: {e}")

    def add_ml_menu_items(self):
        """Add ML-related menu items to existing menus."""
        # Get tools menu
        tools_menu = None
        for action in self.menuBar().actions():
            if action.text() == 'Tools':
                tools_menu = action.menu()
                break

        if tools_menu and HAS_ML_PIPELINE:
            tools_menu.addSeparator()

            # Training data generation
            training_action = QAction('Generate Training Data', self)
            training_action.triggered.connect(self.create_training_data)
            training_action.setToolTip("Generate training data from annotations for ML model training")
            tools_menu.addAction(training_action)

    def create_tool_bar(self):
        """Create the application toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # File operations
        toolbar.addAction('Open', self.open_pdf)
        toolbar.addAction('Save', self.save_annotations)
        toolbar.addSeparator()

        # Navigation
        toolbar.addAction('Previous', self.previous_page)
        toolbar.addAction('Next', self.next_page)
        toolbar.addSeparator()

        # Tools
        toolbar.addAction('Keywords', self.open_keyword_manager)
        toolbar.addAction('Batch Process', self.start_batch_processing)

        # PDFScraper v0.01 - New toolbar items
        toolbar.addSeparator()
        toolbar.addAction('Single File', self.open_single_file_processor)
        toolbar.addAction('ML Trainer', self.open_multi_file_trainer)

    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Progress bar for batch processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Feature status indicators
        feature_status = []
        if HAS_PYMUPDF:
            feature_status.append("PDF✓")
        if HAS_CV2 and HAS_TESSERACT:
            feature_status.append("OCR✓")
        if HAS_ML_PIPELINE:
            feature_status.append("ML✓")
        if HAS_PANDAS:
            feature_status.append("Excel✓")
        if HAS_QT_DARKTHEME:
            feature_status.append("Theme✓")
        if HAS_SINGLE_FILE_PROCESSOR:
            feature_status.append("SFP✓")
        if HAS_MULTI_FILE_TRAINER:
            feature_status.append("MLT✓")

        status_text = "Features: " + " | ".join(feature_status)
        self.status_bar.showMessage(f"PDFScraper v0.01 Ready - {status_text}")

    def init_keywords(self):
        """Initialize keywords from saved settings or defaults."""
        # Keywords already loaded in load_application_settings
        if not self.keywords:
            self.keywords = [
                "Drawing Title", "Drawing Number", "Drawing Page", "Program",
                "Current Revision", "Drawing Date", "Contract Number", "PCA Date",
                "Drawing Type", "Drawing Status", "Vendor Name", "Vendor Part Number",
                "Vendor Cage Code", "Vendor Address", "Notes"
            ]

            # Assign default colors
            default_colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
                "#00FFFF", "#FFA500", "#800080", "#008000", "#FFC0CB",
                "#A52A2A", "#808080", "#000080", "#800000", "#008080"
            ]

            for i, keyword in enumerate(self.keywords):
                if keyword not in self.keyword_colors:
                    color_index = i % len(default_colors)
                    self.keyword_colors[keyword] = default_colors[color_index]

        self.update_keyword_combo()
        self.update_canvas_keywords()

    def setup_auto_save(self):
        """Setup auto-save functionality."""
        self.auto_save_timer.setSingleShot(True)
        self.auto_save_timer.timeout.connect(self.auto_save_annotations)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Navigation shortcuts
        next_shortcut = QAction(self)
        next_shortcut.setShortcut(Qt.Key.Key_Right)
        next_shortcut.triggered.connect(self.next_page)
        self.addAction(next_shortcut)

        prev_shortcut = QAction(self)
        prev_shortcut.setShortcut(Qt.Key.Key_Left)
        prev_shortcut.triggered.connect(self.previous_page)
        self.addAction(prev_shortcut)

    def open_pdf(self):
        """Open a PDF file."""
        if not HAS_PYMUPDF:
            QMessageBox.critical(self, "Error",
                                 "PyMuPDF is required to open PDF files.\n"
                                 "Install with: pip install PyMuPDF")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", "", "PDF files (*.pdf);;All files (*)"
        )

        if not file_path:
            return

        try:
            # Close current document if open
            if self.pdf_document:
                self.pdf_document.close()

            # Open new document
            self.pdf_document = fitz.open(file_path)
            self.current_pdf_path = Path(file_path)
            self.total_pages = len(self.pdf_document)
            self.current_page = 0

            # Load existing annotations if available
            self.load_existing_annotations()

            # Display first page
            self.display_current_page()

            # Update UI
            self.update_navigation_ui()
            self.save_btn.setEnabled(True)
            self.load_annotations_btn.setEnabled(True)

            self.status_bar.showMessage(f"Opened: {self.current_pdf_path.name}")
            self.update_statistics()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open PDF: {str(e)}")

    def display_current_page(self):
        """Display the current PDF page."""
        if not self.pdf_document or self.current_page >= self.total_pages:
            return

        try:
            page = self.pdf_document[self.current_page]

            # Render page as pixmap
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)

            # Convert to QPixmap
            img_data = pix.tobytes("png")
            pixmap = QPixmap()
            pixmap.loadFromData(img_data)

            # Update canvas
            page_size = QSize(int(page.rect.width), int(page.rect.height))
            self.canvas.set_pdf_page(pixmap, page_size)

            # Load annotations for this page
            self.load_page_annotations()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display page: {str(e)}")

    def load_page_annotations(self):
        """Load annotations for the current page."""
        if not self.canvas:
            return

        # Clear current annotations from canvas
        self.canvas.annotations.clear()

        # Load annotations for current page
        page_annotations = {
            ann_id: ann for ann_id, ann in self.annotations.items()
            if ann.page_num == self.current_page
        }

        self.canvas.annotations = page_annotations
        self.canvas.update()

    def previous_page(self):
        """Navigate to previous page."""
        if self.current_page > 0:
            self.save_current_page_annotations()
            self.current_page -= 1
            self.display_current_page()
            self.update_navigation_ui()

    def next_page(self):
        """Navigate to next page."""
        if self.current_page < self.total_pages - 1:
            self.save_current_page_annotations()
            self.current_page += 1
            self.display_current_page()
            self.update_navigation_ui()

    def update_navigation_ui(self):
        """Update navigation UI elements."""
        if self.page_label:
            self.page_label.setText(f"Page: {self.current_page + 1} / {self.total_pages}")

        if self.prev_btn:
            self.prev_btn.setEnabled(self.current_page > 0)

        if self.next_btn:
            self.next_btn.setEnabled(self.current_page < self.total_pages - 1)

    def save_current_page_annotations(self):
        """Save annotations from current page to main storage."""
        if not self.canvas:
            return

        # Remove existing annotations for this page
        page_annotations_to_remove = [
            ann_id for ann_id, ann in self.annotations.items()
            if ann.page_num == self.current_page
        ]

        for ann_id in page_annotations_to_remove:
            del self.annotations[ann_id]

        # Add current canvas annotations
        for ann_id, annotation in self.canvas.annotations.items():
            annotation.page_num = self.current_page
            self.annotations[ann_id] = annotation

    def on_annotation_added(self, annotation: Annotation):
        """Handle annotation added signal."""
        annotation.page_num = self.current_page
        self.annotations[annotation.id] = annotation
        self.trigger_auto_save()
        self.update_statistics()
        self.status_bar.showMessage("Annotation added", 2000)

    def on_annotation_deleted(self, annotation_id: str):
        """Handle annotation deleted signal."""
        if annotation_id in self.annotations:
            del self.annotations[annotation_id]
        self.trigger_auto_save()
        self.update_statistics()
        self.status_bar.showMessage("Annotation deleted", 2000)

    def on_annotation_modified(self, annotation: Annotation):
        """Handle annotation modified signal."""
        annotation.page_num = self.current_page
        self.annotations[annotation.id] = annotation
        self.trigger_auto_save()
        self.status_bar.showMessage("Annotation modified", 2000)

    def trigger_auto_save(self):
        """Trigger auto-save after a short delay."""
        self.auto_save_timer.start(2000)  # 2 second delay

    def auto_save_annotations(self):
        """Auto-save annotations."""
        if self.current_pdf_path and self.annotations:
            self.save_annotations_to_file(self.current_pdf_path.with_suffix('.json'))
            self.status_bar.showMessage("Annotations auto-saved", 2000)

    def save_annotations(self):
        """Save annotations to file."""
        if not self.current_pdf_path:
            QMessageBox.warning(self, "Warning", "No PDF file is currently open.")
            return

        # Save current page annotations first
        self.save_current_page_annotations()

        # Choose save location
        default_path = self.current_pdf_path.with_suffix('.json')
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotations", str(default_path),
            "JSON files (*.json);;All files (*)"
        )

        if file_path:
            self.save_annotations_to_file(Path(file_path))
            self.status_bar.showMessage("Annotations saved", 2000)

    def save_annotations_to_file(self, file_path: Path):
        """Save annotations to specified file."""
        try:
            # Prepare data for saving
            annotations_data = {}
            for ann_id, annotation in self.annotations.items():
                annotations_data[ann_id] = annotation.to_dict()

            data = {
                'pdf_file': str(self.current_pdf_path),
                'total_pages': self.total_pages,
                'annotations': annotations_data,
                'keywords': self.keywords,
                'keyword_colors': self.keyword_colors,
                'created_date': datetime.now().isoformat(),
                'modified_date': datetime.now().isoformat(),
                'application_version': '0.01'
            }

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotations: {str(e)}")

    def load_existing_annotations(self):
        """Load existing annotations for the current PDF."""
        if not self.current_pdf_path:
            return

        annotation_file = self.current_pdf_path.with_suffix('.json')
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)

                # Load annotations
                annotations_data = data.get('annotations', {})
                self.annotations.clear()

                for ann_id, ann_data in annotations_data.items():
                    annotation = Annotation.from_dict(ann_data)
                    self.annotations[ann_id] = annotation

                self.status_bar.showMessage("Existing annotations loaded", 2000)

            except Exception as e:
                QMessageBox.warning(self, "Warning",
                                    f"Failed to load existing annotations: {str(e)}")

    def load_annotations_dialog(self):
        """Show dialog to load annotations from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", "",
            "JSON files (*.json);;All files (*)"
        )

        if file_path:
            self.load_annotations_from_file(Path(file_path))

    def load_annotations_from_file(self, file_path: Path):
        """Load annotations from specified file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Load annotations
            annotations_data = data.get('annotations', {})
            self.annotations.clear()

            for ann_id, ann_data in annotations_data.items():
                annotation = Annotation.from_dict(ann_data)
                self.annotations[ann_id] = annotation

            # Reload current page
            self.load_page_annotations()
            self.update_statistics()

            self.status_bar.showMessage("Annotations loaded", 2000)
            QMessageBox.information(self, "Success",
                                    f"Loaded {len(self.annotations)} annotations.")

        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"Failed to load annotations: {str(e)}")

    def update_keyword_combo(self):
        """Update the keyword combo box."""
        if True:
            current_text = self.keyword_combo.currentText()
            self.keyword_combo.clear()
            self.keyword_combo.addItems(self.keywords)

            # Restore selection if possible
            if current_text in self.keywords:
                self.keyword_combo.setCurrentText(current_text)
            elif self.keywords:
                self.keyword_combo.setCurrentIndex(0)

    def update_canvas_keywords(self):
        """Update canvas with current keywords and colors."""
        if self.canvas:
            self.canvas.keywords = self.keywords
            self.canvas.keyword_colors = self.keyword_colors
            self.canvas.current_keyword = self.keyword_combo.currentText() if self.keyword_combo else ""

    def update_current_keyword(self, keyword: str):
        """Update the current keyword for drawing."""
        if self.canvas:
            self.canvas.current_keyword = keyword

    def update_interaction_mode(self, mode: str):
        """Update the interaction mode."""
        if self.canvas:
            self.canvas.interaction_mode = mode.lower()

    def open_keyword_manager(self):
        """Open the keyword manager dialog."""
        if HAS_KEYWORD_MANAGER:
            dialog = KeywordManagerDialog(self.keywords, self.keyword_colors, self)
            dialog.keywords_updated.connect(self.on_keywords_updated)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                pass  # Changes already applied via signal
        else:
            QMessageBox.information(self, "Keyword Manager",
                                    "Keyword manager module not available.\n"
                                    "Using basic keyword management.")

    def on_keywords_updated(self, keywords: List[str], keyword_colors: Dict[str, str]):
        """Handle updated keywords from manager."""
        self.keywords = keywords
        self.keyword_colors = keyword_colors
        self.update_keyword_combo()
        self.update_canvas_keywords()
        self.status_bar.showMessage("Keywords updated", 2000)

    def delete_selected_annotations(self):
        """Delete selected annotations."""
        if self.canvas:
            self.canvas.delete_selected_annotations()

    def copy_selected(self):
        """Copy selected annotations."""
        if self.canvas and self.canvas.selected_annotations:
            self.canvas.copy_annotations(self.canvas.selected_annotations)
            self.status_bar.showMessage(f"Copied {len(self.canvas.selected_annotations)} annotation(s)", 2000)

    def paste_annotations(self):
        """Paste copied annotations."""
        if self.canvas:
            self.canvas.paste_annotations()

    def select_all_annotations(self):
        """Select all annotations on current page."""
        if self.canvas:
            self.canvas.selected_annotations = list(self.canvas.annotations.keys())
            self.canvas.update()

    def start_batch_processing(self):
        """Start enhanced batch processing of PDFs."""
        # Create and show the batch processing dialog
        dialog = BatchProcessingDialog(self.keywords, self.keyword_colors, self)
        dialog.exec()

        # PDFScraper v0.01 - New methods for enhanced functionality

    def open_single_file_processor(self):
        """Open single file processor dialog."""
        # REPLACE the existing method with:
        dialog = SingleFileProcessorDialog(self.keywords, self.keyword_colors, self)
        dialog.exec()

    def open_multi_file_trainer(self):
        """Open multi-file ML trainer dialog."""
        if not HAS_MULTI_FILE_TRAINER:
            QMessageBox.warning(self, "Feature Unavailable",
                                "Multi-File ML Trainer not available.\n"
                                "Module not found or dependencies missing.")
            return

        dialog = MultiFileTrainerDialog(self)
        dialog.exec()

    def create_training_data(self):
        """Create training data from annotations."""
        if not HAS_ML_PIPELINE:
            QMessageBox.warning(self, "Feature Unavailable",
                                "ML pipeline not available. Install required dependencies:\n"
                                "pip install opencv-python torch torchvision")
            return

        if not self.annotations:
            QMessageBox.information(self, "No Data",
                                    "No annotations available to create training data.")
            return

        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Training Data Output Directory")
        if not output_dir:
            return

        try:
            # Create a temporary directory with annotation files
            temp_dir = Path(output_dir) / "temp_annotations"
            temp_dir.mkdir(exist_ok=True)

            # Save current annotations to temporary location
            if self.current_pdf_path:
                # Copy PDF to temp directory
                import shutil
                temp_pdf = temp_dir / self.current_pdf_path.name
                shutil.copy2(self.current_pdf_path, temp_pdf)

                # Save annotations
                temp_annotation_file = temp_dir / f"{self.current_pdf_path.stem}.json"
                self.save_annotations_to_file(temp_annotation_file)

                # Generate training data
                from ml_batch_processor import TrainingDataGenerator
                generator = TrainingDataGenerator(str(temp_dir))
                success = generator.generate_training_dataset(output_dir)

                # Clean up temporary files
                shutil.rmtree(temp_dir)

                if success:
                    QMessageBox.information(self, "Training Data Created",
                                            f"Training data successfully created in:\n{output_dir}\n\n"
                                            f"Generated files:\n"
                                            f"• training_data.json (complete dataset)\n"
                                            f"• training_data.csv (metadata)\n"
                                            f"• training_stats.json (statistics)\n"
                                            f"• images/ (organized training images)")
                else:
                    QMessageBox.warning(self, "Training Data Creation Failed",
                                        "Failed to create training data. Check the logs for details.")
            else:
                QMessageBox.warning(self, "No PDF Open", "Please open a PDF file first.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create training data: {str(e)}")

    def export_to_excel(self):
        """Export current annotations to Excel."""
        if not HAS_PANDAS:
            QMessageBox.critical(self, "Error",
                                 "pandas is required for Excel export.\n"
                                 "Install with: pip install pandas openpyxl")
            return

        if not self.annotations:
            QMessageBox.information(self, "No Data", "No annotations to export.")
            return

        # Save current page annotations first
        self.save_current_page_annotations()

        # Choose export location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "annotations.xlsx",
            "Excel files (*.xlsx);;All files (*)"
        )

        if file_path:
            self.export_annotations_to_excel(Path(file_path))

    def export_annotations_to_excel(self, file_path: Path):
        """Enhanced Excel export with additional ML-related columns."""
        try:
            # Prepare data with ML-related fields
            export_data = []

            for annotation in self.annotations.values():
                # Calculate pixel coordinates if PDF is loaded
                pixel_coords = [0, 0, 0, 0]
                if self.pdf_document and annotation.page_num < len(self.pdf_document):
                    page = self.pdf_document[annotation.page_num]
                    page_width = page.rect.width
                    page_height = page.rect.height

                    pixel_coords = [
                        annotation.normalized_coords[0] * page_width,
                        annotation.normalized_coords[1] * page_height,
                        annotation.normalized_coords[2] * page_width,
                        annotation.normalized_coords[3] * page_height
                    ]

                export_data.append({
                    'PDF File': str(self.current_pdf_path) if self.current_pdf_path else '',
                    'Page Number': annotation.page_num + 1,
                    'Keyword': annotation.keyword,
                    'Color': annotation.color,
                    'Normalized X': annotation.normalized_coords[0],
                    'Normalized Y': annotation.normalized_coords[1],
                    'Normalized Width': annotation.normalized_coords[2],
                    'Normalized Height': annotation.normalized_coords[3],
                    'Pixel X': pixel_coords[0],
                    'Pixel Y': pixel_coords[1],
                    'Pixel Width': pixel_coords[2],
                    'Pixel Height': pixel_coords[3],
                    'Text Content': annotation.text_content,
                    'Confidence': annotation.confidence,
                    'Extraction Method': getattr(annotation, 'extraction_method', 'manual'),
                    'Processing Time': getattr(annotation, 'processing_time', 0.0),
                    'Created Date': getattr(annotation, 'created_date', ''),
                    'Modified Date': getattr(annotation, 'modified_date', '')
                })

            # Create DataFrame and save with enhanced formatting
            df = pd.DataFrame(export_data)

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Annotations', index=False)

                # Summary sheet
                summary_data = []
                keyword_counts = df['Keyword'].value_counts()
                for keyword, count in keyword_counts.items():
                    avg_confidence = df[df['Keyword'] == keyword]['Confidence'].mean()
                    summary_data.append({
                        'Keyword': keyword,
                        'Count': count,
                        'Average Confidence': round(avg_confidence, 2) if not pd.isna(avg_confidence) else 0
                    })

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Apply formatting
                header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                header_font = Font(color="FFFFFF", bold=True)

                for sheet_name in ['Annotations', 'Summary']:
                    worksheet = writer.sheets[sheet_name]

                    # Header formatting
                    for cell in worksheet[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal="center")

                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter

                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass

                        adjusted_width = min(max(max_length + 2, 10), 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width

            self.status_bar.showMessage("Enhanced Excel export complete", 2000)
            QMessageBox.information(self, "Export Complete",
                                    f"Exported {len(export_data)} annotations to Excel with summary statistics.")

        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                 f"Failed to export to Excel: {str(e)}")

    def update_statistics(self):
        """Update the statistics display."""
        if not self.stats_text:
            return

        total_annotations = len(self.annotations)
        annotations_by_keyword = {}

        for annotation in self.annotations.values():
            keyword = annotation.keyword
            annotations_by_keyword[keyword] = annotations_by_keyword.get(keyword, 0) + 1

        stats_text = f"Total Annotations: {total_annotations}\n"

        if annotations_by_keyword:
            stats_text += "By Keyword:\n"
            for keyword, count in sorted(annotations_by_keyword.items()):
                stats_text += f"  {keyword}: {count}\n"

        # Add feature availability status
        stats_text += "\nFeatures Available:\n"
        stats_text += f"  PDF Processing: {'✓' if HAS_PYMUPDF else '✗'}\n"
        stats_text += f"  ML Pipeline: {'✓' if HAS_ML_PIPELINE else '✗'}\n"
        stats_text += f"  OCR: {'✓' if (HAS_CV2 and HAS_TESSERACT) else '✗'}\n"
        stats_text += f"  Excel Export: {'✓' if HAS_PANDAS else '✗'}\n"
        stats_text += f"  Single File Processor: {'✓' if HAS_SINGLE_FILE_PROCESSOR else '✗'}\n"
        stats_text += f"  Multi-File Trainer: {'✓' if HAS_MULTI_FILE_TRAINER else '✗'}\n"

        self.stats_text.setText(stats_text)

    def show_about(self):
        """Show about dialog."""
        about_text = (
            "PDFScraper v0.01 - Enhanced PDF Annotation Tool\n"
            "Version 0.01\n\n"
            "A comprehensive tool for annotating engineering drawings "
            "with Machine Learning pipeline integration for advanced text extraction.\n\n"
            "New Features in v0.01:\n"
            "• Single File Processor - Process individual PDFs with real-time feedback\n"
            "• Multi-File ML Trainer - Train custom ML models from multiple annotation files\n"
            "• Enhanced batch processing with ML pipeline\n"
            "• Real-time preview and verification\n"
            "• Advanced training data generation\n"
            "• Cross-validation and performance metrics\n\n"
            "Core Features:\n"
            "• Multi-page PDF annotation\n"
            "• Customizable keywords and colors\n"
            "• Auto-save functionality\n"
            "• Multiple OCR engines (Tesseract, EasyOCR)\n"
            "• Smart text classification and post-processing\n"
            "• Excel export with detailed statistics\n"
            "• Copy/Paste/Resize/Delete annotations\n"
            "• Modern theming support\n\n"
            "Requirements:\n"
            "• PyMuPDF (pip install PyMuPDF)\n"
            "• For ML features: opencv-python, torch, easyocr\n"
            "• For Excel: pandas, openpyxl\n"
            "• For themes: qdarktheme"
        )

        QMessageBox.about(self, "About PDFScraper v0.01", about_text)

    def load_application_settings(self):
        """Load application settings."""
        try:
            settings_file = Path.home() / ".pdf_annotator_settings.json"
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = json.load(f)

                # Load keywords and colors
                self.keywords = settings.get('keywords', [])
                self.keyword_colors = settings.get('keyword_colors', {})

                # Restore window geometry
                geometry = settings.get('window_geometry', {})
                if geometry:
                    self.setGeometry(
                        geometry.get('x', 100),
                        geometry.get('y', 100),
                        geometry.get('width', self.ui_default_width),
                        geometry.get('height', self.ui_default_height)
                    )

                # Restore theme
                self.current_theme = settings.get('current_theme', 'auto')

        except Exception as e:
            logger.warning(f"Failed to load application settings: {e}")

    def save_application_settings(self):
        """Save application settings including keywords and theme."""
        try:
            settings = {
                'keywords': self.keywords,
                'keyword_colors': self.keyword_colors,
                'current_theme': getattr(self, 'current_theme', 'auto'),
                'window_geometry': {
                    'x': self.x(),
                    'y': self.y(),
                    'width': self.width(),
                    'height': self.height()
                },
                'application_version': '0.01'
            }

            settings_file = Path.home() / ".pdf_annotator_settings.json"
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save application settings: {e}")

    def closeEvent(self, event):
        """Enhanced close event handler."""
        # Save current work
        if self.current_pdf_path and self.annotations:
            self.save_current_page_annotations()
            self.auto_save_annotations()

        # Stop any running batch processing
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Batch Processing Active',
                'Batch processing is active. Stop and exit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Try to stop gracefully first
                if hasattr(self, 'worker') and self.worker:
                    self.worker.stop()

                # Wait a moment for graceful shutdown
                self.processing_thread.wait(2000)

                # Force terminate if still running
                if self.processing_thread.isRunning():
                    self.processing_thread.terminate()
                    self.processing_thread.wait()
            else:
                event.ignore()
                return

        # Save application settings
        self.save_application_settings()

        # Close PDF document
        if self.pdf_document:
            self.pdf_document.close()

        event.accept()

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("PDFScraper v0.01 - Enhanced PDF Annotation Tool")
    app.setApplicationVersion("0.01")

    # Check dependencies and show detailed information
    missing_critical = []
    missing_optional = []

    if not HAS_PYMUPDF:
        missing_critical.append("PyMuPDF (pip install PyMuPDF)")

    if not HAS_PANDAS:
        missing_optional.append("pandas (pip install pandas openpyxl)")
    if not HAS_CV2:
        missing_optional.append("OpenCV (pip install opencv-python)")
    if not HAS_TESSERACT:
        missing_optional.append("pytesseract (pip install pytesseract)")
    if not HAS_QT_DARKTHEME:
        missing_optional.append("qdarktheme (pip install qdarktheme)")
    if not HAS_SINGLE_FILE_PROCESSOR:
        missing_optional.append("Single File Processor (check single_file_processor.py)")
    if not HAS_MULTI_FILE_TRAINER:
        missing_optional.append("Multi-File Trainer (check multi_file_trainer.py)")

    # Show critical dependency warnings
    if missing_critical:
        msg = "Critical dependencies missing. The application cannot run without:\n\n"
        msg += "\n".join(f"• {dep}" for dep in missing_critical)
        msg += "\n\nPlease install the required dependencies and restart the application."

        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Icon.Critical)
        msgbox.setWindowTitle("Critical Dependencies Missing")
        msgbox.setText(msg)
        msgbox.exec()
        return 1

    # Show optional dependency warnings
    if missing_optional:
        msg = "Optional dependencies missing for enhanced functionality:\n\n"
        msg += "\n".join(f"• {dep}" for dep in missing_optional)
        msg += "\n\nThe application will run with limited features. "
        msg += "Install the missing packages for full ML pipeline support."

        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Icon.Information)
        msgbox.setWindowTitle("Optional Dependencies")
        msgbox.setText(msg)
        msgbox.setDetailedText(
            "Available Features:\n"
            f"• Basic PDF annotation: {'✓' if HAS_PYMUPDF else '✗'}\n"
            f"• Excel export: {'✓' if HAS_PANDAS else '✗'}\n"
            f"• OCR processing: {'✓' if (HAS_CV2 and HAS_TESSERACT) else '✗'}\n"
            f"• ML pipeline: {'✓' if all([HAS_CV2, HAS_TESSERACT, HAS_PANDAS, HAS_ML_PIPELINE]) else '✗'}\n"
            f"• Dark themes: {'✓' if HAS_QT_DARKTHEME else '✗'}\n"
            f"• Single File Processor: {'✓' if HAS_SINGLE_FILE_PROCESSOR else '✗'}\n"
            f"• Multi-File Trainer: {'✓' if HAS_MULTI_FILE_TRAINER else '✗'}\n"
        )
        msgbox.exec()

    # Create and show main window
    try:
        window = PDFAnnotatorMainWindow()
        window.show()

        # Show welcome message on first run
        settings_file = Path.home() / ".pdf_annotator_settings.json"
        if not settings_file.exists():
            welcome_msg = (
                "Welcome to PDFScraper v0.01!\n\n"
                "New Enhanced Features:\n"
                "• Single File Processor - Process individual PDFs with real-time feedback\n"
                "• Multi-File ML Trainer - Train custom ML models from annotation data\n"
                "• Advanced ML-powered text extraction\n"
                "• Multiple OCR engine support\n"
                "• Smart text classification and post-processing\n"
                "• Enhanced batch processing with cross-validation\n"
                "• Real-time preview and verification capabilities\n\n"
                "Get started by opening a PDF file and creating annotations, or use the new\n"
                "Single File Processor for immediate text extraction and verification."
            )

            QMessageBox.information(window, "Welcome to PDFScraper v0.01", welcome_msg)

        return app.exec()

    except Exception as e:
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Icon.Critical)
        msgbox.setWindowTitle("Application Error")
        msgbox.setText(f"Failed to start application: {str(e)}")
        msgbox.exec()
        return 1

if __name__ == "__main__":
    sys.exit(main())