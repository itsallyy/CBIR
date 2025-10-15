import sys
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QScrollArea, QGroupBox, QSpinBox, QComboBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from features import FeatureExtractor, build_database
from retrieval import train_pca, search_similar
from gui_utils import qimage_to_bgr
from dialogs import EnhancedCanvasQueryDialog, EnhancedContainmentQueryDialog, FeatureSelectionDialog, FeatureImportanceDialog, DropLabel, ClickableImage
from canvas_widgets import EnhancedCanvasWidget
from feature_combination import combine_query_feedback


class CBIRApp(QWidget):
    
    
    def __init__(self, feature_extractor, database):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.database = database
        self.selected_features = ["shape", "color", "color_hist", "sift", "harris", "hog", "glcm", "lbp", "orb", "cnn"]
        self.query_features = None
        self.query_image_path = None
        self.result_widgets = []
        self.max_results = 10
        # Train PCA on the database with the default selected features
        self.pca = train_pca(self.database, self.selected_features, n_components=256)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Content-Based Image Retrieval (CBIR) App")
        self.setGeometry(100, 100, 1200, 850)
        self.setStyleSheet("""
            QWidget { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F8E6CB, stop:0.5 #F8D49B, stop:1 #62C4C3);
                font-family: Verdana; font-size: 14px;
            }
            QPushButton { background-color: #F8D49B; color: #333; border: none; padding: 8px; border-radius: 8px; }
            QPushButton:hover { background-color: #62C4C3; }
            QLabel { color: #333; }
            QGroupBox { background: transparent; font-weight: bold; }
            QScrollArea { background: transparent; }
        """)
        main_layout = QVBoxLayout()
        query_mode_box = QGroupBox("Query Mode")
        qm_layout = QHBoxLayout()
        self.load_query_btn = QPushButton("Load Image")
        self.load_query_btn.clicked.connect(self.load_query_image)
        qm_layout.addWidget(self.load_query_btn)
        self.canvas_query_btn = QPushButton("Query Canvas")
        self.canvas_query_btn.clicked.connect(self.open_canvas_dialog)
        qm_layout.addWidget(self.canvas_query_btn)
        self.contain_query_btn = QPushButton("Containment Query")
        self.contain_query_btn.clicked.connect(self.open_containment_dialog)
        qm_layout.addWidget(self.contain_query_btn)
        query_mode_box.setLayout(qm_layout)
        main_layout.addWidget(query_mode_box)
        top_layout = QHBoxLayout()
        self.query_label = DropLabel(self)
        self.query_label.setFixedSize(300,300)
        top_layout.addWidget(self.query_label)
        control_layout = QVBoxLayout()
        self.delete_query_btn = QPushButton("Delete Query")
        self.delete_query_btn.clicked.connect(self.delete_query)
        control_layout.addWidget(self.delete_query_btn)
        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.clicked.connect(self.clear_results)
        control_layout.addWidget(self.clear_results_btn)
        max_layout = QHBoxLayout()
        max_label = QLabel("Max Results:")
        self.max_spin = QSpinBox()
        self.max_spin.setRange(1,100)
        self.max_spin.setValue(self.max_results)
        self.max_spin.valueChanged.connect(lambda val: setattr(self, 'max_results', val))
        max_layout.addWidget(max_label)
        max_layout.addWidget(self.max_spin)
        control_layout.addLayout(max_layout)
        self.features_btn = QPushButton("Select Features")
        self.features_btn.clicked.connect(self.open_feature_dialog)
        control_layout.addWidget(self.features_btn)
        self.feature_importance_btn = QPushButton("Feature Importance")
        self.feature_importance_btn.clicked.connect(self.open_feature_importance)
        control_layout.addWidget(self.feature_importance_btn)
        sim_layout = QHBoxLayout()
        sim_label = QLabel("Similarity Measure:")
        self.similarity_combo = QComboBox()
        self.similarity_combo.addItems(["Cosine", "Cross-bin", "Jensen-Shannon", "Jaccard"])
        sim_layout.addWidget(sim_label)
        sim_layout.addWidget(self.similarity_combo)
        control_layout.addLayout(sim_layout)
        self.search_btn = QPushButton("Search Similar Images")
        self.search_btn.clicked.connect(self.search_similar_images)
        control_layout.addWidget(self.search_btn)
        self.feedback_btn = QPushButton("Relevance Feedback")
        self.feedback_btn.clicked.connect(self.relevance_feedback_search)
        control_layout.addWidget(self.feedback_btn)
        top_layout.addLayout(control_layout)
        main_layout.addLayout(top_layout)
        self.results_area = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QGridLayout()
        self.results_widget.setLayout(self.results_layout)
        self.results_area.setWidget(self.results_widget)
        self.results_area.setWidgetResizable(True)
        main_layout.addWidget(self.results_area)
        self.setLayout(main_layout)

    def delete_query(self):
        self.query_label.clear()
        self.query_features = None
        self.query_image_path = None

    def clear_results(self):
        for i in reversed(range(self.results_layout.count())):
            widgetToRemove = self.results_layout.itemAt(i).widget()
            self.results_layout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
        self.result_widgets = []

    def load_query_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "",
                                                  "Image Files (*.png *.jpg *.bmp *.jpeg *.gif)", options=options)
        if fileName:
            self.query_image_path = fileName
            pixmap = QPixmap(fileName)
            pixmap = pixmap.scaled(300,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_label.setPixmap(pixmap)
            self.query_features = self.feature_extractor.extract_features(fileName)
            self.clear_results()

    def open_canvas_dialog(self):
        dialog = EnhancedCanvasQueryDialog(self)
        if dialog.exec_():
            qimage = dialog.get_canvas_image()
            image_bgr = qimage_to_bgr(qimage)
            self.query_features = self.feature_extractor.extract_features_from_image(image_bgr)
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(300,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_label.setPixmap(pixmap)
            self.clear_results()
            QMessageBox.information(self, "Canvas Query", "Canvas query submitted.")

    def open_containment_dialog(self):
        dialog = EnhancedContainmentQueryDialog(self)
        if dialog.exec_():
            selected_pixmap = dialog.get_selected_region()
            if selected_pixmap is None:
                QMessageBox.warning(self, "Containment Query", "No region was selected.")
                return
            qimage = selected_pixmap.toImage()
            image_bgr = qimage_to_bgr(qimage)
            self.query_features = self.feature_extractor.extract_features_from_image(image_bgr)
            pixmap = selected_pixmap.scaled(300,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_label.setPixmap(pixmap)
            self.clear_results()
            QMessageBox.information(self, "Containment Query", "Containment query submitted.")

    def open_feature_dialog(self):
        dialog = FeatureSelectionDialog(self, initial_selection=self.selected_features)
        if dialog.exec_():
            self.selected_features = dialog.get_selected_features()
            # Retrain PCA with the new selection of features:
            self.pca = train_pca(self.database, self.selected_features, n_components=256)
            QMessageBox.information(self, "Features Selected", "Selected features: " + ", ".join(self.selected_features))

    def open_feature_importance(self):
        dialog = FeatureImportanceDialog(self.selected_features, self)
        if dialog.exec_():
            new_weights = dialog.get_new_weights()
            from features import FEATURE_WEIGHTS
            for feat, weight in new_weights.items():
                FEATURE_WEIGHTS[feat] = weight
            QMessageBox.information(self, "Feature Importance", "Feature weights updated.")

    def search_similar_images(self):
        if self.query_features is None:
            QMessageBox.warning(self, "Warning", "Please provide a query using one of the query methods.")
            return
        sim_measure = self.similarity_combo.currentText()
        results = search_similar(self.query_features, self.database, self.selected_features,
                                 top_k=self.max_results, similarity=sim_measure, pca=self.pca)
        self.clear_results()
        row, col = 0, 0
        for img_path, distance in results:
            img_widget = ClickableImage(img_path, self)
            pixmap = QPixmap(img_path)
            pixmap = pixmap.scaled(200,200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_widget.setPixmap(pixmap)
            img_widget.setToolTip(f"Path: {img_path}\nDistance: {distance:.3f}\nClick to select for feedback")
            self.results_layout.addWidget(img_widget, row, col)
            self.result_widgets.append(img_widget)
            col += 1
            if col >= 5:
                col = 0
                row += 1

    def relevance_feedback_search(self):
        feedback_paths = [widget.file_path for widget in self.result_widgets if widget.selected]
        if not feedback_paths:
            QMessageBox.information(self, "Relevance Feedback", "No images selected for feedback.")
            return
        feedback_features = []
        for path in feedback_paths:
            if path in self.database:
                feedback_features.append(self.database[path])
            else:
                print(f"Feedback image {path} not found in database.")
        if feedback_features:
            new_query_features = combine_query_feedback(self.query_features, feedback_features, self.selected_features)
            for key in self.selected_features:
                self.query_features[key] = new_query_features[key]
            QMessageBox.information(self, "Relevance Feedback", "Relevance feedback applied. Searching again...")
            self.search_similar_images()