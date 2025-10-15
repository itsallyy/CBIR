from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox,
    QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QColorDialog
)
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QPoint, QRect


class EnhancedCanvasQueryDialog(QDialog):
    
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Query Canvas")
        self.resize(640, 720)
        main_layout = QVBoxLayout()
        toolbar = QHBoxLayout()
        self.pen_btn = QPushButton("Pen")
        self.pen_btn.clicked.connect(lambda: self.canvas.set_mode("pen"))
        toolbar.addWidget(self.pen_btn)
        self.line_btn = QPushButton("Line")
        self.line_btn.clicked.connect(lambda: self.canvas.set_mode("line"))
        toolbar.addWidget(self.line_btn)
        self.rect_btn = QPushButton("Rectangle")
        self.rect_btn.clicked.connect(lambda: self.canvas.set_mode("rect"))
        toolbar.addWidget(self.rect_btn)
        self.ellipse_btn = QPushButton("Ellipse")
        self.ellipse_btn.clicked.connect(lambda: self.canvas.set_mode("ellipse"))
        toolbar.addWidget(self.ellipse_btn)
        self.triangle_btn = QPushButton("Triangle")
        self.triangle_btn.clicked.connect(lambda: self.canvas.set_mode("triangle"))
        toolbar.addWidget(self.triangle_btn)
        self.bucket_btn = QPushButton("Bucket Fill")
        self.bucket_btn.clicked.connect(lambda: self.canvas.set_mode("bucket"))
        toolbar.addWidget(self.bucket_btn)
        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.clicked.connect(lambda: self.canvas.set_mode("eraser"))
        toolbar.addWidget(self.eraser_btn)
        toolbar.addWidget(QLabel("Pen Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20)
        self.width_spin.setValue(3)
        self.width_spin.valueChanged.connect(lambda val: self.canvas.set_pen_width(val))
        toolbar.addWidget(self.width_spin)
        self.fill_chk = QCheckBox("Fill")
        self.fill_chk.stateChanged.connect(lambda state: self.canvas.set_fill_shape(state == Qt.Checked))
        toolbar.addWidget(self.fill_chk)
        self.color_btn = QPushButton("Color")
        self.color_btn.clicked.connect(self.choose_color)
        toolbar.addWidget(self.color_btn)
        main_layout.addLayout(toolbar)
        from canvas_widgets import EnhancedCanvasWidget
        self.canvas = EnhancedCanvasWidget(600, 600)
        main_layout.addWidget(self.canvas)
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.canvas.clear)
        btn_layout.addWidget(self.clear_btn)
        self.submit_btn = QPushButton("Submit Drawing")
        self.submit_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.submit_btn)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)
    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_pen_color(color)
    def get_canvas_image(self):
        return self.canvas.get_drawing()


class EnhancedContainmentQueryDialog(QDialog):
    
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Containment Query")
        self.resize(640, 720)
        main_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Image for Containment Query")
        self.load_btn.clicked.connect(self.load_image)
        main_layout.addWidget(self.load_btn)
        self.image_label = None
        sel_btn_layout = QHBoxLayout()
        self.clear_sel_btn = QPushButton("Clear Selection")
        self.clear_sel_btn.clicked.connect(self.clear_selection)
        sel_btn_layout.addWidget(self.clear_sel_btn)
        main_layout.addLayout(sel_btn_layout)
        btn_layout = QHBoxLayout()
        self.submit_btn = QPushButton("Submit Region")
        self.submit_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.submit_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)
        self.selected_pixmap = None
        
    def load_image(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                  "Image Files (*.png *.jpg *.bmp *.jpeg *.gif)")
        if fileName:
            pixmap = QPixmap(fileName)
            pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if self.image_label is None:
                self.image_label = RegionSelectLabel(pixmap, self)
                self.layout().insertWidget(1, self.image_label)
            else:
                self.image_label.setPixmap(pixmap)
            self.selected_pixmap = pixmap
   
    def clear_selection(self):
        if self.image_label:
            self.image_label.clear_selection()
    
    def get_selected_region(self):
        if self.image_label and self.image_label.get_selection_rect() is not None:
            return self.image_label.pixmap().copy(self.image_label.get_selection_rect())
        return None


class RegionSelectLabel(QLabel):
    
    
    def __init__(self, pixmap=None, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        self.moving = False
        self.offset = QPoint(0, 0)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selection_rect is not None and self.selection_rect.contains(event.pos()):
                self.moving = True
                self.offset = event.pos() - self.selection_rect.topLeft()
            else:
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.moving = False
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.moving and self.selection_rect is not None:
            topLeft = event.pos() - self.offset
            self.selection_rect.moveTo(topLeft)
            self.update()
        elif not self.moving and self.start_point is not None:
            self.end_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.moving:
                self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.moving = False
            self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_rect is not None:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
    
    def get_selection_rect(self):
        return self.selection_rect
    
    def clear_selection(self):
        self.selection_rect = None
        self.start_point = None
        self.end_point = None
        self.update()


class FeatureImportanceDialog(QDialog):
    
    
    def __init__(self, selected_features, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Importance")
        self.resize(300, 400)
        self.selected_features = selected_features
        self.new_weights = {}
        layout = QVBoxLayout()
        self.sliders = {}
        for feature in self.selected_features:
            hlayout = QHBoxLayout()
            label = QLabel(feature.capitalize())
            hlayout.addWidget(label)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            from features import FEATURE_WEIGHTS
            spin.setValue(FEATURE_WEIGHTS.get(feature, 1.0))
            self.sliders[feature] = spin
            hlayout.addWidget(spin)
            layout.addLayout(hlayout)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def get_new_weights(self):
        for feature, spin in self.sliders.items():
            self.new_weights[feature] = spin.value()
        return self.new_weights


class DropLabel(QLabel):
    
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drag and Drop Query Image Here")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
   
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    if hasattr(self.parent(), "load_query_image_from_path"):
                        self.parent().load_query_image_from_path(file_path)
                    break


class ClickableImage(QLabel):
    
    
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("border: 2px solid transparent;")
    
    def mousePressEvent(self, event):
        self.selected = not self.selected
        if self.selected:
            self.setStyleSheet("border: 3px solid red;")
        else:
            self.setStyleSheet("border: 2px solid transparent;")


class FeatureSelectionDialog(QDialog):
    
    
    def __init__(self, parent=None, initial_selection=None):
        super().__init__(parent)
        self.setWindowTitle("Select Features for Comparison")
        self.resize(300, 300)
        self.feature_list = ["shape", "color", "color_hist", "sift", "harris", "hog", "glcm", "lbp", "orb", "cnn"]
        self.initial_selection = initial_selection if initial_selection is not None else self.feature_list.copy()
        layout = QVBoxLayout()
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.setChecked(len(self.initial_selection) == len(self.feature_list))
        self.select_all_checkbox.stateChanged.connect(self.toggle_all)
        layout.addWidget(self.select_all_checkbox)
        self.feature_checkboxes = {}
        for feature in self.feature_list:
            cb = QCheckBox(feature.capitalize() + " Features")
            cb.setChecked(feature in self.initial_selection)
            self.feature_checkboxes[feature] = cb
            layout.addWidget(cb)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def toggle_all(self, state):
        check = (state == Qt.Checked)
        for cb in self.feature_checkboxes.values():
            cb.setChecked(check)
    
    def get_selected_features(self):
        return [feature for feature, cb in self.feature_checkboxes.items() if cb.isChecked()]