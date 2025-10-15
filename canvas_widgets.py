from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect


class EnhancedCanvasWidget(QWidget):
    
    
    def __init__(self, width=600, height=600, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.image = QImage(self.size(), QImage.Format_RGBA8888)
        self.image.fill(Qt.white)
        self.last_point = None
        self.start_point = None
        self.mode = "pen"  # Modes: "pen", "line", "rect", "ellipse", "triangle", "bucket", "eraser"
        self.pen_color = Qt.black
        self.pen_width = 3
        self.fill_shape = False
        self.temp_shape = None

    def set_mode(self, mode):
        self.mode = mode

    def set_pen_color(self, color):
        self.pen_color = color

    def set_pen_width(self, width):
        self.pen_width = width

    def set_fill_shape(self, fill):
        self.fill_shape = fill

    def flood_fill(self, x, y, target_color, replacement_color):
        if target_color == replacement_color:
            return
        img = self.image
        width = img.width()
        height = img.height()
        stack = [(x, y)]
        while stack:
            nx, ny = stack.pop()
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            current_color = QColor(img.pixel(nx, ny))
            if current_color == target_color:
                img.setPixel(nx, ny, replacement_color.rgb())
                stack.extend([(nx+1, ny), (nx-1, ny), (nx, ny+1), (nx, ny-1)])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())
        if self.temp_shape:
            pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine)
            painter.setPen(pen)
            if self.fill_shape:
                painter.setBrush(self.pen_color)
            else:
                painter.setBrush(Qt.NoBrush)
            if self.mode in ["rect", "ellipse", "triangle"]:
                if self.mode == "rect":
                    painter.drawRect(self.temp_shape)
                elif self.mode == "ellipse":
                    painter.drawEllipse(self.temp_shape)
                elif self.mode == "triangle":
                    r = self.temp_shape
                    points = [QPoint(r.center().x(), r.top()), QPoint(r.left(), r.bottom()), QPoint(r.right(), r.bottom())]
                    painter.drawPolygon(*points)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.mode == "bucket":
                x, y = event.x(), event.y()
                target = QColor(self.image.pixel(x, y))
                self.flood_fill(x, y, target, QColor(self.pen_color))
            else:
                self.start_point = event.pos()
                self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.mode == "pen":
                painter = QPainter(self.image)
                pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.last_point, event.pos())
                self.last_point = event.pos()
                painter.end()
                self.update()
            elif self.mode == "eraser":
                painter = QPainter(self.image)
                pen = QPen(Qt.white, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.last_point, event.pos())
                self.last_point = event.pos()
                painter.end()
                self.update()
            elif self.mode in ["line", "rect", "ellipse", "triangle"]:
                self.temp_shape = QRect(self.start_point, event.pos()).normalized()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.mode != "bucket":
            painter = QPainter(self.image)
            pen = QPen(self.pen_color if self.mode != "eraser" else Qt.white, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            if self.fill_shape:
                painter.setBrush(self.pen_color if self.mode != "eraser" else Qt.white)
            else:
                painter.setBrush(Qt.NoBrush)
            if self.mode in ["pen", "eraser"]:
                painter.drawLine(self.last_point, event.pos())
            elif self.mode == "line":
                painter.drawLine(self.start_point, event.pos())
            elif self.mode == "rect":
                painter.drawRect(QRect(self.start_point, event.pos()).normalized())
            elif self.mode == "ellipse":
                painter.drawEllipse(QRect(self.start_point, event.pos()).normalized())
            elif self.mode == "triangle":
                r = QRect(self.start_point, event.pos()).normalized()
                points = [QPoint(r.center().x(), r.top()), QPoint(r.left(), r.bottom()), QPoint(r.right(), r.bottom())]
                painter.drawPolygon(*points)
            painter.end()
            self.start_point = None
            self.temp_shape = None
            self.update()

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_drawing(self):
        return self.image