# Import library
from re import T
#from selectors import EpollSelector
import sys
from MainForm import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import * 
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
#rom BlurWindow.blurWindow import blur, GlobalBlur

import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

import os
import cv2
import time
from PIL import Image
from datetime import datetime
import numpy as np


# For icon

# Class Setting
class Settings():
    # Status set custom title bar
    enableCustomTitleBar = True
    # Toggle maximize windows and minimize windows
    toggleMaximizeWindow = False
    # Set color when extend menu information and setting
    colorShowMoreInformation = "background-color: rgba(144, 202, 249,150);"
    colorButtonSettingSelect = "background-color:rgba(144, 202, 249,150);"

    # MENU SELECTED STYLESHEET
    menuSelectStyleSheet = """
    border-bottom: 2px solid rgba(0, 170, 255, 250);
    """
# END Class setting

class CustomGrip(QWidget):
    def __init__(self, parent, position, disable_color = False):

        # SETUP UI
        QWidget.__init__(self)
        self.parent = parent
        self.setParent(parent)
        self.wi = Widgets()

        # SHOW TOP GRIP
        if position == Qt.TopEdge:
            self.wi.top(self)
            self.setGeometry(0, 0, self.parent.width(), 10)
            self.setMaximumHeight(10)

            # GRIPS
            top_left = QSizeGrip(self.wi.top_left)
            top_right = QSizeGrip(self.wi.top_right)

            # RESIZE TOP
            def resize_top(event):
                delta = event.pos()
                height = max(self.parent.minimumHeight(), self.parent.height() - delta.y())
                geo = self.parent.geometry()
                geo.setTop(geo.bottom() - height)
                self.parent.setGeometry(geo)
                event.accept()
            self.wi.top.mouseMoveEvent = resize_top

            # ENABLE COLOR
            if disable_color:
                self.wi.top_left.setStyleSheet("background: transparent")
                self.wi.top_right.setStyleSheet("background: transparent")
                self.wi.top.setStyleSheet("background: transparent")

        # SHOW BOTTOM GRIP
        elif position == Qt.BottomEdge:
            self.wi.bottom(self)
            self.setGeometry(0, self.parent.height() - 10, self.parent.width(), 10)
            self.setMaximumHeight(10)

            # GRIPS
            self.bottom_left = QSizeGrip(self.wi.bottom_left)
            self.bottom_right = QSizeGrip(self.wi.bottom_right)

            # RESIZE BOTTOM
            def resize_bottom(event):
                delta = event.pos()
                height = max(self.parent.minimumHeight(), self.parent.height() + delta.y())
                self.parent.resize(self.parent.width(), height)
                event.accept()
            self.wi.bottom.mouseMoveEvent = resize_bottom

            # ENABLE COLOR
            if disable_color:
                self.wi.bottom_left.setStyleSheet("background: transparent")
                self.wi.bottom_right.setStyleSheet("background: transparent")
                self.wi.bottom.setStyleSheet("background: transparent")

        # SHOW LEFT GRIP
        elif position == Qt.LeftEdge:
            self.wi.left(self)
            self.setGeometry(0, 10, 10, self.parent.height())
            self.setMaximumWidth(10)

            # RESIZE LEFT
            def resize_left(event):
                delta = event.pos()
                width = max(self.parent.minimumWidth(), self.parent.width() - delta.x())
                geo = self.parent.geometry()
                geo.setLeft(geo.right() - width)
                self.parent.setGeometry(geo)
                event.accept()
            self.wi.leftgrip.mouseMoveEvent = resize_left

            # ENABLE COLOR
            if disable_color:
                self.wi.leftgrip.setStyleSheet("background: transparent")

        # RESIZE RIGHT
        elif position == Qt.RightEdge:
            self.wi.right(self)
            self.setGeometry(self.parent.width() - 10, 10, 10, self.parent.height())
            self.setMaximumWidth(10)

            def resize_right(event):
                delta = event.pos()
                width = max(self.parent.minimumWidth(), self.parent.width() + delta.x())
                self.parent.resize(width, self.parent.height())
                event.accept()
            self.wi.rightgrip.mouseMoveEvent = resize_right

            # ENABLE COLOR
            if disable_color:
                self.wi.rightgrip.setStyleSheet("background: transparent")


    def mouseReleaseEvent(self, event):
        self.mousePos = None

    def resizeEvent(self, event):
        if hasattr(self.wi, 'container_top'):
            self.wi.container_top.setGeometry(0, 0, self.width(), 10)

        elif hasattr(self.wi, 'container_bottom'):
            self.wi.container_bottom.setGeometry(0, 0, self.width(), 10)

        elif hasattr(self.wi, 'leftgrip'):
            self.wi.leftgrip.setGeometry(0, 0, 10, self.height() - 20)

        elif hasattr(self.wi, 'rightgrip'):
            self.wi.rightgrip.setGeometry(0, 0, 10, self.height() - 20)



class Widgets(object):
    def top(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        self.container_top = QFrame(Form)
        self.container_top.setObjectName(u"container_top")
        self.container_top.setGeometry(QRect(0, 0, 500, 10))
        self.container_top.setMinimumSize(QSize(0, 10))
        self.container_top.setMaximumSize(QSize(16777215, 10))
        self.container_top.setFrameShape(QFrame.NoFrame)
        self.container_top.setFrameShadow(QFrame.Raised)
        self.top_layout = QHBoxLayout(self.container_top)
        self.top_layout.setSpacing(0)
        self.top_layout.setObjectName(u"top_layout")
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_left = QFrame(self.container_top)
        self.top_left.setObjectName(u"top_left")
        self.top_left.setMinimumSize(QSize(10, 10))
        self.top_left.setMaximumSize(QSize(10, 10))
        self.top_left.setCursor(QCursor(Qt.SizeFDiagCursor))
        self.top_left.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.top_left.setFrameShape(QFrame.NoFrame)
        self.top_left.setFrameShadow(QFrame.Raised)
        self.top_layout.addWidget(self.top_left)
        self.top = QFrame(self.container_top)
        self.top.setObjectName(u"top")
        self.top.setCursor(QCursor(Qt.SizeVerCursor))
        self.top.setStyleSheet(u"background-color: rgb(85, 255, 255);")
        self.top.setFrameShape(QFrame.NoFrame)
        self.top.setFrameShadow(QFrame.Raised)
        self.top_layout.addWidget(self.top)
        self.top_right = QFrame(self.container_top)
        self.top_right.setObjectName(u"top_right")
        self.top_right.setMinimumSize(QSize(10, 10))
        self.top_right.setMaximumSize(QSize(10, 10))
        self.top_right.setCursor(QCursor(Qt.SizeBDiagCursor))
        self.top_right.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.top_right.setFrameShape(QFrame.NoFrame)
        self.top_right.setFrameShadow(QFrame.Raised)
        self.top_layout.addWidget(self.top_right)

    def bottom(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        self.container_bottom = QFrame(Form)
        self.container_bottom.setObjectName(u"container_bottom")
        self.container_bottom.setGeometry(QRect(0, 0, 500, 10))
        self.container_bottom.setMinimumSize(QSize(0, 10))
        self.container_bottom.setMaximumSize(QSize(16777215, 10))
        self.container_bottom.setFrameShape(QFrame.NoFrame)
        self.container_bottom.setFrameShadow(QFrame.Raised)
        self.bottom_layout = QHBoxLayout(self.container_bottom)
        self.bottom_layout.setSpacing(0)
        self.bottom_layout.setObjectName(u"bottom_layout")
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_left = QFrame(self.container_bottom)
        self.bottom_left.setObjectName(u"bottom_left")
        self.bottom_left.setMinimumSize(QSize(10, 10))
        self.bottom_left.setMaximumSize(QSize(10, 10))
        self.bottom_left.setCursor(QCursor(Qt.SizeBDiagCursor))
        self.bottom_left.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.bottom_left.setFrameShape(QFrame.NoFrame)
        self.bottom_left.setFrameShadow(QFrame.Raised)
        self.bottom_layout.addWidget(self.bottom_left)
        self.bottom = QFrame(self.container_bottom)
        self.bottom.setObjectName(u"bottom")
        self.bottom.setCursor(QCursor(Qt.SizeVerCursor))
        self.bottom.setStyleSheet(u"background-color: rgb(85, 170, 0);")
        self.bottom.setFrameShape(QFrame.NoFrame)
        self.bottom.setFrameShadow(QFrame.Raised)
        self.bottom_layout.addWidget(self.bottom)
        self.bottom_right = QFrame(self.container_bottom)
        self.bottom_right.setObjectName(u"bottom_right")
        self.bottom_right.setMinimumSize(QSize(10, 10))
        self.bottom_right.setMaximumSize(QSize(10, 10))
        self.bottom_right.setCursor(QCursor(Qt.SizeFDiagCursor))
        self.bottom_right.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.bottom_right.setFrameShape(QFrame.NoFrame)
        self.bottom_right.setFrameShadow(QFrame.Raised)
        self.bottom_layout.addWidget(self.bottom_right)

    def left(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        self.leftgrip = QFrame(Form)
        self.leftgrip.setObjectName(u"left")
        self.leftgrip.setGeometry(QRect(0, 10, 10, 480))
        self.leftgrip.setMinimumSize(QSize(10, 0))
        self.leftgrip.setCursor(QCursor(Qt.SizeHorCursor))
        self.leftgrip.setStyleSheet(u"background-color: rgb(255, 121, 198);")
        self.leftgrip.setFrameShape(QFrame.NoFrame)
        self.leftgrip.setFrameShadow(QFrame.Raised)

    def right(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(500, 500)
        self.rightgrip = QFrame(Form)
        self.rightgrip.setObjectName(u"right")
        self.rightgrip.setGeometry(QRect(0, 0, 10, 500))
        self.rightgrip.setMinimumSize(QSize(10, 0))
        self.rightgrip.setCursor(QCursor(Qt.SizeHorCursor))
        self.rightgrip.setStyleSheet(u"background-color: rgb(255, 0, 127);")
        self.rightgrip.setFrameShape(QFrame.NoFrame)
        self.rightgrip.setFrameShadow(QFrame.Raised)

# Class MainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        # Initialize
        QMainWindow.__init__(self)
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        Settings.enableCustomTitleBar = True
        # Tun off status
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # Tranference background
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Hide processBar
        # UIFunctions.setValueProcessBar(self, 0, self.uic.labelPercentageRotateImage, self.uic.circularProgressRotateImage)
        # self.uic.frameProcessBar.hide()
        # Blur background
        #blur(self.winId())
        # Button on status bar
        self.uic.btnMaximize.clicked.connect(lambda: UIFunctions.maximizeRestore(self))
        self.uic.btnMinimize.clicked.connect(lambda: self.showMinimized())
        self.uic.btnClose.clicked.connect(lambda: self.close())
        
        # Left menu
        # Set default stack
        # self.uic.stackedWidget.setCurrentWidget(self.uic.stackRotateImage)
        # self.uic.btnRotatePicture.setStyleSheet(UIFunctions.selectMenu(self.uic.btnRotatePicture.styleSheet()))
        # Set button click event
        # self.uic.btnAdjustImage.clicked.connect(lambda: UIFunctions.buttonClick(self))
        # self.uic.btnCheckMissing.clicked.connect(lambda: UIFunctions.buttonClick(self))
        # self.uic.btnCovertClass.clicked.connect(lambda: UIFunctions.buttonClick(self))
        # self.uic.btnCovertFormatLabel.clicked.connect(lambda: UIFunctions.buttonClick(self))
        # self.uic.btnCovertFormatPicture.clicked.connect(lambda: UIFunctions.buttonClick(self))
        # self.uic.btnRotatePicture.clicked.connect(lambda: UIFunctions.buttonClick(self))
        UIFunctions.uiDefinitions(self)
        
        
        # # Stack rotate image
        # self.uic.btnSelectFolderImageRotateImage.clicked.connect(lambda: stackRotateImageFunction.getFolderImageRotate(self))
        # self.uic.btnSelectFolderLabelRotateImage.clicked.connect(lambda: stackRotateImageFunction.getFolderLabelRotate(self))
        # self.uic.btnSelectFolderSaveRotateImage.clicked.connect(lambda: stackRotateImageFunction.getFolderSaveRotate(self))
        # self.uic.checkBoxAllRotateImage.stateChanged.connect(lambda: stackRotateImageFunction.setStatusQCheckBoxAll(self))
        # self.uic.checkBox90RotateImage.stateChanged.connect(lambda: stackRotateImageFunction.setStatusQCheckBox(self))
        # self.uic.checkBox270RotateImage.stateChanged.connect(lambda: stackRotateImageFunction.setStatusQCheckBox(self))
        # self.uic.checkBox180RotateImage.stateChanged.connect(lambda: stackRotateImageFunction.setStatusQCheckBox(self))
        # self.uic.btnStartProcessRotateImage.clicked.connect(lambda: stackRotateImageFunction.startProcessRotateImage(self))
        # Load model
        self.modelViL = ViLTInference()

        # # Stack check label and picture
        self.uic.btnSelectImageToInference.clicked.connect(lambda: stackDemoModel.getImageInference(self))
        # self.uic.btnSelectFolderLabelCheckMissing.clicked.connect(lambda: stackCheckMissingFunction.getFolderLabelCheckMissing(self))
        self.uic.btnStartProcessCheckModel.clicked.connect(lambda: stackDemoModel.startProcessInference(self))
        # Screen shot
        self.uic.btnSceenshot.clicked.connect(lambda: UIFunctions.captureScreen(self))
        # # Stack convert picture image
        # self.uic.btnSelectFolderConvertPictureFormat.clicked.connect(lambda: stackConvertFormatPictureFunction.getFolderImageConvertFormat(self))
        # self.uic.btnSelectFoldeImageConvertFormatSave.clicked.connect(lambda: stackConvertFormatPictureFunction.getFoldeImageConvertFormatSave(self))
        # self.uic.btnStartProcessConvertPictureFormat.clicked.connect(lambda: stackConvertFormatPictureFunction.startProcessConvertImageFormat(self))
        # self.uic.progressBarConvertPictureFormat.hide()
        # self.uic.btnStartProcessConvertPictureFormat.show()
    
    # Mouse event click
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')
    # Resize Event
    # def resizeEvent(self, event):
        # Update Size Grips
        # UIFunctions.resizeGrips(self)

    # Create message box
    def createMessage(self, paramTitle, paramText, paramButton, paramPixmap):
        msg = QMessageBox(self)
        msg.setWindowTitle(paramTitle)
        msg.setText(paramText)
        msg.setStandardButtons(paramButton)
        msg.setIconPixmap(QtGui.QPixmap(paramPixmap))

        x = msg.exec_()
        return x


class ScrollMessageBox(QMessageBox):
    def __init__(self,  paramTitle, paramText, paramButton , *args, **kwargs):
        QMessageBox.__init__(self, *args, **kwargs)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # Set user interface
        self.setWindowTitle(paramTitle)
        self.setStandardButtons(paramButton)
        # icon = QtGui.QIcon()
        
        # icon.addPixmap(QtGui.QPixmap(":Images/LogoVin2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        # icon = QIcon('Images/LogoVin2.png')
        
        # self.setWindowIcon(icon)
        # END set user interface

        # Set content
        self.content = QWidget()
        scroll.setWidget(self.content)
        lay = QVBoxLayout(self.content)
        numberOfRow = 0
        for item in paramText.split("\n"):
            numberOfRow = numberOfRow + 1
            if item != '':
                lay.addWidget(QLabel(item, self))
        #END Set content
        minHeight = numberOfRow*25

        if minHeight > 400:
            minHeight = 400
        # Set layout and style sheet
        self.layout().addWidget(scroll, 0, 0, 1, self.layout().columnCount())

        styleSheet = """
                QScrollArea{min-width:400 px; min-height: %spx};
                font: 12pt "UTM Avo";
                """ % minHeight

        self.setStyleSheet(styleSheet)
        # END Set layout and style sheet

    def returnResultClick(self):
        result = self.exec_()
        return result

    # END createMessage

# Get link directory from click button
def getLinkFromButton():
    # dialog = QFileDialog()
    # folderPath, _ = dialog.getOpenFileNames(None, "Select Folder")

    # Tạo một đối tượng QFileDialog
    dialog = QFileDialog()

    # Thiết lập tiêu đề của hộp thoại (tuỳ chọn)
    dialog.setWindowTitle('Chọn một tệp hình ảnh')

    # Thiết lập bộ lọc chỉ cho phép chọn các tệp hình ảnh (ví dụ: PNG, JPG)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter("Hình ảnh (*.png *.jpg *.jpeg *.bmp *.gif)")

    # Sử dụng phương thức getOpenFileName để lấy tên của tệp hình ảnh đã chọn
    folderPath, _ = dialog.getOpenFileName()

    return folderPath



# Class UIFunctions
class UIFunctions(MainWindow):
    # Maximize button
    def maximizeRestore(self):
        # Check current status window show
        status = Settings.toggleMaximizeWindow
        # If window is not maximize
        if status == False:
            # Change status window show
            Settings.toggleMaximizeWindow = True
            # Show window in maximize mode
            self.showNormal()
            # Set margin
            self.uic.styleSheet.setContentsMargins(0, 0, 0, 0)
            self.uic.btnMaximize.setToolTip("Restore")
            # Set icon for button
            self.uic.btnMaximize.setStyleSheet("""
                QPushButton{
                border-image: url(:/Images/Images/maxHold.png);
                }
                QPushButton::hover{
                border-image:url(:/Images/Images/normalScreen.png);
                }
                QPushButton::pressed{
                border-image:url(:/Images/Images/normalScreenPush.png);
                }
                """)
        # End maximize button
        else:
            # Change status window show
            Settings.toggleMaximizeWindow = False
            # Show window in normal mode
            self.showNormal()
            # Set margin
            self.uic.styleSheet.setContentsMargins(0, 0, 0, 0)
            self.uic.btnMaximize.setToolTip("Maximize")
            # Set icon for button
            self.uic.btnMaximize.setStyleSheet("""
                QPushButton{
                border-image: url(:/Images/Images/maxHold.png);
                }
                QPushButton::hover{
                border-image:url(:/Images/Images/maximizeScreen.png);
                }
                QPushButton::pressed{
                border-image:url(:/Images/Images/maximizsePush.png);
                }
                """)
    # END Maximize button

    # Select Menu
    def selectMenu(styleSheetRaw):
        newStyleSheet = styleSheetRaw + Settings.menuSelectStyleSheet
        return newStyleSheet
    # END Select menu

    # Deselect style sheet
    def deselectMenu(styleSheetRaw):
        newStyleSheet = styleSheetRaw.replace(Settings.menuSelectStyleSheet, "")
        return newStyleSheet
    # END Deselect style sheet

    # Start selection
    def selectStandardMenu(self, widget):
        for w in self.uic.toolBar.findChildren(QPushButton):
            if w.objectName() == widget:
                w.setStyleSheet(UIFunctions.selectMenu(w.styleSheet()))
    # END Selection

    # Reset selection
    def resetStyle(self, widget):
        for w in self.uic.toolBar.findChildren(QPushButton):
            if w.objectName() != widget:
                w.setStyleSheet(UIFunctions.deselectMenu(w.styleSheet()))
    # END reset selection

    # Return status
    def returStatus(self):
        return Settings.toggleMaximizeWindow

    # Set status
    def setStatus(self, status):
        Settings.toggleMaximizeWindow = status

    def uiDefinitions(self):
        # dobleClickMaximizeRestore
        def doubleClickMaximizeRestore(event):
            if event.type() == QEvent.MouseButtonDblClick:
                QTimer.singleShot(250, lambda: UIFunctions.maximizeRestore(self))
        self.uic.menuBar.mouseDoubleClickEvent = doubleClickMaximizeRestore
        # END dobleClickMaximizeRestore

        # Move window
        def moveWindow(event):
            # If maximized change to normal
            if UIFunctions.returStatus(self):
                UIFunctions.maximizeRestore(self)
            # Move window
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()
        self.uic.menuBar.mouseMoveEvent = moveWindow
        # END Move window

        # self.leftGrip = CustomGrip(self, Qt.LeftEdge, True)
        # self.rightGrip = CustomGrip(self, Qt.RightEdge, True)
        # self.topGrip = CustomGrip(self, Qt.TopEdge, True)
        # self.bottomGrip = CustomGrip(self, Qt.BottomEdge, True)

    # def resizeGrips(self):
        # if Settings.enableCustomTitleBar:
            # self.leftGrip.setGeometry(0, 10, 10, self.height())
            # self.rightGrip.setGeometry(self.width() - 10, 10, 10, self.height())
            # self.topGrip.setGeometry(0, 0, self.width(), 10)
            # self.bottomGrip.setGeometry(0, self.height() - 10, self.width(), 10)
    
    # Set value process bar
    def setValueProcessBar(self, value, paramUICLabel, paramUICProcessBar):
        QCoreApplication.processEvents()
        # Set percent for label
        color = "rgba(85, 170, 255, 255)"
        sliderValue = int(value)
        htmlText = """<p align="center"><span style=" font-size:50pt;">{VALUE}</span><span style=" font-size:40pt; vertical-align:super;">%</span></p>"""
        paramUICLabel.setText(htmlText.replace("{VALUE}", str(sliderValue)))

        
        # END set percent for label
        # Process bar stylesheet
        styleSheet = """
        QFrame{
            border-radius: 110px;
            background-color: qconicalgradient(cx:0.5, cy:0.5, angle:90, stop:{STOP_1} rgba(255, 0, 127, 0), stop:{STOP_2} {COLOR});
        }
        """
        # stop works of 1.000 to 0.000
        progress = (100 - value) / 100.0

        stopFirst = str(progress - 0.001)
        stopLast = str(progress)

        if value == 100:
            stopFirst = "1.000"
            stopLast = "1.000"
        
        newStylesheet = styleSheet.replace("{STOP_1}", stopFirst).replace("{STOP_2}", stopLast).replace("{COLOR}", color)
        paramUICProcessBar.setStyleSheet(newStylesheet)
    # END set value process bar

    # Set Disable buton when processing

    # Set disable when processing
    def setButtonIsDisable(self):
        UIFunctions.setValueProcessBar(self, 0, self.uic.labelPercentageRotateImage, self.uic.circularProgressRotateImage)
        # Set status is disable
        self.uic.toolBar.setEnabled(False)
        self.uic.menuBar.setEnabled(False)

        self.uic.processFrameRotateImage.setEnabled(False)

        self.uic.processFrameCheckMissing.setEnabled(False)

        self.uic.processFrameConvertPictureFormat.setEnabled(False)
        self.uic.progressBarConvertPictureFormat.show()
        self.uic.btnStartProcessConvertPictureFormat.hide()


    # Set enable when processed
    def setButtonIsEnable(self):
        self.uic.toolBar.setEnabled(True)
        self.uic.menuBar.setEnabled(True)

        self.uic.processFrameRotateImage.setEnabled(True)

        self.uic.processFrameCheckMissing.setEnabled(True)

        self.uic.processFrameConvertPictureFormat.setEnabled(True)
        self.uic.progressBarConvertPictureFormat.hide()
        self.uic.btnStartProcessConvertPictureFormat.show()
    # END enable when processed

    # Set Image interface
    def setImageInterface(self):
        imagePath = self.uic.plainTextImageToInference.toPlainText()
        if imagePath.lower().endswith(('.png', '.jpg','.webp', '.jpeg')):
            # Đọc hình ảnh từ OpenCV
            openCVImage = cv2.imread(imagePath)
            # Resize hình ảnh với kích thước 510x300 (duy trì tỉ lệ gốc và thêm padding nếu cần)
            resizedImage = UIFunctions.resizeAndPad(self, openCVImage, width=510, height=300)
            #
            # Chuyển đổi hình ảnh từ OpenCV sang QPixmap
            pixmapImage = UIFunctions.convertCVImageToQpixmap(self, resizedImage)


            # pixmap = QPixmap(imagePath)
            self.uic.lblFigureInference.setPixmap(pixmapImage)

        else:
            self.uic.lblFigureInference.setPixmap(QtGui.QPixmap(":/Images/Images/NotFound.png"))
    # END Set Image interface

    # Resize and Pad
    def resizeAndPad(self, image, width, height):
        h, w, _ = image.shape
        aspect_ratio = w / h

        if aspect_ratio > width / height:
            new_w = width
            new_h = int(width / aspect_ratio)
        else:
            new_h = height
            new_w = int(height * aspect_ratio)

        resized_image = cv2.resize(image, (new_w, new_h))

        # Tạo một hình ảnh mới với kích thước cố định và chứa hình ảnh đã resize
        result = np.full((height, width, 3), (54, 51, 50), dtype=np.uint8)

        y_offset = (height - new_h) // 2
        x_offset = (width - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized_image

        return result
    # END resize and Pad

    # Convert image to QPixmap
    def convertCVImageToQpixmap(self, cv_image):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)
    # END Convert image to QPixmap


    # Take screenshot
    def captureScreen(self):
        # Chụp màn hình của cửa sổ chính của ứng dụng
        screenshot = self.centralWidget().grab()
        # Lưu hình ảnh đã chụp vào tệp tin
        #
        dirPath = 'Screenshot'
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        #
        current_time = datetime.now()
        timestamp = current_time.strftime("%y%m%d_%H%M%S")
        #
        pathToImageScreenShot = f"{dirPath}/{timestamp}_Screenshot.png"
        screenshot.save(pathToImageScreenShot)
        replyMessBoxInfor =  self.createMessage("Complete!","Image have been save!",QMessageBox.Ok, ":/Images/Images/Success.png")

    # END take screenshot


# Class stack Rotate Image Function
class stackDemoModel(MainWindow):
    # Get folder path from butoon
    def getImageInference(self):
        pathImageInference = getLinkFromButton()
        self.uic.plainTextImageToInference.setPlainText(pathImageInference)
        UIFunctions.setImageInterface(self)


    # END Get folder path from butoon

    def startProcessInference(self):
        # Get text
        pathImageInference = self.uic.plainTextImageToInference.toPlainText()
        contentOfQuestion = self.uic.plainTextQuestion.toPlainText()
        
        

        # Check path exist
        if os.path.exists(pathImageInference) and pathImageInference.lower().endswith(('.png', '.jpg', '.webp', '.jpeg')):

            UIFunctions.setImageInterface(self)
            imageInference = Image.open(pathImageInference)
            imageInference = imageInference.convert("RGB")
            resultInference = self.modelViL(imageInference, contentOfQuestion)
            # List string result to update to ui
            listResultString = []
            #
            for anwser, prob in resultInference.items():
                resultElement = f'{round(prob*100)} % - {anwser}'
                # resultElement = f'{anwser}'
                listResultString.append(resultElement)
            # Show to ui
            self.uic.lblAnswerTop1.setText(listResultString[0])
            self.uic.lblAnswerTop2.setText(listResultString[1])
            self.uic.lblAnswerTop3.setText(listResultString[2])
            self.uic.lblAnswerTop4.setText(listResultString[3])
            self.uic.lblAnswerTop5.setText(listResultString[4])

            replyMessBoxInfor =  self.createMessage("Complete inference!","Complete",QMessageBox.Ok, ":/Images/Images/Success.png")
            # if replyMessBoxInfor.returnResultClick() == QMessageBox.Yes:
            #     replyMessBoxInfor.close()
                # Set status is enable
            
        else:
            replyMessBoxInfor =  self.createMessage("Warning","Please check your directory",QMessageBox.Ok, ":/Images/Images/Error.png")
            
        # UIFunctions.setButtonIsEnable(self)
        # self.uic.frameProcessBar.hide()
        # stackRotateImageFunction.setValue(self, 30, self.uic.labelPercentageRotateImage, self.uic.circularProgressRotateImage, "rgba(85, 170, 255, 255)")
        # Set image interface

# END Class stack Rotate Image Function


# Class ViLModel
class ViLTInference:
    def __init__(self):
        weight = "dandelin/vilt-b32-finetuned-vqa"
        print("Loading: {}".format(weight))

        self.processor = ViltProcessor.from_pretrained(weight)
        self.model = ViltForQuestionAnswering.from_pretrained(weight)

    def __call__(self, image, text):
        print('Start Inference')
        encoding = self.processor(image, text, return_tensors='pt')

        outputs = self.model(**encoding)
        logits = outputs.logits
        predicted_classes = torch.sigmoid(logits)

        answer_dict = dict()
        probs, classes = torch.topk(predicted_classes, 5)
        for prob, classes_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
            answer_dict[self.model.config.id2label[classes_idx]] = prob

        return answer_dict
# END Class ViLModel




if __name__ == "__main__":
    app = QApplication(sys.argv)
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'Images/LogoVin2.png')
    app.setWindowIcon(QIcon(path))

    main_win = MainWindow()
    main_win.move(70,40)
    main_win.show()
    sys.exit(app.exec())