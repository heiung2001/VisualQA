# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainForm.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1160, 720)
        MainWindow.setMinimumSize(QtCore.QSize(1160, 720))
        MainWindow.setMaximumSize(QtCore.QSize(1160, 720))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Images/Images/LogoVin2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.styleSheet = QtWidgets.QWidget(MainWindow)
        self.styleSheet.setStyleSheet("/* Tooltip */\n"
"QToolTip {\n"
"    color: #333;\n"
"    background-color: #7d6dcf;\n"
"    border: 1px solid #CCC;\n"
"    background-image: none;\n"
"    background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"    border: none;\n"
"    border-left: 2px solid rgb(255, 121, 198);\n"
"    text-align: left;\n"
"    padding-left: 8px;\n"
"    margin: 0px;\n"
"}\n"
"/* END Tooltip */\n"
"\n"
"/* Background*/\n"
"#mainBackground {    \n"
"    background-color: rgb(30,30,30);\n"
"    /*border-image:url(:/Images/Images/background1.jpg);*/\n"
"    opacity: 0.6;\n"
"    /*color: #44475a;*/\n"
"    color: rgb(0,0,0);\n"
"    border-radius: 8px;\n"
"}\n"
"\n"
"#containFrame{\n"
"    border-bottom-left-radius: 8px;\n"
"    border-bottom-right-radius: 8px;\n"
"}\n"
"\n"
"QPlainTextEdit{\n"
"    background-color:#323336;\n"
"    border-radius: 20px;\n"
"    padding-left: 20px;\n"
"    padding-top: 8px;\n"
"    padding-bottom: 8px;\n"
"}\n"
"\n"
"QTextEdit{\n"
"    background-color:#323336;\n"
"    border-radius: 20px;\n"
"    padding-left: 20px;\n"
"    padding-top: 8px;\n"
"    padding-bottom: 8px;\n"
"}\n"
"\n"
"/* END Background*/\n"
"\n"
"/* QMessageBox */\n"
"QMessageBox {\n"
"    background-color: #FFFFFF;\n"
"}\n"
"\n"
"QMessageBox QLabel {\n"
"    color: rgb(0, 0, 0);\n"
"    font: 10pt UTM Avo;\n"
"}\n"
"\n"
"QMessageBox QPushButton{\n"
"    width: 80px;\n"
"    height: 30px;\n"
"    color: rgb(0, 0, 0);\n"
"    font: 10pt UTM Avo;\n"
"    border-radius : 5px;\n"
"    border : 1px solid rgb(107, 107, 107);\n"
"    background-color: rgba(255, 255, 255, 255);\n"
"}\n"
"/* END  QMessageBox */\n"
"\n"
"/* MenuBar */\n"
"#menuBar{\n"
"    background-color: #36383c;\n"
"    border-top-left-radius: 8px;\n"
"    border-top-right-radius: 8px;\n"
"    border-bottom: 1px solid #CCC;\n"
"    \n"
"}\n"
"\n"
"#lblSubtle{\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"\n"
"#btnClose.QPushButton{\n"
"border-image: url(:/Images/Images/closeHold.png);\n"
"}\n"
"\n"
"#btnClose.QPushButton::hover{\n"
"border-image: url(:/Images/Images/closeNormal.png);\n"
"}\n"
"\n"
"#btnClose.QPushButton::pressed{\n"
"border-image: url(:/Images/Images/closePush.png);\n"
"}\n"
"\n"
"\n"
"#btnMaximize.QPushButton{\n"
"border-image: url(:/Images/Images/maxHold.png);\n"
"}\n"
"\n"
"#btnMaximize.QPushButton::hover{\n"
"border-image:url(:/Images/Images/maximizeScreen.png);\n"
"}\n"
"\n"
"\n"
"#btnMaximize.QPushButton::pressed{\n"
"border-image:url(:/Images/Images/maximizsePush.png);\n"
"}\n"
"\n"
"#btnMinimize.QPushButton{\n"
"border-image:url(:/Images/Images/miniHold.png);\n"
"}\n"
"\n"
"#btnMinimize.QPushButton::hover{\n"
"border-image: url(:/Images/Images/miniNormal.png);\n"
"}\n"
"\n"
"\n"
"#btnMinimize.QPushButton::pressed{\n"
"border-image:url(:/Images/Images/miniPush.png);\n"
"}\n"
"\n"
"/* END MenuBar */\n"
"\n"
"\n"
"/*Tool bar*/\n"
"#toolBar .QPushButton {\n"
"    background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"    border: none;\n"
"    border-left:     1px solid transparent;\n"
"    background-color:transparent;\n"
"    text-align: center;\n"
"    padding-left: 0px;\n"
"    padding-right: 0px;\n"
"    \n"
"}\n"
"#toolBar .QPushButton:hover {\n"
"    border-bottom: 2px solid rgba(0, 170, 255, 150);\n"
"}\n"
"#toolBar .QPushButton:pressed {    \n"
"    border-bottom: 2px solid rgba(0, 170, 255, 250);\n"
"}\n"
"\n"
"#btnSetting.QPushButton{\n"
"border-image: url(:/Images/Images/Setting1.png);\n"
"}\n"
"\n"
"#btnSetting.QPushButton::hover{\n"
"border-image: url(:/Images/Images/Setting2.png) ;\n"
"}\n"
"\n"
"#btnSetting.QPushButton::pressed{\n"
"border-image: url(:/Images/Images/Setting3.png) ;\n"
"}\n"
"\n"
"#line{\n"
"border-left: 1px solid black;\n"
"border-style: inset;\n"
"}\n"
"/*END Tool bar*/\n"
"\n"
"\n"
"/*Stack Widget Rotate Image*/\n"
"#stackedWidget .QPushButton {\n"
"    background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"    background-color:#004b7d;\n"
"    text-align: center;\n"
"    padding-left: 10px;\n"
"    padding-right: 10px;\n"
"    border-radius: 20px;\n"
"    color: rgb(0,0,0);\n"
"    /*color: rgb(255,255,255);*/\n"
"}\n"
"\n"
"#stackedWidget .QPushButton:disabled {\n"
"    background-color: rgb(200,200,200);\n"
"    color: rgb(120, 120, 120);\n"
"}\n"
"\n"
"#stackedWidget .QPushButton:hover {\n"
"    background-color:#025e9c;\n"
"    color: rgb(0,0,0);\n"
"    /* color: rgb(255,255,255); */\n"
"}\n"
"#stackedWidget .QPushButton:pressed {    \n"
"    background-color:#0079c9;\n"
"    color: rgb(0,0,0);\n"
"    /* color: rgb(255,255,255); */\n"
"}\n"
"\n"
"\n"
"\n"
"/*END Stack Widget Rotate Image*/\n"
"\n"
"/* Layout ChooseDirection */\n"
"\n"
"#layOutChooseDirectionRotateImage .QCheckBox {\n"
"    background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"    background-color:rgba(210, 210, 210, 200);\n"
"    text-align: center;\n"
"    border-radius: 20px;\n"
"    color: rgb(0,0,0);\n"
"}\n"
"#layOutChooseDirectionRotateImage .QCheckBox:hover {\n"
"    background-color:rgba(55, 177, 237, 255);\n"
"}\n"
"#layOutChooseDirectionRotateImage .QCheckBox:pressed {    \n"
"    background-color:rgba(0, 170, 255, 255);\n"
"}\n"
"\n"
"#layOutChooseDirectionRotateImage .QCheckBox:checked{\n"
"    background-color:rgba(0, 170, 255, 255);\n"
"}\n"
"\n"
"#layOutChooseDirectionRotateImage .QCheckBox:disabled {\n"
"    background-color: rgb(200,200,200);\n"
"}\n"
"\n"
"#layOutChooseDirectionRotateImage .QCheckBox::indicator{\n"
"    margin-left:5%;\n"
"    margin-right:5%;\n"
"}\n"
"\n"
"#checkBoxAllRotateImage.QCheckBox::indicator:checked {\n"
"    image: url(:/Images/Images/AllOfThemWhite.png);\n"
"}\n"
"#checkBoxAllRotateImage.QCheckBox::indicator:unchecked {\n"
"    image: url(:/Images/Images/AllOfThemBlack.png);\n"
"}\n"
"#checkBoxAllRotateImage.QCheckBox::indicator:hover {\n"
"    image: url(:/Images/Images/AllOfThemWhite.png);\n"
"}\n"
"\n"
"\n"
"#checkBox90RotateImage.QCheckBox::indicator:checked {\n"
"    image: url(:/Images/Images/90White.png);\n"
"    \n"
"}\n"
"#checkBox90RotateImage.QCheckBox::indicator:unchecked {\n"
"    image:url(:/Images/Images/90Black.png);\n"
"}\n"
"\n"
"#checkBox90RotateImage.QCheckBox::indicator:hover {\n"
"    image: url(:/Images/Images/90White.png);\n"
"}\n"
"\n"
"#checkBox180RotateImage.QCheckBox::indicator:checked {\n"
"    image: url(:/Images/Images/180White.png);\n"
"}\n"
"\n"
"#checkBox180RotateImage.QCheckBox::indicator:unchecked {\n"
"    image:url(:/Images/Images/180Black.png);\n"
"}\n"
"#checkBox180RotateImage.QCheckBox::indicator:hover {\n"
"    image: url(:/Images/Images/180White.png);\n"
"}\n"
"\n"
"#checkBox270RotateImage.QCheckBox::indicator:checked {\n"
"    image: url(:/Images/Images/270White.png);\n"
"}\n"
"#checkBox270RotateImage.QCheckBox::indicator:unchecked {\n"
"    image:url(:/Images/Images/270Black.png);\n"
"}\n"
"#checkBox270RotateImage.QCheckBox::indicator:hover {\n"
"    image: url(:/Images/Images/270White.png);\n"
"}\n"
"\n"
"/* END Layout ChooseDirection */\n"
"\n"
"\n"
"/*Convert picture format*/\n"
"\n"
"#progressBarConvertPictureFormat.QProgressBar{\n"
"    border-radius: 20px;\n"
"    color:black;\n"
"}\n"
"\n"
"#progressBarConvertPictureFormat.QProgressBar::chunk{\n"
"    background-color : rgba(0, 170, 255, 255);\n"
"    border-radius :20px;\n"
"}\n"
"\n"
"\n"
"\n"
"\n"
"\n"
"")
        self.styleSheet.setObjectName("styleSheet")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.styleSheet)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mainBackground = QtWidgets.QFrame(self.styleSheet)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.mainBackground.setFont(font)
        self.mainBackground.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainBackground.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainBackground.setObjectName("mainBackground")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.mainBackground)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.menuBar = QtWidgets.QFrame(self.mainBackground)
        self.menuBar.setMinimumSize(QtCore.QSize(0, 40))
        self.menuBar.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.menuBar.setFont(font)
        self.menuBar.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.menuBar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.menuBar.setObjectName("menuBar")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.menuBar)
        self.horizontalLayout.setContentsMargins(20, 0, 20, 0)
        self.horizontalLayout.setSpacing(15)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnClose = QtWidgets.QPushButton(self.menuBar)
        self.btnClose.setMinimumSize(QtCore.QSize(15, 15))
        self.btnClose.setMaximumSize(QtCore.QSize(15, 15))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.btnClose.setFont(font)
        self.btnClose.setStyleSheet("")
        self.btnClose.setText("")
        self.btnClose.setDefault(False)
        self.btnClose.setObjectName("btnClose")
        self.horizontalLayout.addWidget(self.btnClose)
        self.btnMinimize = QtWidgets.QPushButton(self.menuBar)
        self.btnMinimize.setMinimumSize(QtCore.QSize(15, 15))
        self.btnMinimize.setMaximumSize(QtCore.QSize(15, 15))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.btnMinimize.setFont(font)
        self.btnMinimize.setStyleSheet("")
        self.btnMinimize.setText("")
        self.btnMinimize.setObjectName("btnMinimize")
        self.horizontalLayout.addWidget(self.btnMinimize)
        self.btnMaximize = QtWidgets.QPushButton(self.menuBar)
        self.btnMaximize.setMinimumSize(QtCore.QSize(15, 15))
        self.btnMaximize.setMaximumSize(QtCore.QSize(15, 15))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.btnMaximize.setFont(font)
        self.btnMaximize.setStyleSheet("")
        self.btnMaximize.setText("")
        self.btnMaximize.setObjectName("btnMaximize")
        self.horizontalLayout.addWidget(self.btnMaximize)
        spacerItem = QtWidgets.QSpacerItem(320, 35, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.lblSubtle = QtWidgets.QLabel(self.menuBar)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setBold(True)
        font.setWeight(75)
        self.lblSubtle.setFont(font)
        self.lblSubtle.setStyleSheet("color: rgba(255, 255, 255, 255);")
        self.lblSubtle.setAlignment(QtCore.Qt.AlignCenter)
        self.lblSubtle.setObjectName("lblSubtle")
        self.horizontalLayout.addWidget(self.lblSubtle)
        spacerItem1 = QtWidgets.QSpacerItem(319, 35, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label = QtWidgets.QLabel(self.menuBar)
        self.label.setMinimumSize(QtCore.QSize(80, 20))
        self.label.setMaximumSize(QtCore.QSize(80, 20))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/Images/Images/VinBigdataLogo.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout_2.addWidget(self.menuBar)
        self.containFrame = QtWidgets.QFrame(self.mainBackground)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.containFrame.setFont(font)
        self.containFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.containFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.containFrame.setObjectName("containFrame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.containFrame)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.informationFrame = QtWidgets.QFrame(self.containFrame)
        self.informationFrame.setMinimumSize(QtCore.QSize(40, 0))
        self.informationFrame.setMaximumSize(QtCore.QSize(40, 16777215))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.informationFrame.setFont(font)
        self.informationFrame.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.informationFrame.setAutoFillBackground(False)
        self.informationFrame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.informationFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.informationFrame.setObjectName("informationFrame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.informationFrame)
        self.verticalLayout_3.setContentsMargins(20, -1, 8, 30)
        self.verticalLayout_3.setSpacing(40)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem2)
        self.lblFacebook = QtWidgets.QLabel(self.informationFrame)
        self.lblFacebook.setMinimumSize(QtCore.QSize(20, 20))
        self.lblFacebook.setMaximumSize(QtCore.QSize(20, 20))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.lblFacebook.setFont(font)
        self.lblFacebook.setText("")
        self.lblFacebook.setPixmap(QtGui.QPixmap(":/Images/Images/icons8-facebook-f-30.png"))
        self.lblFacebook.setScaledContents(True)
        self.lblFacebook.setAlignment(QtCore.Qt.AlignCenter)
        self.lblFacebook.setIndent(0)
        self.lblFacebook.setObjectName("lblFacebook")
        self.verticalLayout_3.addWidget(self.lblFacebook)
        self.lblYoutube = QtWidgets.QLabel(self.informationFrame)
        self.lblYoutube.setMinimumSize(QtCore.QSize(20, 20))
        self.lblYoutube.setMaximumSize(QtCore.QSize(20, 20))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.lblYoutube.setFont(font)
        self.lblYoutube.setText("")
        self.lblYoutube.setPixmap(QtGui.QPixmap(":/Images/Images/icons8-youtube-30.png"))
        self.lblYoutube.setScaledContents(True)
        self.lblYoutube.setAlignment(QtCore.Qt.AlignCenter)
        self.lblYoutube.setIndent(0)
        self.lblYoutube.setObjectName("lblYoutube")
        self.verticalLayout_3.addWidget(self.lblYoutube)
        self.lblWebsite = QtWidgets.QLabel(self.informationFrame)
        self.lblWebsite.setMinimumSize(QtCore.QSize(20, 20))
        self.lblWebsite.setMaximumSize(QtCore.QSize(20, 20))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.lblWebsite.setFont(font)
        self.lblWebsite.setText("")
        self.lblWebsite.setPixmap(QtGui.QPixmap(":/Images/Images/icons8-geography-30.png"))
        self.lblWebsite.setScaledContents(True)
        self.lblWebsite.setAlignment(QtCore.Qt.AlignCenter)
        self.lblWebsite.setIndent(0)
        self.lblWebsite.setObjectName("lblWebsite")
        self.verticalLayout_3.addWidget(self.lblWebsite)
        self.horizontalLayout_3.addWidget(self.informationFrame)
        self.stackedWidget = QtWidgets.QStackedWidget(self.containFrame)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        self.stackedWidget.setFont(font)
        self.stackedWidget.setStyleSheet("")
        self.stackedWidget.setObjectName("stackedWidget")
        self.stackCheckModel = QtWidgets.QWidget()
        self.stackCheckModel.setObjectName("stackCheckModel")
        self.containFrameCheckModel = QtWidgets.QFrame(self.stackCheckModel)
        self.containFrameCheckModel.setGeometry(QtCore.QRect(9, 0, 1100, 661))
        self.containFrameCheckModel.setMinimumSize(QtCore.QSize(1100, 0))
        self.containFrameCheckModel.setMaximumSize(QtCore.QSize(1100, 16777215))
        self.containFrameCheckModel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.containFrameCheckModel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.containFrameCheckModel.setObjectName("containFrameCheckModel")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.containFrameCheckModel)
        self.horizontalLayout_16.setContentsMargins(-1, 0, 40, 0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.processFrameCheckModel = QtWidgets.QFrame(self.containFrameCheckModel)
        self.processFrameCheckModel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.processFrameCheckModel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.processFrameCheckModel.setObjectName("processFrameCheckModel")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.processFrameCheckModel)
        self.verticalLayout_11.setContentsMargins(30, 30, -1, 0)
        self.verticalLayout_11.setSpacing(20)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.lblSelectFolderLabelCheckMissing_2 = QtWidgets.QLabel(self.processFrameCheckModel)
        self.lblSelectFolderLabelCheckMissing_2.setMinimumSize(QtCore.QSize(0, 30))
        self.lblSelectFolderLabelCheckMissing_2.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.lblSelectFolderLabelCheckMissing_2.setFont(font)
        self.lblSelectFolderLabelCheckMissing_2.setStyleSheet("color: rgb(255,255,255)")
        self.lblSelectFolderLabelCheckMissing_2.setObjectName("lblSelectFolderLabelCheckMissing_2")
        self.verticalLayout_11.addWidget(self.lblSelectFolderLabelCheckMissing_2)
        self.frameSelectFolderCheckMissing = QtWidgets.QFrame(self.processFrameCheckModel)
        self.frameSelectFolderCheckMissing.setMinimumSize(QtCore.QSize(0, 60))
        self.frameSelectFolderCheckMissing.setMaximumSize(QtCore.QSize(16777215, 90))
        self.frameSelectFolderCheckMissing.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frameSelectFolderCheckMissing.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameSelectFolderCheckMissing.setObjectName("frameSelectFolderCheckMissing")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.frameSelectFolderCheckMissing)
        self.horizontalLayout_18.setContentsMargins(0, 4, 0, 10)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.plainTextImageToInference = QtWidgets.QPlainTextEdit(self.frameSelectFolderCheckMissing)
        font = QtGui.QFont()
        font.setFamily("SF Pro Text")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.plainTextImageToInference.setFont(font)
        self.plainTextImageToInference.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.plainTextImageToInference.setStyleSheet("color: rgb(200,200,200)")
        self.plainTextImageToInference.setReadOnly(True)
        self.plainTextImageToInference.setObjectName("plainTextImageToInference")
        self.horizontalLayout_18.addWidget(self.plainTextImageToInference)
        self.btnSelectImageToInference = QtWidgets.QPushButton(self.frameSelectFolderCheckMissing)
        self.btnSelectImageToInference.setMinimumSize(QtCore.QSize(40, 40))
        self.btnSelectImageToInference.setMaximumSize(QtCore.QSize(40, 40))
        self.btnSelectImageToInference.setStyleSheet("")
        self.btnSelectImageToInference.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Images/Images/icons8-opened-folder-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnSelectImageToInference.setIcon(icon1)
        self.btnSelectImageToInference.setObjectName("btnSelectImageToInference")
        self.horizontalLayout_18.addWidget(self.btnSelectImageToInference)
        self.verticalLayout_11.addWidget(self.frameSelectFolderCheckMissing)
        self.lblSelectFolderLabelCheckMissing = QtWidgets.QLabel(self.processFrameCheckModel)
        self.lblSelectFolderLabelCheckMissing.setMinimumSize(QtCore.QSize(0, 30))
        self.lblSelectFolderLabelCheckMissing.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.lblSelectFolderLabelCheckMissing.setFont(font)
        self.lblSelectFolderLabelCheckMissing.setStyleSheet("color: rgb(255,255,255)")
        self.lblSelectFolderLabelCheckMissing.setObjectName("lblSelectFolderLabelCheckMissing")
        self.verticalLayout_11.addWidget(self.lblSelectFolderLabelCheckMissing)
        self.plainTextQuestion = QtWidgets.QTextEdit(self.processFrameCheckModel)
        font = QtGui.QFont()
        font.setFamily("SF Pro Text")
        font.setPointSize(15)
        self.plainTextQuestion.setFont(font)
        self.plainTextQuestion.setStyleSheet("color: rgb(255,255,255)")
        self.plainTextQuestion.setObjectName("plainTextQuestion")
        self.verticalLayout_11.addWidget(self.plainTextQuestion)
        self.btnStartProcessCheckModel = QtWidgets.QPushButton(self.processFrameCheckModel)
        self.btnStartProcessCheckModel.setMinimumSize(QtCore.QSize(0, 40))
        self.btnStartProcessCheckModel.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.btnStartProcessCheckModel.setFont(font)
        self.btnStartProcessCheckModel.setStyleSheet("color: rgb(255,255,255)")
        self.btnStartProcessCheckModel.setObjectName("btnStartProcessCheckModel")
        self.verticalLayout_11.addWidget(self.btnStartProcessCheckModel)
        self.horizontalLayout_16.addWidget(self.processFrameCheckModel)
        self.resultFrame = QtWidgets.QVBoxLayout()
        self.resultFrame.setContentsMargins(-1, 30, -1, -1)
        self.resultFrame.setSpacing(13)
        self.resultFrame.setObjectName("resultFrame")
        self.lblSelectFolderLabelCheckMissing_4 = QtWidgets.QLabel(self.containFrameCheckModel)
        self.lblSelectFolderLabelCheckMissing_4.setMinimumSize(QtCore.QSize(0, 30))
        self.lblSelectFolderLabelCheckMissing_4.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.lblSelectFolderLabelCheckMissing_4.setFont(font)
        self.lblSelectFolderLabelCheckMissing_4.setStyleSheet("color: rgb(255,255,255)")
        self.lblSelectFolderLabelCheckMissing_4.setObjectName("lblSelectFolderLabelCheckMissing_4")
        self.resultFrame.addWidget(self.lblSelectFolderLabelCheckMissing_4)
        self.lblFigureInference = QtWidgets.QLabel(self.containFrameCheckModel)
        self.lblFigureInference.setMinimumSize(QtCore.QSize(510, 300))
        self.lblFigureInference.setMaximumSize(QtCore.QSize(510, 300))
        self.lblFigureInference.setStyleSheet("border-radius: 13px;")
        self.lblFigureInference.setText("")
        self.lblFigureInference.setPixmap(QtGui.QPixmap(":/Images/Images/BG_Default.png"))
        self.lblFigureInference.setScaledContents(True)
        self.lblFigureInference.setObjectName("lblFigureInference")
        self.resultFrame.addWidget(self.lblFigureInference)
        self.lblSelectFolderLabelCheckMissing_3 = QtWidgets.QLabel(self.containFrameCheckModel)
        self.lblSelectFolderLabelCheckMissing_3.setMinimumSize(QtCore.QSize(0, 30))
        self.lblSelectFolderLabelCheckMissing_3.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.lblSelectFolderLabelCheckMissing_3.setFont(font)
        self.lblSelectFolderLabelCheckMissing_3.setStyleSheet("color: rgb(255,255,255)")
        self.lblSelectFolderLabelCheckMissing_3.setObjectName("lblSelectFolderLabelCheckMissing_3")
        self.resultFrame.addWidget(self.lblSelectFolderLabelCheckMissing_3)
        self.gridGroupBox = QtWidgets.QGroupBox(self.containFrameCheckModel)
        self.gridGroupBox.setMinimumSize(QtCore.QSize(510, 0))
        self.gridGroupBox.setMaximumSize(QtCore.QSize(510, 16777215))
        self.gridGroupBox.setStyleSheet("background-color: #323336;\n"
"border-radius: 13px;\n"
"")
        self.gridGroupBox.setObjectName("gridGroupBox")
        self.gridResult = QtWidgets.QGridLayout(self.gridGroupBox)
        self.gridResult.setObjectName("gridResult")
        self.lblAnswerTop2 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("SF Pro")
        font.setPointSize(16)
        self.lblAnswerTop2.setFont(font)
        self.lblAnswerTop2.setStyleSheet("color: rgb(255,255,255)")
        self.lblAnswerTop2.setObjectName("lblAnswerTop2")
        self.gridResult.addWidget(self.lblAnswerTop2, 1, 1, 1, 1)
        self.lblTop2 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lblTop2.setFont(font)
        self.lblTop2.setStyleSheet("color: rgb(215,215,215)")
        self.lblTop2.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTop2.setObjectName("lblTop2")
        self.gridResult.addWidget(self.lblTop2, 1, 0, 1, 1)
        self.lblTop1 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lblTop1.setFont(font)
        self.lblTop1.setStyleSheet("color: rgb(215,215,215);\n"
"border-bottom: solid;")
        self.lblTop1.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTop1.setObjectName("lblTop1")
        self.gridResult.addWidget(self.lblTop1, 0, 0, 1, 1)
        self.lblAnswerTop1 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("SF Pro")
        font.setPointSize(16)
        self.lblAnswerTop1.setFont(font)
        self.lblAnswerTop1.setStyleSheet("color: rgb(255,255,255);\n"
"border-bottom: solid;")
        self.lblAnswerTop1.setObjectName("lblAnswerTop1")
        self.gridResult.addWidget(self.lblAnswerTop1, 0, 1, 1, 1)
        self.lblTop3 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lblTop3.setFont(font)
        self.lblTop3.setStyleSheet("color: rgb(215,215,215)")
        self.lblTop3.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTop3.setObjectName("lblTop3")
        self.gridResult.addWidget(self.lblTop3, 2, 0, 1, 1)
        self.lblTop4 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lblTop4.setFont(font)
        self.lblTop4.setStyleSheet("color: rgb(215,215,215)")
        self.lblTop4.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTop4.setObjectName("lblTop4")
        self.gridResult.addWidget(self.lblTop4, 3, 0, 1, 1)
        self.lblAnswerTop3 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("SF Pro")
        font.setPointSize(16)
        self.lblAnswerTop3.setFont(font)
        self.lblAnswerTop3.setStyleSheet("color: rgb(255,255,255)")
        self.lblAnswerTop3.setObjectName("lblAnswerTop3")
        self.gridResult.addWidget(self.lblAnswerTop3, 2, 1, 1, 1)
        self.lblAnswerTop5 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("SF Pro")
        font.setPointSize(16)
        self.lblAnswerTop5.setFont(font)
        self.lblAnswerTop5.setStyleSheet("color: rgb(255,255,255)")
        self.lblAnswerTop5.setObjectName("lblAnswerTop5")
        self.gridResult.addWidget(self.lblAnswerTop5, 4, 1, 1, 1)
        self.lblAnswerTop4 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("SF Pro")
        font.setPointSize(16)
        self.lblAnswerTop4.setFont(font)
        self.lblAnswerTop4.setStyleSheet("color: rgb(255,255,255)")
        self.lblAnswerTop4.setObjectName("lblAnswerTop4")
        self.gridResult.addWidget(self.lblAnswerTop4, 3, 1, 1, 1)
        self.lblTop5 = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lblTop5.setFont(font)
        self.lblTop5.setStyleSheet("color: rgb(215,215,215)")
        self.lblTop5.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTop5.setObjectName("lblTop5")
        self.gridResult.addWidget(self.lblTop5, 4, 0, 1, 1)
        self.gridResult.setColumnStretch(0, 1)
        self.gridResult.setColumnStretch(1, 6)
        self.resultFrame.addWidget(self.gridGroupBox)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(40, 15, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.btnSceenshot = QtWidgets.QPushButton(self.containFrameCheckModel)
        self.btnSceenshot.setMinimumSize(QtCore.QSize(150, 40))
        self.btnSceenshot.setMaximumSize(QtCore.QSize(1000, 40))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnSceenshot.setFont(font)
        self.btnSceenshot.setStyleSheet("color: rgb(255,255,255)")
        self.btnSceenshot.setIconSize(QtCore.QSize(30, 16))
        self.btnSceenshot.setObjectName("btnSceenshot")
        self.horizontalLayout_2.addWidget(self.btnSceenshot)
        self.btnFlag = QtWidgets.QPushButton(self.containFrameCheckModel)
        self.btnFlag.setMinimumSize(QtCore.QSize(80, 40))
        self.btnFlag.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Montserrat Thin")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnFlag.setFont(font)
        self.btnFlag.setStyleSheet("QPushButton {\n"
"    background-color:#910101;\n"
"    color: rgb(255,255,255)\n"
"    /*color: rgb(255,255,255);*/\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color: rgb(200,200,200);\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color:#870101;\n"
"    /* color: rgb(255,255,255); */\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color:#9e0202;\n"
"}")
        self.btnFlag.setIconSize(QtCore.QSize(30, 16))
        self.btnFlag.setObjectName("btnFlag")
        self.horizontalLayout_2.addWidget(self.btnFlag)
        self.resultFrame.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_16.addLayout(self.resultFrame)
        self.stackedWidget.addWidget(self.stackCheckModel)
        self.horizontalLayout_3.addWidget(self.stackedWidget)
        self.verticalLayout_2.addWidget(self.containFrame)
        self.verticalLayout.addWidget(self.mainBackground)
        MainWindow.setCentralWidget(self.styleSheet)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lblSubtle.setText(_translate("MainWindow", "DEMO VQA-MODEL - GROUP 4"))
        self.lblSelectFolderLabelCheckMissing_2.setText(_translate("MainWindow", "Select Image"))
        self.plainTextImageToInference.setPlainText(_translate("MainWindow", "Select your image path"))
        self.plainTextImageToInference.setPlaceholderText(_translate("MainWindow", "Select your image path"))
        self.lblSelectFolderLabelCheckMissing.setText(_translate("MainWindow", "Question"))
        self.plainTextQuestion.setPlaceholderText(_translate("MainWindow", "Insert your question"))
        self.btnStartProcessCheckModel.setText(_translate("MainWindow", "Start Inference"))
        self.lblSelectFolderLabelCheckMissing_4.setText(_translate("MainWindow", "Image"))
        self.lblSelectFolderLabelCheckMissing_3.setText(_translate("MainWindow", "Answer"))
        self.lblAnswerTop2.setText(_translate("MainWindow", "The second answer reponse from model"))
        self.lblTop2.setText(_translate("MainWindow", "2"))
        self.lblTop1.setText(_translate("MainWindow", "1"))
        self.lblAnswerTop1.setText(_translate("MainWindow", "The first answer reponse from model"))
        self.lblTop3.setText(_translate("MainWindow", "3"))
        self.lblTop4.setText(_translate("MainWindow", "4"))
        self.lblAnswerTop3.setText(_translate("MainWindow", "The third answer reponse from model"))
        self.lblAnswerTop5.setText(_translate("MainWindow", "The fifth answer reponse from model"))
        self.lblAnswerTop4.setText(_translate("MainWindow", "The fourth answer reponse from model"))
        self.lblTop5.setText(_translate("MainWindow", "5"))
        self.btnSceenshot.setText(_translate("MainWindow", "Screenshot"))
        self.btnFlag.setText(_translate("MainWindow", "Flags"))
import Images_rc
