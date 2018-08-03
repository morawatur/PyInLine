import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

class ImagePropertiesWin(QtWidgets.QWidget):
    def __init__(self):
        super(ImagePropertiesWin, self).__init__()
        self.initUI()

    def initUI(self):
        self.name_input = QtWidgets.QLineEdit('img01', self)
        self.num_in_ser_input = QtWidgets.QLineEdit('ble', self)
        self.defocus_input = QtWidgets.QLineEdit('0.0', self)
        self.width_input = QtWidgets.QLineEdit('ble', self)
        self.height_input = QtWidgets.QLineEdit('ble', self)
        self.px_size_input = QtWidgets.QLineEdit('ble', self)
        set_name_button = QtWidgets.QPushButton('Set', self)
        set_num_in_ser_button = QtWidgets.QPushButton('Set', self)
        # self.setLayout(vbox_main)
        self.move(250, 5)
        self.setWindowTitle('Image properties')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing
