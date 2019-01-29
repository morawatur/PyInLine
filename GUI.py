import re
import sys
from os import path
from functools import partial
import numpy as np

from PyQt5 import QtGui, QtCore, QtWidgets
import Dm3Reader3 as dm3
import Constants as const
import CudaConfig as ccfg
import ImageSupport as imsup
import CrossCorr as cc
import Transform as tr
import Holo as holo
import Propagation as prop

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# --------------------------------------------------------

def pass_fun():
    pass

# --------------------------------------------------------

class RgbColorTable:
    def __init__(self):
        step = 6
        inc_range = np.arange(0, 256, step)
        dec_range = np.arange(255, -1, -step)
        bcm1 = [QtGui.qRgb(0, i, 255) for i in inc_range]
        gcm1 = [QtGui.qRgb(0, 255, i) for i in dec_range]
        gcm2 = [QtGui.qRgb(i, 255, 0) for i in inc_range]
        rcm1 = [QtGui.qRgb(255, i, 0) for i in dec_range]
        rcm2 = [QtGui.qRgb(255, 0, i) for i in inc_range]
        bcm2 = [QtGui.qRgb(i, 0, 255) for i in dec_range]
        self.cm = bcm1 + gcm1 + gcm2 + rcm1 + rcm2 + bcm2

# --------------------------------------------------------

class MarkButton(QtWidgets.QPushButton):
    def __init__(self, text, width, height, fun):
        super(MarkButton, self).__init__(text)
        self.default_style = 'color:transparent; width:{0}; height:{1}; padding:-1px;'.format(width, height)
        self.clicked_style = 'background-color:rgb(140, 140, 140); color:transparent; width:{0}; height:{1}; padding:-1px;'.format(width, height)
        self.was_clicked = False
        self.do_something = fun
        self.initUI()

    def initUI(self):
        self.setStyleSheet(self.default_style)
        self.clicked.connect(self.handle_button)

    def handle_button(self):
        self.was_clicked = not self.was_clicked
        if self.was_clicked:
            self.setStyleSheet(self.clicked_style)
        else:
            self.setStyleSheet(self.default_style)
        self.do_something()

# --------------------------------------------------------

class ButtonGrid(QtWidgets.QGridLayout):
    def __init__(self, n_per_row=1, n_per_col=-1, fun=pass_fun):
        super(ButtonGrid, self).__init__()
        self.n_rows = n_per_row
        self.n_cols = n_per_row if n_per_col < 0 else n_per_col
        self.n_rows_min = 1
        self.n_rows_max = 5
        self.grid_dim_sz = 120
        self.btn_fun = fun
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)
        self.create_grid()

    def create_grid(self):
        if self.count() > 0:
            n_rows = int(np.sqrt(self.count()))
            n_cols = n_rows
            old_positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]
            for pos in old_positions:
                button = self.itemAtPosition(pos[0], pos[1]).widget()
                button.deleteLater()

        positions = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols)]
        btn_width = np.ceil(self.grid_dim_sz / self.n_rows)

        for pos in positions:
            button = MarkButton('{0}'.format(pos), btn_width, btn_width, self.btn_fun)
            self.addWidget(button, *pos)

    def increase_mesh(self):
        if self.n_rows + 1 <= self.n_rows_max:
            self.n_rows += 1
            self.n_cols += 1
            self.create_grid()
            # self.parent().display.show_grid = True

    def decrease_mesh(self):
        if self.n_rows - 1 >= self.n_rows_min:
            self.n_rows -= 1
            self.n_cols -= 1
            self.create_grid()

# --------------------------------------------------------

class LabelExt(QtWidgets.QLabel):
    def __init__(self, parent, image=None):
        super(LabelExt, self).__init__(parent)
        self.image = image
        self.setImage()
        self.n_imgs = len(imsup.CreateImageListFromFirstImage(self.image))
        self.pointSets = []
        for i in range(self.n_imgs):
            self.pointSets.append([])
        # self.pointSets = [[]] * n_imgs
        self.frag_coords = []
        self.show_lines = True
        self.show_labs = True
        self.show_grid = True
        self.show_aper = False
        self.show_hann = False
        self.aper_diam = 0
        self.hann_width = 0
        self.gain = 0.0
        self.bias = 0.0
        self.gamma = 0.0
        self.rgb_cm = RgbColorTable()

    # prowizorka - stałe liczbowe do poprawy
    def paintEvent(self, event):
        super(LabelExt, self).paintEvent(event)
        linePen = QtGui.QPen(QtCore.Qt.yellow)
        linePen.setCapStyle(QtCore.Qt.RoundCap)
        linePen.setWidth(3)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        imgIdx = self.image.numInSeries - 1
        qp.setPen(linePen)
        qp.setBrush(QtCore.Qt.yellow)

        for pt in self.pointSets[imgIdx]:
            # rect = QtCore.QRect(pt[0]-3, pt[1]-3, 7, 7)
            # qp.drawArc(rect, 0, 16*360)
            qp.drawEllipse(pt[0]-3, pt[1]-3, 7, 7)

        linePen.setWidth(2)
        if self.show_lines:
            qp.setPen(linePen)
            for pt1, pt2 in zip(self.pointSets[imgIdx], self.pointSets[imgIdx][1:] + self.pointSets[imgIdx][:1]):
                line = QtCore.QLine(pt1[0], pt1[1], pt2[0], pt2[1])
                qp.drawLine(line)

        linePen.setStyle(QtCore.Qt.DashLine)
        linePen.setColor(QtCore.Qt.yellow)
        linePen.setCapStyle(QtCore.Qt.FlatCap)
        qp.setPen(linePen)
        qp.setBrush(QtCore.Qt.NoBrush)
        if len(self.pointSets[imgIdx]) == 2:
            pt1, pt2 = self.pointSets[imgIdx]
            pt1, pt2 = convert_points_to_tl_br(pt1, pt2)
            w = np.abs(pt2[0] - pt1[0])
            h = np.abs(pt2[1] - pt1[1])
            rect = QtCore.QRect(pt1[0], pt1[1], w, h)
            qp.drawRect(rect)
            sq_coords = imsup.MakeSquareCoords(pt1 + pt2)
            sq_pt1 = sq_coords[:2][::-1]
            sq_pt2 = sq_coords[2:][::-1]
            w = np.abs(sq_pt2[0]-sq_pt1[0])
            h = np.abs(sq_pt2[1]-sq_pt1[1])
            square = QtCore.QRect(sq_pt1[0], sq_pt1[1], w, h)
            linePen.setColor(QtCore.Qt.red)
            qp.setPen(linePen)
            qp.drawRect(square)

        if self.show_aper:
            linePen.setStyle(QtCore.Qt.SolidLine)
            linePen.setColor(QtCore.Qt.red)
            qp.setPen(linePen)
            tl_x = (const.ccWidgetDim - self.aper_diam) // 2
            qp.drawEllipse(tl_x, tl_x, self.aper_diam, self.aper_diam)

        if self.show_hann:
            linePen.setStyle(QtCore.Qt.SolidLine)
            linePen.setColor(QtCore.Qt.blue)
            qp.setPen(linePen)
            tl_x = (const.ccWidgetDim - self.hann_width) // 2
            qp.drawRect(tl_x, tl_x, self.hann_width, self.hann_width)

        if len(self.frag_coords) > 0:
            qp.setPen(QtCore.Qt.NoPen)
            qp.setBrush(QtGui.QColor(240, 240, 240, 100))
            grid_dim = self.frag_coords[0]
            sq_w = int(np.ceil(const.ccWidgetDim / grid_dim))
            for pos in self.frag_coords[1:]:
                sq_tl = [ x * sq_w for x in pos ]
                sq = QtCore.QRect(sq_tl[0], sq_tl[1], sq_w, sq_w)
                qp.drawRect(sq)
        qp.end()

    def mouseReleaseEvent(self, QMouseEvent):
        pos = QMouseEvent.pos()
        currPos = [pos.x(), pos.y()]
        self.pointSets[self.image.numInSeries - 1].append(currPos)
        self.repaint()

        if self.parent().show_labels_checkbox.isChecked():
            lab = QtWidgets.QLabel('{0}'.format(len(self.pointSets[self.image.numInSeries - 1])), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pos.x()+4, pos.y()+4)
            lab.show()

    def setImage(self, dispAmp=True, dispPhs=False, logScale=False, color=False, update_bcg=False, bright=0, cont=255, gamma=1.0):
        self.image.MoveToCPU()

        # if image wasn't cropped then update buffer
        if self.image.buffer.am.shape[0] == self.image.height:
            self.image.UpdateBuffer()

        if dispAmp:
            px_arr = np.copy(self.image.buffer.am)
            if logScale:
                buf_am = np.copy(px_arr)
                buf_am[np.where(buf_am <= 0)] = 1e-5
                px_arr = np.log(buf_am)
        else:
            px_arr = np.copy(self.image.buffer.ph)
            if not dispPhs:
                self.image.update_cos_phase()
                px_arr = np.cos(px_arr)

        if not update_bcg:
            pixmap_to_disp = imsup.ScaleImage(px_arr, 0.0, 255.0)
        else:
            pixmap_to_disp = update_image_bright_cont_gamma(px_arr, brg=bright, cnt=cont, gam=gamma)

        # final image with all properties set
        q_image = QtGui.QImage(pixmap_to_disp.astype(np.uint8), pixmap_to_disp.shape[0], pixmap_to_disp.shape[1], QtGui.QImage.Format_Indexed8)

        if color:
            q_image.setColorTable(self.rgb_cm.cm)

        pixmap = QtGui.QPixmap(q_image)
        pixmap = pixmap.scaledToWidth(const.ccWidgetDim)
        self.setPixmap(pixmap)
        self.repaint()

    def update_labs(self, dispLabs=True):
        if len(self.pointSets) < self.image.numInSeries:
            self.pointSets.append([])

        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

        if dispLabs:
            self.show_labels()

    def change_image(self, new_idx, dispAmp=True, dispPhs=False, logScale=False, dispLabs=True, color=False):
        curr = self.image
        first = imsup.GetFirstImage(curr)
        imgs = imsup.CreateImageListFromFirstImage(first)
        if 0 > new_idx > len(imgs) - 1:
            return

        new_img = imgs[new_idx]
        new_img.ReIm2AmPh()
        self.image = new_img
        self.setImage(dispAmp, dispPhs, logScale, color)
        self.update_labs(dispLabs)

    def change_image_adjacent(self, dir_to_img=1, dispAmp=True, dispPhs=False, logScale=False, dispLabs=True, color=False):
        if dir_to_img == 1:
            new_img = self.image.next
        else:
            new_img = self.image.prev

        if new_img is None:
            return

        new_img.ReIm2AmPh()
        self.image = new_img
        self.setImage(dispAmp, dispPhs, logScale, color)
        self.update_labs(dispLabs)

    def hide_labels(self):
        labsToDel = self.children()
        for child in labsToDel:
            child.deleteLater()

    def show_labels(self):
        imgIdx = self.image.numInSeries - 1
        for pt, idx in zip(self.pointSets[imgIdx], range(1, len(self.pointSets[imgIdx]) + 1)):
            lab = QtWidgets.QLabel('{0}'.format(idx), self)
            lab.setStyleSheet('font-size:14pt; background-color:white; border:1px solid rgb(0, 0, 0);')
            lab.move(pt[0] + 4, pt[1] + 4)
            lab.show()

# --------------------------------------------------------

def update_image_bright_cont_gamma(img_src, brg=0, cnt=1, gam=1.0):
    Imin, Imax = det_Imin_Imax_from_contrast(cnt)

    # option 1 (c->b->g)
    # correct contrast
    img_scaled = imsup.ScaleImage(img_src, Imin, Imax)
    # correct brightness
    img_scaled += brg
    img_scaled[img_scaled < 0.0] = 0.0
    # correct gamma
    img_scaled **= gam
    img_scaled[img_scaled > 255.0] = 255.0

    # # option 2 (c->g->b)
    # # correct contrast
    # img_scaled = imsup.ScaleImage(img_src, Imin, Imax)
    # img_scaled[img_scaled < 0.0] = 0.0
    # # correct gamma
    # img_scaled **= gam
    # # correct brightness
    # img_scaled += brg
    # img_scaled[img_scaled < 0.0] = 0.0
    # img_scaled[img_scaled > 255.0] = 255.0
    return img_scaled

# --------------------------------------------------------

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.markedPoints = []
        self.markedPointsData = []
        self.canvas.mpl_connect('button_press_event', self.getXYDataOnClick)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, dataX, dataY, xlab='x', ylab='y'):
        self.figure.clear()
        self.markedPoints = []
        self.markedPointsData = []
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axis([ min(dataX)-0.5, max(dataX)+0.5, min(dataY)-0.5, max(dataY)+0.5 ])
        ax = self.figure.add_subplot(111)
        ax.plot(dataX, dataY, '.-')
        self.canvas.draw()

    def getXYDataOnClick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if len(self.markedPoints) == 2:
            for pt in self.markedPoints:
                pt.remove()
            self.markedPoints = []
            self.markedPointsData = []
        pt, = plt.plot(event.xdata, event.ydata, 'ro')
        print(event.xdata, event.ydata)
        self.markedPoints.append(pt)
        self.markedPointsData.append([event.xdata, event.ydata])

# --------------------------------------------------------

class InLineWidget(QtWidgets.QWidget):
    def __init__(self):
        super(InLineWidget, self).__init__()
        file_dialog = QtWidgets.QFileDialog()
        image_path = file_dialog.getOpenFileName()[0]
        if image_path == '':
            print('No images to read. Exiting...')
            exit()
        image = LoadImageSeriesFromFirstFile(image_path)
        self.display = LabelExt(self, image)
        self.display.setFixedWidth(const.ccWidgetDim)
        self.display.setFixedHeight(const.ccWidgetDim)
        self.plot_widget = PlotWidget()
        self.backup_image = None
        self.changes_made = []
        self.shift = [0, 0]
        self.rot_angle = 0
        self.mag_coeff = 1.0
        self.warp_points = []
        self.curr_iter = 0
        self.curr_exit_wave = None      # !!!
        self.curr_ewr_imgs = None       # !!!
        self.curr_ddfs = [0]            # !!!
        self.last_tot_error = 0.0       # !!!
        self.initUI()

    def initUI(self):
        self.plot_widget.canvas.setFixedHeight(350)

        self.curr_info_label = QtWidgets.QLabel('', self)
        self.update_curr_info_label()

        # ------------------------------
        # Navigation panel (1)
        # ------------------------------

        self.clear_prev_checkbox = QtWidgets.QCheckBox('Clear prev. images', self)
        self.clear_prev_checkbox.setChecked(True)

        prev_button = QtWidgets.QPushButton('Prev', self)
        next_button = QtWidgets.QPushButton('Next', self)
        lswap_button = QtWidgets.QPushButton('L-Swap', self)
        rswap_button = QtWidgets.QPushButton('R-Swap', self)
        flip_button = QtWidgets.QPushButton('Flip', self)
        set_name_button = QtWidgets.QPushButton('Set name', self)
        reset_names_button = QtWidgets.QPushButton('Reset names', self)
        zoom_button = QtWidgets.QPushButton('Crop N ROIs', self)
        delete_button = QtWidgets.QPushButton('Delete', self)
        clear_button = QtWidgets.QPushButton('Clear', self)
        undo_button = QtWidgets.QPushButton('Undo', self)

        prev_button.clicked.connect(self.go_to_prev_image)
        next_button.clicked.connect(self.go_to_next_image)
        lswap_button.clicked.connect(self.swap_left)
        rswap_button.clicked.connect(self.swap_right)
        flip_button.clicked.connect(self.flip_image_h)
        set_name_button.clicked.connect(self.set_image_name)
        reset_names_button.clicked.connect(self.reset_image_names)
        zoom_button.clicked.connect(self.zoom_n_fragments)
        delete_button.clicked.connect(self.delete_image)
        clear_button.clicked.connect(self.clear_image)
        undo_button.clicked.connect(self.remove_last_point)

        self.name_input = QtWidgets.QLineEdit(self.display.image.name, self)
        self.n_to_zoom_input = QtWidgets.QLineEdit('1', self)

        hbox_name = QtWidgets.QHBoxLayout()
        hbox_name.addWidget(set_name_button)
        hbox_name.addWidget(self.name_input)

        hbox_zoom = QtWidgets.QHBoxLayout()
        hbox_zoom.addWidget(zoom_button)
        hbox_zoom.addWidget(self.n_to_zoom_input)

        self.tab_nav = QtWidgets.QWidget()
        self.tab_nav.layout = QtWidgets.QGridLayout()
        self.tab_nav.layout.setColumnStretch(0, 1)
        self.tab_nav.layout.setColumnStretch(1, 1)
        self.tab_nav.layout.setColumnStretch(2, 1)
        self.tab_nav.layout.setColumnStretch(3, 1)
        self.tab_nav.layout.setColumnStretch(4, 1)
        self.tab_nav.layout.setColumnStretch(5, 1)
        self.tab_nav.layout.setRowStretch(0, 1)
        self.tab_nav.layout.setRowStretch(7, 1)
        self.tab_nav.layout.addWidget(prev_button, 1, 1, 1, 2)
        self.tab_nav.layout.addWidget(next_button, 1, 3, 1, 2)
        self.tab_nav.layout.addWidget(lswap_button, 2, 1, 1, 2)
        self.tab_nav.layout.addWidget(rswap_button, 2, 3, 1, 2)
        self.tab_nav.layout.addWidget(flip_button, 3, 1, 1, 2)
        self.tab_nav.layout.addWidget(clear_button, 3, 3, 1, 2)
        self.tab_nav.layout.addWidget(zoom_button, 4, 1)
        self.tab_nav.layout.addWidget(self.n_to_zoom_input, 4, 2)
        self.tab_nav.layout.addWidget(delete_button, 4, 3, 1, 2)
        self.tab_nav.layout.addWidget(set_name_button, 5, 1)
        self.tab_nav.layout.addWidget(self.name_input, 5, 2)
        self.tab_nav.layout.addWidget(undo_button, 5, 3, 1, 2)
        self.tab_nav.layout.addWidget(reset_names_button, 6, 1, 1, 2)
        self.tab_nav.layout.addWidget(self.clear_prev_checkbox, 6, 3, 1, 2)
        self.tab_nav.setLayout(self.tab_nav.layout)

        # ------------------------------
        # Display panel (2)
        # ------------------------------

        self.show_lines_checkbox = QtWidgets.QCheckBox('Show lines', self)
        self.show_lines_checkbox.setChecked(True)
        self.show_lines_checkbox.toggled.connect(self.toggle_lines)

        self.show_labels_checkbox = QtWidgets.QCheckBox('Show labels', self)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.toggle_labels)

        self.log_scale_checkbox = QtWidgets.QCheckBox('Log scale', self)
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.toggled.connect(self.update_display)

        self.amp_radio_button = QtWidgets.QRadioButton('Amplitude', self)
        self.phs_radio_button = QtWidgets.QRadioButton('Phase', self)
        self.cos_phs_radio_button = QtWidgets.QRadioButton('Phase cosine', self)
        self.amp_radio_button.setChecked(True)

        amp_phs_group = QtWidgets.QButtonGroup(self)
        amp_phs_group.addButton(self.amp_radio_button)
        amp_phs_group.addButton(self.phs_radio_button)
        amp_phs_group.addButton(self.cos_phs_radio_button)

        self.gray_radio_button = QtWidgets.QRadioButton('Grayscale', self)
        self.color_radio_button = QtWidgets.QRadioButton('Color', self)
        self.gray_radio_button.setChecked(True)

        color_group = QtWidgets.QButtonGroup(self)
        color_group.addButton(self.gray_radio_button)
        color_group.addButton(self.color_radio_button)

        fname_label = QtWidgets.QLabel('File name', self)
        self.fname_input = QtWidgets.QLineEdit(self.display.image.name, self)

        unwrap_button = QtWidgets.QPushButton('Unwrap', self)
        wrap_button = QtWidgets.QPushButton('Wrap', self)
        export_button = QtWidgets.QPushButton('Export', self)
        export_all_button = QtWidgets.QPushButton('Export all', self)
        norm_phase_button = QtWidgets.QPushButton('Normalize phase', self)

        self.export_tiff_radio_button = QtWidgets.QRadioButton('TIFF image', self)
        self.export_bin_radio_button = QtWidgets.QRadioButton('Binary', self)
        self.export_tiff_radio_button.setChecked(True)

        export_group = QtWidgets.QButtonGroup(self)
        export_group.addButton(self.export_tiff_radio_button)
        export_group.addButton(self.export_bin_radio_button)

        self.amp_radio_button.toggled.connect(self.update_display)
        self.phs_radio_button.toggled.connect(self.update_display)
        self.cos_phs_radio_button.toggled.connect(self.update_display)
        self.gray_radio_button.toggled.connect(self.update_display)
        self.color_radio_button.toggled.connect(self.update_display)
        unwrap_button.clicked.connect(self.unwrap_img_phase)
        wrap_button.clicked.connect(self.wrap_img_phase)
        export_button.clicked.connect(self.export_image)
        export_all_button.clicked.connect(self.export_all)
        norm_phase_button.clicked.connect(self.norm_phase)

        grid_disp = QtWidgets.QGridLayout()
        grid_disp.setColumnStretch(0, 1)
        grid_disp.setColumnStretch(5, 1)
        grid_disp.setRowStretch(0, 1)
        # grid_disp.setRowStretch(4, 1)
        grid_disp.addWidget(self.show_lines_checkbox, 1, 1)
        grid_disp.addWidget(self.show_labels_checkbox, 2, 1)
        grid_disp.addWidget(self.log_scale_checkbox, 3, 1)
        grid_disp.addWidget(self.amp_radio_button, 1, 2)
        grid_disp.addWidget(self.phs_radio_button, 2, 2)
        grid_disp.addWidget(self.cos_phs_radio_button, 3, 2)
        grid_disp.addWidget(unwrap_button, 1, 4)
        grid_disp.addWidget(wrap_button, 2, 4)
        grid_disp.addWidget(self.gray_radio_button, 1, 3)
        grid_disp.addWidget(self.color_radio_button, 2, 3)
        grid_disp.addWidget(norm_phase_button, 3, 3, 1, 2)

        grid_exp = QtWidgets.QGridLayout()
        grid_exp.setColumnStretch(0, 1)
        grid_exp.setColumnStretch(4, 1)
        grid_exp.setRowStretch(0, 1)
        grid_exp.setRowStretch(3, 1)
        grid_exp.addWidget(fname_label, 1, 1)
        grid_exp.addWidget(self.fname_input, 2, 1)
        grid_exp.addWidget(export_button, 1, 2)
        grid_exp.addWidget(export_all_button, 2, 2)
        grid_exp.addWidget(self.export_tiff_radio_button, 1, 3)
        grid_exp.addWidget(self.export_bin_radio_button, 2, 3)

        self.tab_disp = QtWidgets.QWidget()
        self.tab_disp.layout = QtWidgets.QVBoxLayout()
        self.tab_disp.layout.addLayout(grid_disp)
        self.tab_disp.layout.addLayout(grid_exp)
        self.tab_disp.setLayout(self.tab_disp.layout)

        # ------------------------------
        # Manual alignment panel (3)
        # ------------------------------

        self.left_button = QtWidgets.QPushButton(QtGui.QIcon('gui/left.png'), '', self)
        self.right_button = QtWidgets.QPushButton(QtGui.QIcon('gui/right.png'), '', self)
        self.up_button = QtWidgets.QPushButton(QtGui.QIcon('gui/up.png'), '', self)
        self.down_button = QtWidgets.QPushButton(QtGui.QIcon('gui/down.png'), '', self)
        self.rot_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_right.png'), '', self)
        self.rot_counter_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_left.png'), '', self)
        self.apply_button = QtWidgets.QPushButton('Apply changes', self)
        self.reset_button = QtWidgets.QPushButton('Reset', self)

        self.px_shift_input = QtWidgets.QLineEdit('0', self)
        self.rot_angle_input = QtWidgets.QLineEdit('0.0', self)

        self.manual_mode_checkbox = QtWidgets.QCheckBox('Manual mode', self)
        self.manual_mode_checkbox.setChecked(False)
        self.manual_mode_checkbox.clicked.connect(self.create_backup_image)

        self.left_button.clicked.connect(self.move_left)
        self.right_button.clicked.connect(self.move_right)
        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)
        self.rot_clockwise_button.clicked.connect(self.rot_right)
        self.rot_counter_clockwise_button.clicked.connect(self.rot_left)
        self.apply_button.clicked.connect(self.apply_changes)
        self.reset_button.clicked.connect(self.reset_changes)

        self.disable_manual_panel()

        # ------------------------------
        # Automatic alignment panel (4)
        # ------------------------------

        self.btn_grid = ButtonGrid(1, fun=self.get_clicked_coords)
        mesh_up_button = QtWidgets.QPushButton('Mesh+', self)
        mesh_down_button = QtWidgets.QPushButton('Mesh-', self)
        cross_corr_w_prev_button = QtWidgets.QPushButton('Cross-corr. with prev.', self)
        cross_corr_n_images_button = QtWidgets.QPushButton('Cross-corr. N images', self)
        shift_button = QtWidgets.QPushButton('Shift', self)
        warp_button = QtWidgets.QPushButton('Warp', self)
        set_df_button = QtWidgets.QPushButton('Set df', self)

        self.df_min_label = QtWidgets.QLabel('df min [nm]', self)
        df_max_label = QtWidgets.QLabel('df max [nm]', self)
        df_step_label = QtWidgets.QLabel('delta df [nm]', self)

        self.n_to_cc_input = QtWidgets.QLineEdit(str(self.display.n_imgs), self)
        self.df_min_input = QtWidgets.QLineEdit('0.0', self)
        self.df_max_input = QtWidgets.QLineEdit('0.0', self)
        self.df_step_input = QtWidgets.QLineEdit('0.0', self)

        self.det_df_checkbox = QtWidgets.QCheckBox('Determine df values', self)
        self.det_df_checkbox.setChecked(False)
        self.det_df_checkbox.toggled.connect(self.manage_df_inputs)

        mesh_up_button.clicked.connect(self.btn_grid.increase_mesh)
        mesh_down_button.clicked.connect(self.btn_grid.decrease_mesh)
        cross_corr_w_prev_button.clicked.connect(self.cross_corr_with_prev)
        cross_corr_n_images_button.clicked.connect(self.cross_corr_n_images)
        shift_button.clicked.connect(self.align_shift)
        warp_button.clicked.connect(partial(self.warp_image, False))
        set_df_button.clicked.connect(self.set_defocus)

        self.df_min_label.setAlignment(QtCore.Qt.AlignCenter)
        df_max_label.setAlignment(QtCore.Qt.AlignCenter)
        df_step_label.setAlignment(QtCore.Qt.AlignCenter)

        cross_corr_n_images_button.setFixedWidth(130)
        self.disable_df_inputs()

        grid_manual = QtWidgets.QGridLayout()
        grid_manual.setColumnStretch(0, 1)
        grid_manual.setColumnStretch(1, 2)
        grid_manual.setColumnStretch(2, 2)
        grid_manual.setColumnStretch(3, 2)
        grid_manual.setColumnStretch(4, 2)
        grid_manual.setColumnStretch(5, 2)
        grid_manual.setColumnStretch(6, 2)
        grid_manual.setColumnStretch(7, 2)
        grid_manual.setColumnStretch(8, 1)
        grid_manual.setRowStretch(0, 1)
        grid_manual.setRowStretch(4, 1)
        grid_manual.addWidget(self.left_button, 2, 1)
        grid_manual.addWidget(self.right_button, 2, 3)
        grid_manual.addWidget(self.up_button, 1, 2)
        grid_manual.addWidget(self.down_button, 3, 2)
        grid_manual.addWidget(self.px_shift_input, 2, 2)
        grid_manual.addWidget(self.manual_mode_checkbox, 1, 4)
        grid_manual.addWidget(self.apply_button, 2, 4)
        grid_manual.addWidget(self.reset_button, 3, 4)
        grid_manual.addWidget(self.rot_counter_clockwise_button, 2, 5)
        grid_manual.addWidget(self.rot_angle_input, 2, 6)
        grid_manual.addWidget(self.rot_clockwise_button, 2, 7)

        grid_auto = QtWidgets.QGridLayout()
        grid_auto.setColumnStretch(0, 1)
        grid_auto.setColumnStretch(1, 1)
        grid_auto.setColumnStretch(2, 1)
        grid_auto.setColumnStretch(3, 1)
        grid_auto.setColumnStretch(4, 1)
        grid_auto.setColumnStretch(5, 1)
        grid_auto.setColumnStretch(6, 1)
        grid_auto.setColumnStretch(7, 1)
        grid_auto.setRowStretch(0, 1)
        grid_auto.setRowStretch(6, 1)
        grid_auto.addLayout(self.btn_grid, 1, 1, 4, 1)
        grid_auto.addWidget(mesh_up_button, 1, 2)
        grid_auto.addWidget(mesh_down_button, 2, 2)
        grid_auto.addWidget(cross_corr_w_prev_button, 1, 3, 1, 2)
        grid_auto.addWidget(cross_corr_n_images_button, 2, 3)
        grid_auto.addWidget(self.n_to_cc_input, 2, 4)
        grid_auto.addWidget(shift_button, 3, 3)
        grid_auto.addWidget(warp_button, 4, 3)
        grid_auto.addWidget(self.det_df_checkbox, 1, 5)
        grid_auto.addWidget(self.df_min_label, 2, 5)
        grid_auto.addWidget(df_max_label, 3, 5)
        grid_auto.addWidget(df_step_label, 4, 5)
        grid_auto.addWidget(self.df_min_input, 2, 6)
        grid_auto.addWidget(self.df_max_input, 3, 6)
        grid_auto.addWidget(self.df_step_input, 4, 6)
        grid_auto.addWidget(set_df_button, 5, 6)

        self.tab_align = QtWidgets.QWidget()
        self.tab_align.layout = QtWidgets.QVBoxLayout()
        self.tab_align.layout.addLayout(grid_manual)
        self.tab_align.layout.addLayout(grid_auto)
        self.tab_align.setLayout(self.tab_align.layout)

        # ------------------------------
        # IWFR panel (5)
        # ------------------------------

        run_ewr_next_button = QtWidgets.QPushButton('Run next EWR iters -->', self)
        reset_ewr_button = QtWidgets.QPushButton('Reset EWR', self)
        sum_button = QtWidgets.QPushButton('Sum', self)
        diff_button = QtWidgets.QPushButton('Diff', self)
        fft_button = QtWidgets.QPushButton('FFT', self)
        amplify_button = QtWidgets.QPushButton('Amplify', self)
        plot_button = QtWidgets.QPushButton('Plot profile', self)

        in_focus_label = QtWidgets.QLabel('In-focus image number', self)
        start_num_label = QtWidgets.QLabel('Starting image number', self)
        n_to_ewr_label = QtWidgets.QLabel('Num. of images to use', self)
        aperture_label = QtWidgets.QLabel('Aperture radius [px]', self)
        hann_win_label = QtWidgets.QLabel('Hann window [px]', self)
        amp_factor_label = QtWidgets.QLabel('Amp. factor', self)
        int_width_label = QtWidgets.QLabel('Profile width [px]', self)

        self.in_focus_input = QtWidgets.QLineEdit('1', self)
        self.start_num_input = QtWidgets.QLineEdit('1', self)
        self.n_to_ewr_input = QtWidgets.QLineEdit(str(self.display.n_imgs), self)
        self.n_iters_input = QtWidgets.QLineEdit('10', self)
        self.aperture_input = QtWidgets.QLineEdit(str(const.aperture), self)
        self.hann_win_input = QtWidgets.QLineEdit(str(const.hann_win), self)
        self.amp_factor_input = QtWidgets.QLineEdit('2.0', self)
        self.int_width_input = QtWidgets.QLineEdit('1', self)

        self.use_aberrs_checkbox = QtWidgets.QCheckBox('Use aberrations', self)
        self.use_aberrs_checkbox.setChecked(False)

        self.det_abs_df_checkbox = QtWidgets.QCheckBox('Det. abs. defoc. values', self)
        self.det_abs_df_checkbox.setChecked(True)

        self.show_aperture_checkbox = QtWidgets.QCheckBox('display', self)
        self.show_aperture_checkbox.setChecked(False)
        self.show_aperture_checkbox.toggled.connect(self.toggle_aperture)

        self.show_hann_win_checkbox = QtWidgets.QCheckBox('display', self)
        self.show_hann_win_checkbox.setChecked(False)
        self.show_hann_win_checkbox.toggled.connect(self.toggle_hann_win)

        run_ewr_next_button.clicked.connect(self.run_ewr_next_iters)
        reset_ewr_button.clicked.connect(self.reset_ewr)
        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)
        fft_button.clicked.connect(self.disp_fft)
        amplify_button.clicked.connect(self.amplify_phase)
        plot_button.clicked.connect(self.plot_profile)

        grid_iwfr = QtWidgets.QGridLayout()
        grid_iwfr.setColumnStretch(0, 1)
        grid_iwfr.setColumnStretch(1, 1)
        grid_iwfr.setColumnStretch(2, 1)
        grid_iwfr.setColumnStretch(3, 1)
        grid_iwfr.setColumnStretch(4, 1)
        grid_iwfr.setColumnStretch(5, 1)
        grid_iwfr.setColumnStretch(6, 1)
        grid_iwfr.setRowStretch(0, 1)
        grid_iwfr.setRowStretch(7, 1)
        grid_iwfr.addWidget(start_num_label, 1, 1)
        grid_iwfr.addWidget(self.start_num_input, 1, 2)
        grid_iwfr.addWidget(in_focus_label, 2, 1)
        grid_iwfr.addWidget(self.in_focus_input, 2, 2)
        grid_iwfr.addWidget(n_to_ewr_label, 3, 1)
        grid_iwfr.addWidget(self.n_to_ewr_input, 3, 2)
        grid_iwfr.addWidget(self.n_iters_input, 5, 2)
        grid_iwfr.addWidget(aperture_label, 2, 4)
        grid_iwfr.addWidget(self.aperture_input, 3, 4)
        grid_iwfr.addWidget(hann_win_label, 4, 4)
        grid_iwfr.addWidget(self.hann_win_input, 5, 4)
        grid_iwfr.addWidget(self.show_aperture_checkbox, 3, 5)
        grid_iwfr.addWidget(self.show_hann_win_checkbox, 5, 5)
        grid_iwfr.addWidget(self.use_aberrs_checkbox, 1, 4)
        grid_iwfr.addWidget(self.det_abs_df_checkbox, 4, 1)
        grid_iwfr.addWidget(run_ewr_next_button, 5, 1)
        grid_iwfr.addWidget(reset_ewr_button, 5, 3)

        grid_calc = QtWidgets.QGridLayout()
        grid_calc.setColumnStretch(0, 1)
        grid_calc.setColumnStretch(1, 1)
        grid_calc.setColumnStretch(2, 1)
        grid_calc.setColumnStretch(3, 1)
        grid_calc.setColumnStretch(4, 1)
        grid_calc.setRowStretch(0, 1)
        grid_calc.setRowStretch(4, 1)
        grid_calc.addWidget(fft_button, 1, 1)
        grid_calc.addWidget(sum_button, 2, 1)
        grid_calc.addWidget(diff_button, 3, 1)
        grid_calc.addWidget(amp_factor_label, 1, 2)
        grid_calc.addWidget(self.amp_factor_input, 2, 2)
        grid_calc.addWidget(amplify_button, 3, 2)
        grid_calc.addWidget(plot_button, 3, 3)
        grid_calc.addWidget(int_width_label, 1, 3)
        grid_calc.addWidget(self.int_width_input, 2, 3)

        self.tab_rec = QtWidgets.QWidget()
        self.tab_rec.layout = QtWidgets.QVBoxLayout()
        self.tab_rec.layout.addLayout(grid_iwfr)
        self.tab_rec.layout.addLayout(grid_calc)
        self.tab_rec.setLayout(self.tab_rec.layout)

        # ------------------------------
        # Simulations panel (7)
        # ------------------------------

        sim_button = QtWidgets.QPushButton('Simulate', self)

        df_sim_1_label = QtWidgets.QLabel('Start defocus [nm]', self)
        df_sim_2_label = QtWidgets.QLabel('Stop defocus [nm]', self)
        df_sim_3_label = QtWidgets.QLabel('Step [nm]', self)
        A1_sim_label = QtWidgets.QLabel('A1 [nm]', self)
        phi1_sim_label = QtWidgets.QLabel('A1 angle [deg]', self)

        self.df_sim_1_input = QtWidgets.QLineEdit('0', self)
        self.df_sim_2_input = QtWidgets.QLineEdit('0', self)
        self.df_sim_3_input = QtWidgets.QLineEdit('0', self)
        self.A1_sim_input = QtWidgets.QLineEdit('0', self)
        self.phi1_sim_input = QtWidgets.QLineEdit('0', self)

        sim_button.clicked.connect(self.simulate_images_for_df)

        self.tab_sim = QtWidgets.QWidget()
        self.tab_sim.layout = QtWidgets.QGridLayout()
        self.tab_sim.layout.setColumnStretch(0, 1)
        self.tab_sim.layout.setColumnStretch(1, 1)
        self.tab_sim.layout.setColumnStretch(2, 1)
        self.tab_sim.layout.setColumnStretch(3, 1)
        self.tab_sim.layout.setColumnStretch(4, 1)
        self.tab_sim.layout.setColumnStretch(5, 1)
        self.tab_sim.layout.setRowStretch(0, 1)
        self.tab_sim.layout.setRowStretch(6, 1)
        self.tab_sim.layout.addWidget(df_sim_1_label, 1, 2)
        self.tab_sim.layout.addWidget(df_sim_2_label, 1, 3)
        self.tab_sim.layout.addWidget(df_sim_3_label, 1, 4)
        self.tab_sim.layout.addWidget(sim_button, 2, 1)
        self.tab_sim.layout.addWidget(self.df_sim_1_input, 2, 2)
        self.tab_sim.layout.addWidget(self.df_sim_2_input, 2, 3)
        self.tab_sim.layout.addWidget(self.df_sim_3_input, 2, 4)
        self.tab_sim.layout.addWidget(A1_sim_label, 4, 2)
        self.tab_sim.layout.addWidget(self.A1_sim_input, 4, 3)
        self.tab_sim.layout.addWidget(phi1_sim_label, 5, 2)
        self.tab_sim.layout.addWidget(self.phi1_sim_input, 5, 3)
        self.tab_sim.setLayout(self.tab_sim.layout)

        # ------------------------------
        # Bright/Gamma/Contrast panel (7)
        # ------------------------------

        bright_label = QtWidgets.QLabel('Brightness', self)
        cont_label = QtWidgets.QLabel('Contrast', self)
        gamma_label = QtWidgets.QLabel('Gamma', self)

        self.bright_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bright_slider.setFixedHeight(14)
        self.bright_slider.setRange(-255, 255)
        self.bright_slider.setValue(0)

        self.cont_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cont_slider.setFixedHeight(14)
        self.cont_slider.setRange(1, 1785)
        self.cont_slider.setValue(255)

        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setFixedHeight(14)
        self.gamma_slider.setRange(10, 190)
        self.gamma_slider.setValue(100)

        self.bright_input = QtWidgets.QLineEdit('0', self)
        self.cont_input = QtWidgets.QLineEdit('255', self)
        self.gamma_input = QtWidgets.QLineEdit('1.0', self)

        reset_bright_button = QtWidgets.QPushButton('Reset B', self)
        reset_cont_button = QtWidgets.QPushButton('Reset C', self)
        reset_gamma_button = QtWidgets.QPushButton('Reset G', self)

        self.bright_slider.valueChanged.connect(self.disp_bright_value)
        self.cont_slider.valueChanged.connect(self.disp_cont_value)
        self.gamma_slider.valueChanged.connect(self.disp_gamma_value)

        self.bright_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.cont_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.gamma_slider.sliderReleased.connect(self.update_display_and_bcg)

        self.bright_input.returnPressed.connect(self.update_display_and_bcg)
        self.cont_input.returnPressed.connect(self.update_display_and_bcg)
        self.gamma_input.returnPressed.connect(self.update_display_and_bcg)

        reset_bright_button.clicked.connect(self.reset_bright)
        reset_cont_button.clicked.connect(self.reset_cont)
        reset_gamma_button.clicked.connect(self.reset_gamma)

        self.tab_corr = QtWidgets.QWidget()
        self.tab_corr.layout = QtWidgets.QGridLayout()
        self.tab_corr.layout.setColumnStretch(0, 1)
        self.tab_corr.layout.setColumnStretch(1, 2)
        self.tab_corr.layout.setColumnStretch(2, 1)
        self.tab_corr.layout.setColumnStretch(3, 1)
        self.tab_corr.layout.setColumnStretch(4, 1)
        self.tab_corr.layout.setRowStretch(0, 1)
        self.tab_corr.layout.setRowStretch(7, 1)
        self.tab_corr.layout.addWidget(bright_label, 1, 2)
        self.tab_corr.layout.addWidget(self.bright_slider, 2, 1)
        self.tab_corr.layout.addWidget(self.bright_input, 2, 2)
        self.tab_corr.layout.addWidget(reset_bright_button, 2, 3)
        self.tab_corr.layout.addWidget(cont_label, 3, 2)
        self.tab_corr.layout.addWidget(self.cont_slider, 4, 1)
        self.tab_corr.layout.addWidget(self.cont_input, 4, 2)
        self.tab_corr.layout.addWidget(reset_cont_button, 4, 3)
        self.tab_corr.layout.addWidget(gamma_label, 5, 2)
        self.tab_corr.layout.addWidget(self.gamma_slider, 6, 1)
        self.tab_corr.layout.addWidget(self.gamma_input, 6, 2)
        self.tab_corr.layout.addWidget(reset_gamma_button, 6, 3)
        self.tab_corr.setLayout(self.tab_corr.layout)

        # ------------------------------
        # Main layout
        # ------------------------------

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab_nav, 'Navigation')
        self.tabs.addTab(self.tab_disp, 'Display')
        self.tabs.addTab(self.tab_align, 'Alignment')
        self.tabs.addTab(self.tab_rec, 'Reconstruction')
        self.tabs.addTab(self.tab_sim, 'Simulation')
        self.tabs.addTab(self.tab_corr, 'Corrections')

        vbox_panel = QtWidgets.QVBoxLayout()
        vbox_panel.addWidget(self.curr_info_label)
        vbox_panel.addWidget(self.tabs)
        vbox_panel.addWidget(self.plot_widget)

        hbox_main = QtWidgets.QHBoxLayout()
        hbox_main.addWidget(self.display)
        hbox_main.addLayout(vbox_panel)

        self.setLayout(hbox_main)
        # self.reset_image_names()

        self.move(250, 50)
        self.setWindowTitle('Holo window')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

    def update_curr_info_label(self):
        curr_img = self.display.image
        self.curr_info_label.setText('{0}, dim = {1} px, df = {2:.0f} nm'.format(curr_img.name, curr_img.width, curr_img.defocus * 1e9))

    def enable_manual_panel(self):
        self.left_button.setEnabled(True)
        self.right_button.setEnabled(True)
        self.up_button.setEnabled(True)
        self.down_button.setEnabled(True)
        self.rot_clockwise_button.setEnabled(True)
        self.rot_counter_clockwise_button.setEnabled(True)
        self.px_shift_input.setEnabled(True)
        self.rot_angle_input.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.reset_button.setEnabled(True)

    def disable_manual_panel(self):
        self.left_button.setEnabled(False)
        self.right_button.setEnabled(False)
        self.up_button.setEnabled(False)
        self.down_button.setEnabled(False)
        self.rot_clockwise_button.setEnabled(False)
        self.rot_counter_clockwise_button.setEnabled(False)
        self.px_shift_input.setEnabled(False)
        self.rot_angle_input.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.reset_button.setEnabled(False)

    def enable_df_inputs(self):
        self.df_min_label.setText('df min [nm]')
        self.df_max_input.setEnabled(True)
        self.df_step_input.setEnabled(True)

    def disable_df_inputs(self):
        self.df_min_label.setText('df const [nm]')
        self.df_max_input.setEnabled(False)
        self.df_step_input.setEnabled(False)

    def manage_df_inputs(self):
        if self.det_df_checkbox.isChecked():
            self.enable_df_inputs()
        else:
            self.disable_df_inputs()

    def set_image_name(self):
        self.display.image.name = self.name_input.text()
        self.fname_input.setText(self.name_input.text())

    def reset_image_names(self):
        curr_img = self.display.image
        first_img = imsup.GetFirstImage(curr_img)
        img_queue = imsup.CreateImageListFromFirstImage(first_img)
        for img, idx in zip(img_queue, range(len(img_queue))):
            img.numInSeries = idx + 1
            img.name = 'img0{0}'.format(idx+1) if idx < 9 else 'img{0}'.format(idx+1)
        self.update_curr_info_label()
        self.name_input.setText(curr_img.name)
        self.fname_input.setText(curr_img.name)

    def set_defocus(self):
        self.display.image.defocus = float(self.df_min_input.text()) * 1e-9

    def go_to_image(self, new_idx):
        is_show_labels_checked = self.show_labels_checkbox.isChecked()
        first_img = imsup.GetFirstImage(self.display.image)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        if new_idx > len(imgs) - 1:
            new_idx = len(imgs) - 1
        curr_img = imgs[new_idx]
        if curr_img.name == '':
            curr_img.name = 'img0{0}'.format(new_idx + 1) if new_idx < 9 else 'img{0}'.format(new_idx + 1)
        self.name_input.setText(curr_img.name)
        self.fname_input.setText(curr_img.name)
        self.manual_mode_checkbox.setChecked(False)
        self.disable_manual_panel()
        self.display.image = curr_img
        self.display.update_labs(is_show_labels_checked)
        self.update_curr_info_label()
        self.update_display_and_bcg()

    def go_to_prev_image(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            return
        prev_idx = curr_img.prev.numInSeries - 1
        self.go_to_image(prev_idx)

    def go_to_next_image(self):
        curr_img = self.display.image
        if curr_img.next is None:
            return
        next_idx = curr_img.next.numInSeries - 1
        self.go_to_image(next_idx)

    def go_to_last_image(self):
        curr_img = self.display.image
        last_img = imsup.GetLastImage(curr_img)
        last_idx = last_img.numInSeries - 1
        self.go_to_image(last_idx)

    def flip_image_h(self):
        imsup.flip_image_h(self.display.image)
        self.display.setImage()

    def export_image(self):
        curr_num = self.display.image.numInSeries
        curr_img = self.display.image
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        fname = self.fname_input.text()
        if fname == '':
            if is_amp_checked:
                fname = 'amp{0}'.format(curr_num)
            elif is_phs_checked:
                fname = 'phs{0}'.format(curr_num)
            else:
                fname = 'cos_phs{0}'.format(curr_num)

        if self.export_bin_radio_button.isChecked():
            fname_ext = ''
            if is_amp_checked:
                # np.save(fname, curr_img.amPh.am)
                curr_img.amPh.am.tofile(fname)
            elif is_phs_checked:
                # np.save(fname, curr_img.amPh.ph)
                curr_img.amPh.ph.tofile(fname)
            else:
                cos_phs = np.cos(curr_img.amPh.ph)
                cos_phs.tofile(fname)
                # np.save(fname, cos_phs)
            print('Saved image to binary file: "{0}"'.format(fname))
        else:
            fname_ext = '.tif'
            img_fname = '{0}{1}'.format(fname, fname_ext)
            log = True if self.log_scale_checkbox.isChecked() else False
            color = True if self.color_radio_button.isChecked() else False

            if is_amp_checked:
                imsup.SaveAmpImage(curr_img, img_fname, log, color)
            elif is_phs_checked:
                imsup.SavePhaseImage(curr_img, img_fname, log, color)
            else:
                phs_tmp = np.copy(curr_img.amPh.ph)
                curr_img.amPh.ph = np.cos(phs_tmp)
                imsup.SavePhaseImage(curr_img, img_fname, log, color)
                curr_img.amPh.ph = np.copy(phs_tmp)
            print('Saved image as "{0}"'.format(img_fname))

        # save log file
        log_fname = '{0}_log.txt'.format(fname)
        with open(log_fname, 'w') as log_file:
            log_file.write('File name:\t{0}{1}\n'
                           'Image name:\t{2}\n'
                           'Image size:\t{3}x{4}\n'
                           'Calibration:\t{5} nm\n'.format(fname, fname_ext, curr_img.name, curr_img.width, curr_img.height, curr_img.px_dim * 1e9))
        print('Saved log file: "{0}"'.format(log_fname))

    def export_all(self):
        first_img = imsup.GetFirstImage(self.display.image)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        for img in imgs:
            self.go_to_image(img.numInSeries - 1)
            self.export_image()
        print('All images saved')

    def delete_image(self):
        curr_img = self.display.image
        if curr_img.prev is None and curr_img.next is None:
            return

        curr_idx = curr_img.numInSeries - 1
        first_img = imsup.GetFirstImage(curr_img)
        all_img_list = imsup.CreateImageListFromFirstImage(first_img)

        new_idx = curr_idx - 1 if curr_img.prev is not None else curr_idx + 1
        self.go_to_image(new_idx)

        del all_img_list[curr_idx]
        del self.display.pointSets[curr_idx]

    def toggle_lines(self):
        self.display.show_lines = not self.display.show_lines
        self.display.repaint()

    def toggle_labels(self):
        self.display.show_labs = not self.display.show_labs
        if self.display.show_labs:
            self.display.show_labels()
        else:
            self.display.hide_labels()

    def update_display(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()
        self.display.setImage(dispAmp=is_amp_checked, dispPhs=is_phs_checked,
                              logScale=is_log_scale_checked, color=is_color_checked)

    def update_bcg(self):
        bright_val = int(self.bright_input.text())
        cont_val = int(self.cont_input.text())
        gamma_val = float(self.gamma_input.text())

        self.change_bright_slider_value()
        self.change_cont_slider_value()
        self.change_gamma_slider_value()

        self.display.setImage(update_bcg=True, bright=bright_val, cont=cont_val, gamma=gamma_val)

    def update_display_and_bcg(self):
        is_amp_checked = self.amp_radio_button.isChecked()
        is_phs_checked = self.phs_radio_button.isChecked()
        is_log_scale_checked = self.log_scale_checkbox.isChecked()
        is_color_checked = self.color_radio_button.isChecked()

        bright_val = int(self.bright_input.text())
        cont_val = int(self.cont_input.text())
        gamma_val = float(self.gamma_input.text())

        self.change_bright_slider_value()
        self.change_cont_slider_value()
        self.change_gamma_slider_value()

        self.display.setImage(dispAmp=is_amp_checked, dispPhs=is_phs_checked,
                              logScale=is_log_scale_checked, color=is_color_checked,
                              update_bcg=True, bright=bright_val, cont=cont_val, gamma=gamma_val)

    def disp_bright_value(self):
        self.bright_input.setText('{0:.0f}'.format(self.bright_slider.value()))

    def disp_cont_value(self):
        self.cont_input.setText('{0:.0f}'.format(self.cont_slider.value()))

    def disp_gamma_value(self):
        self.gamma_input.setText('{0:.2f}'.format(self.gamma_slider.value() * 0.01))

    def change_bright_slider_value(self):
        b = int(self.bright_input.text())
        b_min = self.bright_slider.minimum()
        b_max = self.bright_slider.maximum()
        if b < b_min:
            b = b_min
        elif b > b_max:
            b = b_max
        self.bright_slider.setValue(b)

    def change_cont_slider_value(self):
        c = int(self.cont_input.text())
        c_min = self.cont_slider.minimum()
        c_max = self.cont_slider.maximum()
        if c < c_min:
            c = c_min
        elif c > c_max:
            c = c_max
        self.cont_slider.setValue(c)

    def change_gamma_slider_value(self):
        g = float(self.gamma_input.text()) * 100
        g_min = self.gamma_slider.minimum()
        g_max = self.gamma_slider.maximum()
        if g < g_min:
            g = g_min
        elif g > g_max:
            g = g_max
        self.gamma_slider.setValue(g)

    def reset_bright(self):
        self.bright_input.setText('0')
        self.update_display_and_bcg()

    def reset_cont(self):
        self.cont_input.setText('255')
        self.update_display_and_bcg()

    def reset_gamma(self):
        self.gamma_input.setText('1.0')
        self.update_display_and_bcg()

    def unwrap_img_phase(self):
        curr_img = self.display.image
        new_phs = tr.unwrap_phase(curr_img.amPh.ph)
        curr_img.amPh.ph = np.copy(new_phs)
        self.update_display()

    def wrap_img_phase(self):
        curr_img = self.display.image
        uw_min = np.min(curr_img.amPh.ph)

        if uw_min > 0:
            uw_min = 0
        new_phs = (curr_img.amPh.ph - uw_min) % (2 * np.pi) - np.pi

        curr_img.amPh.ph = np.copy(new_phs)
        self.update_display()

    def norm_phase(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) == 0:
            print('Mark the reference point on the image')
            return
        pt_disp = self.display.pointSets[curr_idx][0]
        pt_real = CalcRealTLCoords(curr_img.width, pt_disp)

        first_img = imsup.GetFirstImage(curr_img)
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        for img in img_list:
            new_phs = norm_phase_to_pt(img.amPh.ph, pt_real)
            img.amPh.ph = np.copy(new_phs)
            img.update_cos_phase()
        self.update_display()
        print('All phases normalized')

    def zoom_n_fragments(self):
        curr_idx = self.display.image.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) < 2:
            print('You have to mark two points on the image in order to zoom!')
            return

        curr_img = self.display.image
        pt1, pt2 = self.display.pointSets[curr_idx][:2]
        pt1, pt2 = convert_points_to_tl_br(pt1, pt2)
        disp_crop_coords = pt1 + pt2
        real_tl_coords = CalcRealTLCoords(curr_img.width, disp_crop_coords)
        real_sq_coords = imsup.MakeSquareCoords(real_tl_coords)

        n_to_zoom = np.int(self.n_to_zoom_input.text())
        first_img = imsup.GetFirstImage(curr_img)
        insert_idx = curr_idx + n_to_zoom
        img_list = imsup.CreateImageListFromFirstImage(first_img)
        img_list2 = img_list[curr_idx:insert_idx]

        for img, n in zip(img_list2, range(insert_idx, insert_idx + n_to_zoom)):
            frag = zoom_fragment(img, real_sq_coords)
            img_list.insert(n, frag)
            self.display.pointSets.insert(n, [])
            print('{0} cropped'.format(img.name))

        img_list.UpdateLinks()

        if self.clear_prev_checkbox.isChecked():
            del img_list[curr_idx:insert_idx]
            del self.display.pointSets[curr_idx:insert_idx]
            self.display.image = img_list[curr_idx]       # !!!

        self.go_to_image(curr_idx)
        print('Zooming complete!')

    def clear_image(self):
        labToDel = self.display.children()
        for child in labToDel:
            child.deleteLater()
        self.display.pointSets[self.display.image.numInSeries - 1][:] = []
        self.display.repaint()

    def remove_last_point(self):
        curr_idx = self.display.image.numInSeries - 1
        if len(self.display.pointSets[curr_idx]) == 0:
            return
        all_labels = self.display.children()
        if len(all_labels) > 0:
            last_label = all_labels[-1]
            last_label.deleteLater()
        del self.display.pointSets[curr_idx][-1]
        self.display.repaint()

    def create_backup_image(self):
        if self.manual_mode_checkbox.isChecked():
            if self.backup_image is None:
                self.backup_image = imsup.copy_am_ph_image(self.display.image)
            self.enable_manual_panel()
        else:
            self.disable_manual_panel()

    def move_left(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([0, -n_px])

    def move_right(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([0, n_px])

    def move_up(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([-n_px, 0])

    def move_down(self):
        n_px = int(self.px_shift_input.text())
        self.move_image([n_px, 0])

    def move_image(self, shift):
        bckp = self.backup_image
        curr = self.display.image
        total_shift = list(np.array(curr.shift) + np.array(shift))

        if curr.rot != 0:
            tmp = tr.RotateImageSki(bckp, curr.rot)
            shifted_img = cc.shift_am_ph_image(tmp, total_shift)
        else:
            shifted_img = cc.shift_am_ph_image(bckp, total_shift)

        curr.amPh.am = np.copy(shifted_img.amPh.am)
        curr.amPh.ph = np.copy(shifted_img.amPh.ph)
        curr.shift = total_shift
        self.display.setImage()

    def rot_left(self):
        ang = float(self.rot_angle_input.text())
        self.rotate_image(ang)

    def rot_right(self):
        ang = float(self.rot_angle_input.text())
        self.rotate_image(-ang)

    def rotate_image(self, rot):
        bckp = self.backup_image
        curr = self.display.image
        total_rot = curr.rot + rot

        if curr.shift != 0:
            tmp = cc.shift_am_ph_image(bckp, curr.shift)
            rotated_img = tr.RotateImageSki(tmp, total_rot)
        else:
            rotated_img = tr.RotateImageSki(bckp, total_rot)

        curr.amPh.am = np.copy(rotated_img.amPh.am)
        curr.amPh.ph = np.copy(rotated_img.amPh.ph)
        curr.rot = total_rot
        self.display.setImage()

    def repeat_prev_mods(self):
        curr = imsup.copy_am_ph_image(self.backup_image)
        for mod in self.changes_made:
            curr = modify_image(curr, mod[:2], bool(mod[2]))
        self.display.image = curr

    def apply_changes(self):
        self.backup_image = None

    def reset_changes(self):
        self.display.image = imsup.copy_am_ph_image(self.backup_image)
        self.backup_image = None
        # self.changes_made = []
        self.display.setImage()

    def disp_fft(self):
        curr_img = self.display.image
        curr_fft = cc.FFT(curr_img)
        curr_fft = cc.FFT2Diff(curr_fft)
        curr_fft.ReIm2AmPh()
        curr_fft.MoveToCPU()
        curr_fft = imsup.create_imgexp_from_img(curr_fft)
        curr_fft = rescale_image_buffer_to_window(curr_fft, const.ccWidgetDim)
        self.insert_img_after_curr(curr_fft)
        self.log_scale_checkbox.setChecked(True)

    def toggle_aperture(self):
        aper_r = int(self.aperture_input.text())
        aper_d = 2 * aper_r
        self.display.aper_diam = real_to_disp_len(const.ccWidgetDim, self.display.image.width, aper_d)
        self.display.show_aper = not self.display.show_aper
        self.display.repaint()

    def toggle_hann_win(self):
        hann_w = int(self.hann_win_input.text())
        self.display.hann_width = real_to_disp_len(const.ccWidgetDim, self.display.image.width, hann_w)
        self.display.show_hann = not self.display.show_hann
        self.display.repaint()

    def cross_corr_with_prev(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            print('There is no reference image!')
            return
        img_list_to_cc = imsup.CreateImageListFromImage(curr_img.prev, 2)
        img_aligned = self.cross_corr_core(img_list_to_cc)[0]
        self.insert_img_after_curr(img_aligned)
        self.display.frag_coords = []
        self.btn_grid.create_grid()
        self.go_to_next_image()

    def cross_corr_n_images(self):
        n_to_cc = int(self.n_to_cc_input.text())
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        insert_idx = curr_idx + n_to_cc
        first_img = imsup.GetFirstImage(curr_img)
        all_img_list = imsup.CreateImageListFromFirstImage(first_img)
        n_imgs = len(all_img_list)
        if insert_idx > n_imgs:
            n_to_cc = n_imgs - curr_idx
            insert_idx = curr_idx + n_to_cc
        img_list_to_cc = all_img_list[curr_idx:insert_idx]
        img_align_list = self.cross_corr_core(img_list_to_cc)

        for img, n in zip(img_align_list, range(insert_idx, insert_idx + n_to_cc)):
            all_img_list.insert(n, img)
            self.display.pointSets.insert(n, [])

        all_img_list.UpdateLinks()

        if self.clear_prev_checkbox.isChecked():
            del all_img_list[curr_idx:insert_idx]
            del self.display.pointSets[curr_idx:insert_idx]
            self.display.image = all_img_list[curr_idx]     # !!!

        self.display.frag_coords = []
        self.btn_grid.create_grid()
        self.go_to_image(curr_idx)
        print('Cross-correlation done!')

    def cross_corr_core(self, img_list_to_cc):
        self.get_clicked_coords()
        if self.det_df_checkbox.isChecked():
            df_min = float(self.df_min_input.text())
            df_max = float(self.df_max_input.text())
            df_step = float(self.df_step_input.text())
            img_align_list = cross_corr_images(img_list_to_cc, self.btn_grid.n_rows, self.display.frag_coords[1:],
                                               df_min=df_min, df_max=df_max, df_step=df_step)
        else:
            df_const = float(self.df_min_input.text())
            img_align_list = cross_corr_images(img_list_to_cc, self.btn_grid.n_rows, self.display.frag_coords[1:],
                                               df_min=df_const)
        return img_align_list

    def align_shift(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        img_width = curr_img.width

        points1 = self.display.pointSets[curr_idx - 1]
        points2 = self.display.pointSets[curr_idx]
        n_points1 = len(points1)
        n_points2 = len(points2)

        if n_points1 != n_points2:
            print('Mark the same number of points on both images!')
            return

        set1 = [CalcRealCoords(img_width, pt1) for pt1 in points1]
        set2 = [CalcRealCoords(img_width, pt2) for pt2 in points2]

        shift_sum = np.zeros(2, dtype=np.int32)
        for pt1, pt2 in zip(set1, set2):
            shift = np.array(pt1) - np.array(pt2)
            shift_sum += shift

        shift_avg = list(shift_sum // n_points1)
        shift_avg.reverse()     # !!!
        self.shift = shift_avg

        shifted_img2 = cc.shift_am_ph_image(curr_img, shift_avg)
        shifted_img2 = imsup.create_imgexp_from_img(shifted_img2)
        self.insert_img_after_curr(shifted_img2)

    def reshift(self):
        curr_img = self.display.image
        shift = self.shift

        shifted_img = cc.shift_am_ph_image(curr_img, shift)
        shifted_img = imsup.create_imgexp_from_img(shifted_img)
        self.insert_img_after_curr(shifted_img)

    def warp_image(self, more_accurate=False):
        curr_img = self.display.image
        curr_idx = self.display.image.numInSeries - 1
        real_points1 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx-1])
        real_points2 = CalcRealCoordsForSetOfPoints(curr_img.width, self.display.pointSets[curr_idx])
        user_points1 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points1)
        user_points2 = CalcTopLeftCoordsForSetOfPoints(curr_img.width, real_points2)

        self.warp_points = [ user_points1, user_points2 ]

        if more_accurate:
            n_div = const.nDivForUnwarp
            frag_dim_size = curr_img.width // n_div

            # points #1
            grid_points1 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points1 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points1 ]

            for pt1 in user_points1:
                closest_node = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt1 ]
                grid_points1 = [ pt1 if grid_node == closest_node else grid_node for grid_node in grid_points1 ]

            src = np.array(grid_points1)

            # points #2
            grid_points2 = [ (b, a) for a in range(n_div) for b in range(n_div) ]
            grid_points2 = [ [ gptx * frag_dim_size for gptx in gpt ] for gpt in grid_points2 ]
            for pt2 in user_points2:
                closestNode = [ np.floor(x / frag_dim_size) * frag_dim_size for x in pt2 ]
                grid_points2 = [ pt2 if gridNode == closestNode else gridNode for gridNode in grid_points2 ]

            dst = np.array(grid_points2)

        else:
            src = np.array(user_points1)
            dst = np.array(user_points2)

        img_warp = tr.WarpImage(curr_img, src, dst)

        curr_num = self.display.image.numInSeries
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.insert(1, img_warp)
        tmp_img_list.UpdateLinks()
        self.display.pointSets.insert(curr_num, [])
        self.go_to_next_image()

    def rewarp(self):
        curr_img = self.display.image
        user_pts1 = self.warp_points[0]
        user_pts2 = self.warp_points[1]

        src = np.array(user_pts1)
        dst = np.array(user_pts2)

        warped_img = tr.WarpImage(curr_img, src, dst)
        self.insert_img_after_curr(warped_img)

    def insert_img_after_curr(self, img):
        curr_num = self.display.image.numInSeries
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.insert(1, img)
        self.display.pointSets.insert(curr_num, [])
        tmp_img_list.UpdateLinks()
        self.go_to_next_image()

    def insert_img_last(self, img):
        img.next = None
        tmp_img_list = imsup.CreateImageListFromFirstImage(self.display.image)
        tmp_img_list.append(img)
        self.display.pointSets.append([])
        tmp_img_list.UpdateLinks()

    def calc_phs_sum(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_sum = holo.calc_phase_sum(rec_holo1, rec_holo2)
        self.insert_img_after_curr(phs_sum)

    def calc_phs_diff(self):
        rec_holo1 = self.display.image.prev
        rec_holo2 = self.display.image

        phs_diff = holo.calc_phase_diff(rec_holo1, rec_holo2)
        np.savetxt('ph_diff', phs_diff.amPh.ph)     # !!!
        self.insert_img_after_curr(phs_diff)

    def amplify_phase(self):
        curr_img = self.display.image
        curr_name = self.name_input.text()
        amp_factor = float(self.amp_factor_input.text())

        phs_amplified = imsup.copy_am_ph_image(curr_img)
        phs_amplified.amPh.ph *= amp_factor
        phs_amplified.update_cos_phase()
        phs_amplified.name = '{0}_x{1:.0f}'.format(curr_name, amp_factor)
        self.insert_img_after_curr(phs_amplified)
        self.cos_phs_radio_button.setChecked(True)

    def swap_left(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        prev_idx = curr_idx - 1

        first_img = imsup.GetFirstImage(curr_img)
        imgs = imsup.CreateImageListFromFirstImage(first_img)

        if curr_img.prev is None:
            prev_idx = len(imgs) - 1

        imgs[prev_idx], imgs[curr_idx] = imgs[curr_idx], imgs[prev_idx]
        imgs[0].prev = None
        imgs[len(imgs)-1].next = None

        left_idx, right_idx = min(prev_idx, curr_idx), max(prev_idx, curr_idx)
        imgs[left_idx].numInSeries = imgs[right_idx].numInSeries
        imgs.UpdateLinks()

        ps = self.display.pointSets
        if len(ps[prev_idx]) > 0:
            ps[prev_idx], ps[curr_idx] = ps[curr_idx], ps[prev_idx]

        if prev_idx != len(imgs) - 1:
            self.go_to_next_image()
        else:
            self.go_to_image(curr_idx)

    def swap_right(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        next_idx = curr_idx + 1

        if curr_img.next is None:
            next_idx = 0

        first_img = imsup.GetFirstImage(curr_img)
        imgs = imsup.CreateImageListFromFirstImage(first_img)
        imgs[curr_idx], imgs[next_idx] = imgs[next_idx], imgs[curr_idx]

        imgs[0].prev = None
        imgs[len(imgs)-1].next = None

        left_idx, right_idx = min(curr_idx, next_idx), max(curr_idx, next_idx)
        imgs[left_idx].numInSeries = imgs[right_idx].numInSeries
        imgs.UpdateLinks()

        ps = self.display.pointSets
        if len(ps[curr_idx]) > 0:
            ps[curr_idx], ps[next_idx] = ps[next_idx], ps[curr_idx]

        if next_idx != 0:
            self.go_to_prev_image()
        else:
            self.go_to_image(curr_idx)

    def get_clicked_coords(self):
        n_frags = self.btn_grid.count()
        frag_coords = [ int(np.sqrt(n_frags)) ]
        for pos in range(n_frags):
            btn = self.btn_grid.itemAt(pos).widget()
            if btn.was_clicked:
                values = re.search('([0-9]), ([0-9])', btn.text())
                frag_pos = (int(values.group(2)), int(values.group(1)))
                frag_coords.append(frag_pos)
        self.display.frag_coords = frag_coords
        self.display.repaint()

    def run_ewr(self):
        start_num = int(self.start_num_input.text())
        n_to_ewr = int(self.n_to_ewr_input.text())
        n_iters = int(self.n_iters_input.text())

        curr_img = self.display.image
        tmp = imsup.GetFirstImage(curr_img)
        for idx in range(start_num):
            tmp = tmp.next
        if tmp is None:
            print('Starting number is invalid!')
            return
        start_img = tmp

        imgs_from_start = imsup.CreateImageListFromFirstImage(start_img)
        n_imgs = len(imgs_from_start)

        if (start_img.numInSeries - 1) + n_to_ewr > n_imgs:
            n_to_ewr = n_imgs - (start_img.numInSeries - 1)

        imgs_to_iwfr = imsup.CreateImageListFromImage(start_img, n_to_ewr)

        if self.det_abs_df_checkbox.isChecked():
            idx_in_focus = int(self.in_focus_input.text()) - 1
            cc.DetermineAbsoluteDefocus(imgs_to_iwfr, idx_in_focus)
        # else:
        #    cc.shift_absolute_defocus(imgs_to_iwfr, idx_in_focus)

        for img in imgs_to_iwfr:
            print("{0:.2f} nm".format(img.defocus * 1e9))

        print('Starting IWFR...')
        # exit_wave = prop.run_iwfr(imgs_to_iwfr, n_iters)

        for i in range(0, n_iters):
            print('Iteration no {0}...'.format(i + 1))
            imgs_to_iwfr, exit_wave = prop.run_iteration_of_iwfr(imgs_to_iwfr, self.use_aberrs_checkbox.isChecked(),
                                                                 ap=int(self.aperture_input.text()),
                                                                 hann=int(self.hann_win_input.text()))
            ccfg.GetGPUMemoryUsed()
            exit_wave.ReIm2AmPh()
            exit_wave.MoveToCPU()
            exit_wave.name = 'wave_fun_0{0}'.format(i+1) if i < 9 else 'wave_fun_{0}'.format(i+1)
            exit_wave = rescale_image_buffer_to_window(exit_wave, const.ccWidgetDim)

            self.insert_img_last(exit_wave)
            self.go_to_last_image()
            if i < n_iters - 1:
                exit_wave.MoveToGPU()
                exit_wave.AmPh2ReIm()

        print('All done')

        # exit_wave.ReIm2AmPh()
        # exit_wave.MoveToCPU()
        # self.insert_img_last(exit_wave)
        # self.go_to_last_image()

    def run_ewr_next_iters(self):
        n_iters = int(self.n_iters_input.text())

        for it in range(n_iters):
            self.curr_iter += 1
            print('Iteration no {0}'.format(self.curr_iter))
            print('Backward propagation...')

            if self.curr_iter == 1:
                start_img_num = int(self.start_num_input.text())
                n_to_ewr = int(self.n_to_ewr_input.text())

                curr_img = self.display.image
                tmp = imsup.GetFirstImage(curr_img)

                for idx in range(start_img_num-1):
                    if tmp.next is None:
                        print('Starting number is invalid!')
                        break
                    tmp = tmp.next

                start_img = tmp
                imgs_from_start = imsup.CreateImageListFromFirstImage(start_img)
                n_imgs = len(imgs_from_start)

                if n_to_ewr > n_imgs:
                    n_to_ewr = n_imgs

                self.curr_ewr_imgs = imsup.CreateImageListFromImage(start_img, n_to_ewr)

                if self.det_abs_df_checkbox.isChecked():
                    idx_in_focus = int(self.in_focus_input.text()) - start_img_num
                    cc.DetermineAbsoluteDefocus(self.curr_ewr_imgs, idx_in_focus)
            # ---- (only 1st iteration) ----

            if it == 0:
                for img in self.curr_ewr_imgs:
                    print('{0:.1f} nm'.format(img.defocus * 1e9))

            self.curr_exit_wave = prop.run_backprop_iter(self.curr_ewr_imgs,
                                                         self.use_aberrs_checkbox.isChecked(),
                                                         ap=int(self.aperture_input.text()),
                                                         hann=int(self.hann_win_input.text()))

            print('Forward propagation...')
            tot_error = prop.run_forwprop_iter(self.curr_exit_wave, self.curr_ewr_imgs,
                                               self.use_aberrs_checkbox.isChecked(),
                                               ap=int(self.aperture_input.text()),
                                               hann=int(self.hann_win_input.text()))

            delta_tot_error = tot_error - self.last_tot_error
            self.last_tot_error = tot_error
            print('Total error = {0:.3f} %'.format(tot_error * 100))
            print('Delta error = {0:.3f} %'.format(delta_tot_error * 100))

        ccfg.GetGPUMemoryUsed()
        exit_wave_copy = imsup.CopyImage(self.curr_exit_wave)
        exit_wave_copy.ReIm2AmPh()
        exit_wave_copy.MoveToCPU()
        exit_wave_copy = rescale_image_buffer_to_window(exit_wave_copy, const.ccWidgetDim)

        c_it = self.curr_iter
        exit_wave_copy.name = 'wave_fun_0{0}'.format(c_it) if c_it < 10 else 'wave_fun_{0}'.format(c_it)
        self.insert_img_last(exit_wave_copy)
        self.go_to_last_image()
        print('Done')

    def reset_ewr(self):
        tmp_ddfs = [0]
        for img1, img2 in zip(self.curr_ewr_imgs[:-1], self.curr_ewr_imgs[1:]):
            tmp_ddfs.append(img1.defocus - img2.defocus)
        start_idx = int(self.start_num_input.text()) - 1
        end_idx = start_idx + int(self.n_to_ewr_input.text())
        first_img = imsup.GetFirstImage(self.display.image)
        new_img_list = imsup.CreateImageListFromFirstImage(first_img)
        for img, ddf in zip(new_img_list[start_idx:end_idx], tmp_ddfs):
            img.defocus = ddf
        self.curr_iter = 0
        self.curr_exit_wave = None
        self.last_tot_error = 0.0
        self.update_curr_info_label()
        # self.curr_ewr_imgs = None
        print('EWR procedure was reset')

    def plot_profile(self):
        curr_img = self.display.image
        curr_idx = curr_img.numInSeries - 1
        px_sz = curr_img.px_dim
        print(px_sz)
        points = self.display.pointSets[curr_idx][:2]
        points = np.array([ CalcRealCoords(curr_img.width, pt) for pt in points ])

        # find rotation center (center of the line)
        rot_center = np.average(points, 0).astype(np.int32)
        print('rotCenter = {0}'.format(rot_center))

        # find direction (angle) of the line
        dir_info = FindDirectionAngles(points[0], points[1])
        dir_angle = imsup.Degrees(dir_info[0])
        proj_dir = dir_info[2]
        print('dir angle = {0:.2f} deg'.format(dir_angle))

        # shift image by -center
        shift_to_rot_center = list(-rot_center)
        shift_to_rot_center.reverse()
        img_shifted = cc.shift_am_ph_image(curr_img, shift_to_rot_center)

        # rotate image by angle
        img_rot = tr.RotateImageSki(img_shifted, dir_angle)

        # crop fragment (height = distance between two points)
        pt_diffs = points[0] - points[1]
        frag_dim1 = int(np.sqrt(pt_diffs[0] ** 2 + pt_diffs[1] ** 2))
        frag_dim2 = int(self.int_width_input.text())

        if proj_dir == 0:
            frag_width, frag_height = frag_dim1, frag_dim2
        else:
            frag_width, frag_height = frag_dim2, frag_dim1

        frag_coords = imsup.DetermineCropCoordsForNewDims(img_rot.width, img_rot.height, frag_width, frag_height)
        print('Frag dims = {0}, {1}'.format(frag_width, frag_height))
        print('Frag coords = {0}'.format(frag_coords))
        img_cropped = imsup.crop_am_ph_roi_cpu(img_rot, frag_coords)

        # calculate projection of intensity
        if self.amp_radio_button.isChecked():
            int_matrix = np.copy(img_cropped.amPh.am)
        elif self.phs_radio_button.isChecked():
            ph_min = np.min(img_cropped.amPh.ph)
            ph_fix = -ph_min if ph_min < 0 else 0
            img_cropped.amPh.ph += ph_fix
            int_matrix = np.copy(img_cropped.amPh.ph)
        else:
            cos_ph_min = np.min(img_cropped.cos_phase)
            cos_ph_fix = -cos_ph_min if cos_ph_min < 0 else 0
            img_cropped.cos_phase += cos_ph_fix
            int_matrix = np.copy(img_cropped.cos_phase)

        int_profile = np.sum(int_matrix, proj_dir)  # 0 - horizontal projection, 1 - vertical projection
        dists = np.arange(0, int_profile.shape[0], 1) * px_sz
        dists *= 1e9

        self.plot_widget.plot(dists, int_profile, 'Distance [nm]', 'Intensity [a.u.]')

    def simulate_images_for_df(self):
        curr_img = self.display.image
        df1 = float(self.df_sim_1_input.text()) * 1e-9
        df2 = float(self.df_sim_2_input.text()) * 1e-9
        df3 = float(self.df_sim_3_input.text()) * 1e-9
        use_aberrs = self.use_aberrs_checkbox.isChecked()
        A1 = float(self.A1_sim_input.text()) * 1e-9
        phi1 = float(self.phi1_sim_input.text())
        aper = int(self.aperture_input.text())
        hann_win = int(self.hann_win_input.text())

        print('df1 = {0:.0f} nm\ndf2 = {1:.0f} nm\ndf3 = {2:.0f} nm'.format(df1 * 1e9, df2 * 1e9, df3 * 1e9))
        print('A1 amp = {0:.0f} nm\nA1 ang = {1:.0f} deg\nAperture = {2:.0f} px'.format(A1 * 1e9, phi1, aper))

        # const.A1_amp = A1
        # const.A1_phs = phi1

        sim_imgs = prop.simulate_images(curr_img, df1, df2, df3, use_aberrs, A1, phi1, aper, hann_win)
        for img in sim_imgs:
            img = imsup.create_imgexp_from_img(img)
            img = rescale_image_buffer_to_window(img, const.ccWidgetDim)
            img.name = curr_img.name + '_{0:.0f}nm'.format(img.defocus * 1e9)
            self.insert_img_after_curr(img)
            print('{0} added'.format(img.name))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_6:
            self.go_to_next_image()
        elif event.key() == QtCore.Qt.Key_4:
            self.go_to_prev_image()

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        img_name_match = re.search('(.+)/(.+).dm3$', imgPath)
        img_name_text = img_name_match.group(2)

        imgData, pxDims = dm3.ReadDm3File(imgPath)
        imsup.Image.px_dim_default = pxDims[0]
        imgData = np.abs(imgData)
        img = imsup.ImageExp(imgData.shape[0], imgData.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'],
                             num=imgNum, px_dim_sz=pxDims[0])
        # img.LoadAmpData(np.sqrt(imgData).astype(np.float32))
        img.LoadAmpData(imgData.astype(np.float32))
        img = rescale_image_buffer_to_window(img, const.ccWidgetDim)
        img.name = img_name_text
        # img.amPh.ph = np.copy(img.amPh.am)
        # ---
        # imsup.RemovePixelArtifacts(img, const.minPxThreshold, const.maxPxThreshold)
        # imsup.RemovePixelArtifacts(img, 0.7, 1.3)
        # img.UpdateBuffer()
        # ---
        imgList.append(img)

        imgNum += 1
        imgNumTextNew = imgNumText.replace(str(imgNum-1), str(imgNum))
        if imgNum == 10:
            imgNumTextNew = imgNumTextNew[1:]
        imgPath = RReplace(imgPath, imgNumText, imgNumTextNew, 1)
        imgNumText = imgNumTextNew

    imgList.UpdateAndRestrainLinks()
    return imgList[0]

# --------------------------------------------------------

def delete_n_prev_images(imgs, idx1, idx2):      # like del imgs[idx1:idx2]
    if imgs[idx1].prev is not None:
        imgs[idx1].prev.next = imgs[idx2]
    imgs[idx2].prev = imgs[idx1].prev
    del imgs[idx1:idx2]
    imgs.UpdateLinks()

# --------------------------------------------------------

def rescale_image_buffer_to_window(img, win_dim):
    zoom_factor = win_dim / img.width
    img_to_disp = tr.RescaleImageSki(img, zoom_factor)
    img.buffer = imsup.ComplexAmPhMatrix(img_to_disp.height, img_to_disp.width, img_to_disp.memType)
    img.buffer.am = np.copy(img_to_disp.amPh.am)
    img.buffer.ph = np.copy(img_to_disp.amPh.ph)
    return img

# --------------------------------------------------------

def cross_corr_images(img_list, n_div, frag_coords, df_min=0.0, df_max=-1.0, df_step=1.0):
    img_align_list = imsup.ImageList()
    img_list[0].shift = [0, 0]
    img_align_list.append(img_list[0])
    if df_max < 0.0:
        df_max = df_min + 1.0
        df_step = 2.0
    for img in img_list[1:]:
        mcf_best = cc.MaximizeMCFCore(img.prev, img, n_div, frag_coords,
                                      df_min, df_max, df_step, use_other_aberrs=False)
        new_shift = cc.GetShift(mcf_best)
        img.shift = [ sp + sn for sp, sn in zip(img.prev.shift, new_shift) ]
        # img.shift = list(np.array(img.shift) + np.array(new_shift))
        img_shifted = cc.shift_am_ph_image(img, img.shift)
        # img.defocus = mcf_best.defocus
        img_shifted.defocus = mcf_best.defocus
        img_align_list.append(img_shifted)
    return img_align_list

# --------------------------------------------------------

def zoom_fragment(img, coords):
    crop_img = imsup.crop_am_ph_roi(img, coords)
    crop_img = imsup.create_imgexp_from_img(crop_img)
    crop_img.MoveToCPU()

    crop_img.defocus = img.defocus
    crop_img = rescale_image_buffer_to_window(crop_img, const.ccWidgetDim)
    return crop_img

# --------------------------------------------------------

def modify_image(img, mod=list([0, 0]), is_shift=True):
    if is_shift:
        mod_img = cc.shift_am_ph_image(img, mod)
    else:
        mod_img = tr.RotateImageSki(img, mod[0])

    return mod_img

# --------------------------------------------------------

def norm_phase_to_pt(phase, pt):
    y, x = pt
    phase_norm = phase - phase[y, x]
    return phase_norm

# --------------------------------------------------------

def FindDirectionAngles(p1, p2):
    lpt = p1[:] if p1[0] < p2[0] else p2[:]     # left point
    rpt = p1[:] if p1[0] > p2[0] else p2[:]     # right point
    dx = np.abs(rpt[0] - lpt[0])
    dy = np.abs(rpt[1] - lpt[1])
    sign = 1 if rpt[1] < lpt[1] else -1
    projDir = 1         # projection on y axis
    if dx > dy:
        sign *= -1
        projDir = 0     # projection on x axis
    diff1 = dx if dx < dy else dy
    diff2 = dx if dx > dy else dy
    ang1 = np.arctan2(diff1, diff2)
    ang2 = np.pi / 2 - ang1
    ang1 *= sign
    ang2 *= (-sign)
    return ang1, ang2, projDir

# --------------------------------------------------------

def CalcTopLeftCoords(imgWidth, midCoords):
    topLeftCoords = [ mc + imgWidth // 2 for mc in midCoords ]
    return topLeftCoords

# --------------------------------------------------------

def CalcTopLeftCoordsForSetOfPoints(imgWidth, points):
    topLeftPoints = [ CalcTopLeftCoords(imgWidth, pt) for pt in points ]
    return topLeftPoints

# --------------------------------------------------------

def CalcRealTLCoords(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ int(dc * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealTLCoordsForSetOfPoints(imgWidth, points):
    realCoords = [ CalcRealTLCoords(imgWidth, pt) for pt in points ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoords(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    factor = imgWidth / dispWidth
    realCoords = [ int((dc - dispWidth // 2) * factor) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def CalcRealCoordsForSetOfPoints(imgWidth, points):
    realPoints = [ CalcRealCoords(imgWidth, pt) for pt in points ]
    return realPoints

# --------------------------------------------------------

def CalcRealTLCoordsForPaddedImage(imgWidth, dispCoords):
    dispWidth = const.ccWidgetDim
    padImgWidthReal = np.ceil(imgWidth / 512.0) * 512.0
    pad = (padImgWidthReal - imgWidth) / 2.0
    factor = padImgWidthReal / dispWidth
    # dispPad = pad / factor
    # realCoords = [ (dc - dispPad) * factor for dc in dispCoords ]
    realCoords = [ int(dc * factor - pad) for dc in dispCoords ]
    return realCoords

# --------------------------------------------------------

def real_to_disp_len(disp_len, img_len, r_len):
    factor = disp_len / img_len
    d_len = r_len * factor
    return d_len

# --------------------------------------------------------

# def CalcDispCoords(dispWidth, imgWidth, realCoords):
#     factor = dispWidth / imgWidth
#     dispCoords = [ (rc * factor) + const.ccWidgetDim // 2 for rc in realCoords ]
#     return dispCoords

# --------------------------------------------------------

def CalcDistance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

# --------------------------------------------------------

def CalcInnerAngle(a, b, c):
    alpha = np.arccos(np.abs((a*a + b*b - c*c) / (2*a*b)))
    return imsup.Degrees(alpha)

# --------------------------------------------------------

def CalcOuterAngle(p1, p2):
    dist = CalcDistance(p1, p2)
    betha = np.arcsin(np.abs(p1[0] - p2[0]) / dist)
    return imsup.Degrees(betha)

# --------------------------------------------------------

def CalcNewCoords(p1, newCenter):
    p2 = [ px - cx for px, cx in zip(p1, newCenter) ]
    return p2

# --------------------------------------------------------

def CalcRotAngle(p1, p2):
    z1 = np.complex(p1[0], p1[1])
    z2 = np.complex(p2[0], p2[1])
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)
    # rotAngle = np.abs(imsup.Degrees(phi2 - phi1))
    rotAngle = imsup.Degrees(phi2 - phi1)
    if np.abs(rotAngle) > 180:
        rotAngle = -np.sign(rotAngle) * (360 - np.abs(rotAngle))
    return rotAngle

# --------------------------------------------------------

def convert_points_to_tl_br(p1, p2):
    tl = list(np.amin([p1, p2], axis=0))
    br = list(np.amax([p1, p2], axis=0))
    return tl, br

# --------------------------------------------------------

def det_Imin_Imax_from_contrast(dI, def_max=256.0):
    dImin = dI // 2 + 1
    dImax = dI - dImin
    Imin = def_max // 2 - dImin
    Imax = def_max // 2 + dImax
    return Imin, Imax

# --------------------------------------------------------

def SwitchXY(xy):
    return [xy[1], xy[0]]

# --------------------------------------------------------

def RReplace(text, old, new, occurence):
    rest = text.rsplit(old, occurence)
    return new.join(rest)

# --------------------------------------------------------

def RunInLineWindow():
    app = QtWidgets.QApplication(sys.argv)
    ilWindow = InLineWidget()
    sys.exit(app.exec_())

# --------------------------------------------------------

def trace_contour(arr, xy):
    contour = []
    x, y = xy
    adj_sum = 1
    mid = np.ones(2)
    last_xy = [1, 1]
    print(xy)

    while adj_sum > 0:
        adj_arr = arr[x-1:x+2, y-1:y+2]
        adj_arr[last_xy[0], last_xy[1]] = 0
        adj_arr[1, 1] = 0
        adj_sum = np.sum(np.array(adj_arr))
        if adj_sum > 0:
            next_xy = [ idx[0] for idx in np.where(adj_arr == 1) ]
            last_xy = list(2 * mid - np.array(next_xy))
            next_xy = list(np.array(next_xy)-1 + np.array(xy))
            contour.append(next_xy)
            x, y = next_xy
            xy = [x, y]
            print(next_xy)

    print(len(contour))
    cont_arr = np.zeros(arr.shape)
    for idxs in contour:
        cont_arr[idxs[0], idxs[1]] = 1

    # cont_img = imsup.ImageExp(cont_arr.shape[0], cont_arr.shape[1])
    # cont_img.LoadAmpData(cont_arr)
    # imsup.DisplayAmpImage(cont_img)

    return contour

# --------------------------------------------------------

def find_contours(img):
    # arrow_dirs = np.zeros((img.height, img.width))
    ph_arr = np.copy(img.amPh.ph)
    ph_arr_scaled = imsup.ScaleImage(ph_arr, 0, 1)
    ph_arr_scaled[ph_arr_scaled < 0.98] = 0
    ph_arr_scaled[ph_arr_scaled >= 0.98] = 1
    # ph_arr_corr = imsup.ScaleImage(ph_arr_scaled, 0, 1)

    for i in range(100, img.height):
        for j in range(100, img.width):
            if ph_arr_scaled[i, j] == 1:
                print('Found one!')
                print(i, j)
                contour = trace_contour(ph_arr_scaled, [i, j])
                return contour