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
        self.pointSets = [[]] * self.n_imgs
        # self.pointSets = [[]]
        self.frag_coords = [0]
        self.show_lines = True
        self.show_labs = True
        self.show_grid = True
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

        if self.frag_coords[0] > 0:
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

        if dispAmp:
            self.image.buffer = np.copy(self.image.amPh.am)
            if logScale:
                buf = self.image.buffer
                buf[np.where(buf <= 0)] = 1e-5
                self.image.buffer = np.log(buf)
        elif dispPhs:
            self.image.buffer = np.copy(self.image.amPh.ph)
        else:
            if self.image.cos_phase is None:
                self.image.update_cos_phase()
            self.image.buffer = np.copy(self.image.cos_phase)

        if not update_bcg:
            buf_scaled = imsup.ScaleImage(self.image.buffer, 0.0, 255.0)
        else:
            buf_scaled = update_image_bright_cont_gamma(self.image.buffer, brg=bright, cnt=cont, gam=gamma)

        # final image with all properties set
        q_image = QtGui.QImage(buf_scaled.astype(np.uint8), self.image.width, self.image.height, QtGui.QImage.Format_Indexed8)

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

class TriangulateWidget(QtWidgets.QWidget):
    def __init__(self):
        super(TriangulateWidget, self).__init__()
        file_dialog = QtWidgets.QFileDialog()
        image_path = file_dialog.getOpenFileName()[0]
        if image_path == '':
            print('No images to read. Exiting...')
            exit()
        image = LoadImageSeriesFromFirstFile(image_path)
        image.name = 'img01'
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
        self.initUI()

    def initUI(self):
        self.plot_widget.canvas.setFixedHeight(250)

        prev_button = QtWidgets.QPushButton('Prev', self)
        next_button = QtWidgets.QPushButton('Next', self)

        prev_button.clicked.connect(self.go_to_prev_image)
        next_button.clicked.connect(self.go_to_next_image)

        lswap_button = QtWidgets.QPushButton('L-Swap', self)
        rswap_button = QtWidgets.QPushButton('R-Swap', self)

        lswap_button.clicked.connect(self.swap_left)
        rswap_button.clicked.connect(self.swap_right)

        flip_button = QtWidgets.QPushButton('Flip', self)

        name_it_button = QtWidgets.QPushButton('Name it!', self)
        self.name_input = QtWidgets.QLineEdit('img01', self)

        zoom_button = QtWidgets.QPushButton('Zoom N images', self)
        self.n_to_zoom_input = QtWidgets.QLineEdit(str(self.display.n_imgs), self)

        hbox_name = QtWidgets.QHBoxLayout()
        hbox_name.addWidget(name_it_button)
        hbox_name.addWidget(self.name_input)

        hbox_zoom = QtWidgets.QHBoxLayout()
        hbox_zoom.addWidget(zoom_button)
        hbox_zoom.addWidget(self.n_to_zoom_input)

        name_it_button.setFixedWidth(130)
        zoom_button.setFixedWidth(130)
        self.name_input.setFixedWidth(130)
        self.n_to_zoom_input.setFixedWidth(130)

        flip_button.clicked.connect(self.flip_image_h)
        name_it_button.clicked.connect(self.set_image_name)
        zoom_button.clicked.connect(self.zoom_n_fragments)

        export_button = QtWidgets.QPushButton('Export', self)
        export_all_button = QtWidgets.QPushButton('Export all', self)
        delete_button = QtWidgets.QPushButton('Delete', self)
        clear_button = QtWidgets.QPushButton('Clear', self)
        undo_button = QtWidgets.QPushButton('Undo', self)

        self.export_png_radio_button = QtWidgets.QRadioButton('PNG image', self)
        self.export_bin_radio_button = QtWidgets.QRadioButton('Binary', self)
        self.export_png_radio_button.setChecked(True)

        export_group = QtWidgets.QButtonGroup(self)
        export_group.addButton(self.export_png_radio_button)
        export_group.addButton(self.export_bin_radio_button)

        export_button.clicked.connect(self.export_image)
        export_all_button.clicked.connect(self.export_all)
        delete_button.clicked.connect(self.delete_image)
        clear_button.clicked.connect(self.clear_image)
        undo_button.clicked.connect(self.remove_last_point)

        self.left_button = QtWidgets.QPushButton(QtGui.QIcon('gui/left.png'), '', self)
        self.right_button = QtWidgets.QPushButton(QtGui.QIcon('gui/right.png'), '', self)
        self.up_button = QtWidgets.QPushButton(QtGui.QIcon('gui/up.png'), '', self)
        self.down_button = QtWidgets.QPushButton(QtGui.QIcon('gui/down.png'), '', self)
        self.px_shift_input = QtWidgets.QLineEdit('0', self)

        self.rot_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_right.png'), '', self)
        self.rot_counter_clockwise_button = QtWidgets.QPushButton(QtGui.QIcon('gui/rot_left.png'), '', self)
        self.rot_angle_input = QtWidgets.QLineEdit('0.0', self)

        self.px_shift_input.setFixedWidth(60)
        self.rot_angle_input.setFixedWidth(60)

        self.left_button.clicked.connect(self.move_left)
        self.right_button.clicked.connect(self.move_right)
        self.up_button.clicked.connect(self.move_up)
        self.down_button.clicked.connect(self.move_down)

        self.rot_counter_clockwise_button.clicked.connect(self.rot_left)
        self.rot_clockwise_button.clicked.connect(self.rot_right)

        self.left_button.setFixedWidth(60)
        self.right_button.setFixedWidth(60)
        self.up_button.setFixedWidth(60)
        self.down_button.setFixedWidth(60)
        self.rot_counter_clockwise_button.setFixedWidth(60)
        self.rot_clockwise_button.setFixedWidth(60)

        self.apply_button = QtWidgets.QPushButton('Apply changes', self)
        self.reset_button = QtWidgets.QPushButton('Reset', self)
        self.apply_button.clicked.connect(self.apply_changes)
        self.reset_button.clicked.connect(self.reset_changes)

        self.disable_manual_panel()

        self.manual_mode_checkbox = QtWidgets.QCheckBox('Manual mode', self)
        self.manual_mode_checkbox.setChecked(False)
        self.manual_mode_checkbox.clicked.connect(self.create_backup_image)

        grid_manual = QtWidgets.QGridLayout()
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

        self.show_lines_checkbox = QtWidgets.QCheckBox('Show lines', self)
        self.show_lines_checkbox.setChecked(True)
        self.show_lines_checkbox.toggled.connect(self.toggle_lines)

        self.show_labels_checkbox = QtWidgets.QCheckBox('Show labels', self)
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.toggled.connect(self.toggle_labels)

        self.log_scale_checkbox = QtWidgets.QCheckBox('Log scale', self)
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.toggled.connect(self.update_display)

        unwrap_button = QtWidgets.QPushButton('Unwrap', self)
        wrap_button = QtWidgets.QPushButton('Wrap', self)

        unwrap_button.clicked.connect(self.unwrap_img_phase)
        wrap_button.clicked.connect(self.wrap_img_phase)

        self.amp_radio_button = QtWidgets.QRadioButton('Amplitude', self)
        self.phs_radio_button = QtWidgets.QRadioButton('Phase', self)
        self.cos_phs_radio_button = QtWidgets.QRadioButton('Phase cosine', self)
        self.amp_radio_button.setChecked(True)

        self.amp_radio_button.toggled.connect(self.update_display)
        self.phs_radio_button.toggled.connect(self.update_display)
        self.cos_phs_radio_button.toggled.connect(self.update_display)

        amp_phs_group = QtWidgets.QButtonGroup(self)
        amp_phs_group.addButton(self.amp_radio_button)
        amp_phs_group.addButton(self.phs_radio_button)
        amp_phs_group.addButton(self.cos_phs_radio_button)

        self.gray_radio_button = QtWidgets.QRadioButton('Grayscale', self)
        self.color_radio_button = QtWidgets.QRadioButton('Color', self)
        self.gray_radio_button.setChecked(True)

        self.gray_radio_button.toggled.connect(self.update_display)
        self.color_radio_button.toggled.connect(self.update_display)

        color_group = QtWidgets.QButtonGroup(self)
        color_group.addButton(self.gray_radio_button)
        color_group.addButton(self.color_radio_button)

        hbox_unwrap_gray = QtWidgets.QHBoxLayout()
        hbox_unwrap_gray.addWidget(unwrap_button)
        hbox_unwrap_gray.addWidget(self.gray_radio_button)

        hbox_wrap_color = QtWidgets.QHBoxLayout()
        hbox_wrap_color.addWidget(wrap_button)
        hbox_wrap_color.addWidget(self.color_radio_button)

        self.btn_grid = ButtonGrid(1, fun=self.get_clicked_coords)

        # n_imgs = len(imsup.CreateImageListFromFirstImage(self.display.image))
        self.n_to_cc_input = QtWidgets.QLineEdit(str(self.display.n_imgs), self)
        cross_corr_w_prev_button = QtWidgets.QPushButton('Cross-corr. with prev.', self)
        cross_corr_n_images_button = QtWidgets.QPushButton('Cross-corr. N images', self)
        cross_corr_n_images_button.setFixedWidth(130)
        shift_button = QtWidgets.QPushButton('Shift', self)
        warp_button = QtWidgets.QPushButton('Warp', self)

        mesh_up_button = QtWidgets.QPushButton('Mesh+', self)
        mesh_down_button = QtWidgets.QPushButton('Mesh-', self)

        cross_corr_w_prev_button.clicked.connect(self.cross_corr_with_prev)
        cross_corr_n_images_button.clicked.connect(self.cross_corr_n_images)
        shift_button.clicked.connect(self.align_shift)
        warp_button.clicked.connect(partial(self.warp_image, False))
        mesh_up_button.clicked.connect(self.btn_grid.increase_mesh)
        mesh_down_button.clicked.connect(self.btn_grid.decrease_mesh)

        self.det_df_checkbox = QtWidgets.QCheckBox('Determine df values', self)
        self.det_df_checkbox.setChecked(False)
        self.det_df_checkbox.toggled.connect(self.manage_df_inputs)

        self.df_min_label = QtWidgets.QLabel('df min [nm]', self)
        df_max_label = QtWidgets.QLabel('df max [nm]', self)
        df_step_label = QtWidgets.QLabel('delta df [nm]', self)

        self.df_min_label.setAlignment(QtCore.Qt.AlignCenter)
        df_max_label.setAlignment(QtCore.Qt.AlignCenter)
        df_step_label.setAlignment(QtCore.Qt.AlignCenter)

        self.df_min_input = QtWidgets.QLineEdit('0.0', self)
        self.df_max_input = QtWidgets.QLineEdit('0.0', self)
        self.df_step_input = QtWidgets.QLineEdit('0.0', self)
        self.disable_df_inputs()

        fname_label = QtWidgets.QLabel('File name', self)
        self.fname_input = QtWidgets.QLineEdit(self.display.image.name, self)
        self.fname_input.setFixedWidth(150)

        in_focus_label = QtWidgets.QLabel('In-focus', self)
        self.in_focus_input = QtWidgets.QLineEdit('1', self)

        n_to_ewr_label = QtWidgets.QLabel('Num. of images to use', self)
        self.n_to_ewr_input = QtWidgets.QLineEdit(str(self.display.n_imgs), self)

        n_iters_label = QtWidgets.QLabel('Num. of iterations', self)
        self.n_iters_input = QtWidgets.QLineEdit('10', self)

        run_ewr_button = QtWidgets.QPushButton('Run reconstruction', self)
        run_ewr_button.clicked.connect(self.run_ewr)

        self.use_aberrs_checkbox = QtWidgets.QCheckBox('Use aberrations', self)
        self.use_aberrs_checkbox.setChecked(False)

        self.det_abs_df_checkbox = QtWidgets.QCheckBox('Det. abs. defoc. values', self)
        self.det_abs_df_checkbox.setChecked(True)

        aperture_label = QtWidgets.QLabel('Aperture radius [px]', self)
        self.aperture_input = QtWidgets.QLineEdit(str(const.aperture), self)

        hann_win_label = QtWidgets.QLabel('Hann window [px]', self)
        self.hann_win_input = QtWidgets.QLineEdit(str(const.hann_win), self)

        sum_button = QtWidgets.QPushButton('Sum', self)
        diff_button = QtWidgets.QPushButton('Diff', self)

        sum_button.clicked.connect(self.calc_phs_sum)
        diff_button.clicked.connect(self.calc_phs_diff)

        amp_factor_label = QtWidgets.QLabel('Amp. factor', self)
        self.amp_factor_input = QtWidgets.QLineEdit('2.0', self)

        amplify_button = QtWidgets.QPushButton('Amplify', self)
        amplify_button.clicked.connect(self.amplify_phase)

        int_width_label = QtWidgets.QLabel('Profile width [px]', self)
        self.int_width_input = QtWidgets.QLineEdit('1', self)

        plot_button = QtWidgets.QPushButton('Plot profile', self)
        plot_button.clicked.connect(self.plot_profile)

        norm_phase_button = QtWidgets.QPushButton('Normalize phase', self)
        norm_phase_button.clicked.connect(self.norm_phase)

        grid_nav = QtWidgets.QGridLayout()
        grid_nav.addWidget(prev_button, 0, 0)
        grid_nav.addWidget(next_button, 0, 1)
        grid_nav.addWidget(lswap_button, 1, 0)
        grid_nav.addWidget(rswap_button, 1, 1)
        grid_nav.addWidget(flip_button, 2, 0)
        grid_nav.addWidget(clear_button, 2, 1)
        grid_nav.addWidget(delete_button, 3, 1)
        grid_nav.addLayout(hbox_zoom, 3, 0)
        grid_nav.addLayout(hbox_name, 4, 0)
        grid_nav.addWidget(undo_button, 4, 1)

        grid_disp = QtWidgets.QGridLayout()
        grid_disp.setColumnStretch(0, 0)
        grid_disp.setColumnStretch(1, 0)
        grid_disp.setColumnStretch(2, 0)
        grid_disp.addWidget(self.show_lines_checkbox, 1, 0)
        grid_disp.addWidget(self.show_labels_checkbox, 2, 0)
        grid_disp.addWidget(self.log_scale_checkbox, 3, 0)
        grid_disp.addWidget(self.amp_radio_button, 1, 1)
        grid_disp.addWidget(self.phs_radio_button, 2, 1)
        grid_disp.addWidget(self.cos_phs_radio_button, 3, 1)
        grid_disp.addLayout(hbox_unwrap_gray, 1, 2)
        grid_disp.addLayout(hbox_wrap_color, 2, 2)
        grid_disp.addWidget(norm_phase_button, 3, 2)
        grid_disp.addWidget(fname_label, 0, 4)
        grid_disp.addWidget(self.fname_input, 1, 4)
        grid_disp.addWidget(export_button, 2, 4)
        grid_disp.addWidget(export_all_button, 3, 4)
        grid_disp.addWidget(self.export_png_radio_button, 1, 5)
        grid_disp.addWidget(self.export_bin_radio_button, 2, 5)

        mesh_down_button.setFixedWidth(mesh_up_button.width())

        grid_cross_corr = QtWidgets.QGridLayout()
        grid_cross_corr.addLayout(self.btn_grid, 0, 0, 4, 1)
        grid_cross_corr.addWidget(mesh_up_button, 0, 1)
        grid_cross_corr.addWidget(mesh_down_button, 0, 2)
        grid_cross_corr.addWidget(self.det_df_checkbox, 0, 3)
        grid_cross_corr.addWidget(self.df_min_label, 1, 3)
        grid_cross_corr.addWidget(df_max_label, 2, 3)
        grid_cross_corr.addWidget(df_step_label, 3, 3)
        grid_cross_corr.addWidget(self.df_min_input, 1, 4)
        grid_cross_corr.addWidget(self.df_max_input, 2, 4)
        grid_cross_corr.addWidget(self.df_step_input, 3, 4)
        grid_cross_corr.addWidget(cross_corr_w_prev_button, 1, 1, 1, 2)
        grid_cross_corr.addWidget(cross_corr_n_images_button, 2, 1)
        grid_cross_corr.addWidget(self.n_to_cc_input, 2, 2)
        grid_cross_corr.addWidget(shift_button, 3, 1)
        grid_cross_corr.addWidget(warp_button, 3, 2)

        grid_ewr = QtWidgets.QGridLayout()
        grid_ewr.addWidget(in_focus_label, 0, 0)
        grid_ewr.addWidget(self.in_focus_input, 1, 0)
        grid_ewr.addWidget(n_to_ewr_label, 2, 0)
        grid_ewr.addWidget(self.n_to_ewr_input, 3, 0)
        grid_ewr.addWidget(aperture_label, 0, 1)
        grid_ewr.addWidget(self.aperture_input, 1, 1)
        grid_ewr.addWidget(hann_win_label, 0, 2)
        grid_ewr.addWidget(self.hann_win_input, 1, 2)
        grid_ewr.addWidget(amp_factor_label, 0, 3)
        grid_ewr.addWidget(self.amp_factor_input, 1, 3)
        grid_ewr.addWidget(n_iters_label, 2, 1)
        grid_ewr.addWidget(self.n_iters_input, 3, 1)
        grid_ewr.addWidget(self.use_aberrs_checkbox, 4, 0)
        grid_ewr.addWidget(self.det_abs_df_checkbox, 5, 0)
        grid_ewr.addWidget(run_ewr_button, 4, 1)
        grid_ewr.addWidget(amplify_button, 2, 3)
        grid_ewr.addWidget(sum_button, 2, 2)
        grid_ewr.addWidget(diff_button, 3, 2)
        grid_ewr.addWidget(int_width_label, 3, 3)
        grid_ewr.addWidget(self.int_width_input, 4, 3)
        grid_ewr.addWidget(plot_button, 4, 2)

        # ----

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
        self.gamma_slider.setRange(10, 190)     # !!!
        self.gamma_slider.setValue(100)

        self.bright_slider.valueChanged.connect(self.disp_bright_value)
        self.cont_slider.valueChanged.connect(self.disp_cont_value)
        self.gamma_slider.valueChanged.connect(self.disp_gamma_value)

        self.bright_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.cont_slider.sliderReleased.connect(self.update_display_and_bcg)
        self.gamma_slider.sliderReleased.connect(self.update_display_and_bcg)

        self.bright_input = QtWidgets.QLineEdit('0', self)
        self.cont_input = QtWidgets.QLineEdit('255', self)
        self.gamma_input = QtWidgets.QLineEdit('1.0', self)

        self.bright_input.returnPressed.connect(self.update_display_and_bcg)
        self.cont_input.returnPressed.connect(self.update_display_and_bcg)
        self.gamma_input.returnPressed.connect(self.update_display_and_bcg)

        reset_bright_button = QtWidgets.QPushButton('Reset B', self)
        reset_cont_button = QtWidgets.QPushButton('Reset C', self)
        reset_gamma_button = QtWidgets.QPushButton('Reset G', self)

        reset_bright_button.clicked.connect(self.reset_bright)
        reset_cont_button.clicked.connect(self.reset_cont)
        reset_gamma_button.clicked.connect(self.reset_gamma)

        grid_sliders = QtWidgets.QGridLayout()
        grid_sliders.addWidget(bright_label, 0, 1)
        grid_sliders.addWidget(self.bright_slider, 1, 0)
        grid_sliders.addWidget(self.bright_input, 1, 1)
        grid_sliders.addWidget(reset_bright_button, 1, 2)
        grid_sliders.addWidget(cont_label, 2, 1)
        grid_sliders.addWidget(self.cont_slider, 3, 0)
        grid_sliders.addWidget(self.cont_input, 3, 1)
        grid_sliders.addWidget(reset_cont_button, 3, 2)
        grid_sliders.addWidget(gamma_label, 4, 1)
        grid_sliders.addWidget(self.gamma_slider, 5, 0)
        grid_sliders.addWidget(self.gamma_input, 5, 1)
        grid_sliders.addWidget(reset_gamma_button, 5, 2)

        # ----

        vbox_panel = QtWidgets.QVBoxLayout()
        vbox_panel.addLayout(grid_nav)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_disp)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_manual)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_cross_corr)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_ewr)
        vbox_panel.addStretch(1)
        vbox_panel.addLayout(grid_sliders)

        hbox_panel = QtWidgets.QHBoxLayout()
        hbox_panel.addWidget(self.display)
        hbox_panel.addLayout(vbox_panel)

        vbox_main = QtWidgets.QVBoxLayout()
        vbox_main.addLayout(hbox_panel)
        vbox_main.addWidget(self.plot_widget)
        self.setLayout(vbox_main)

        self.move(250, 5)
        self.setWindowTitle('PyInLine')
        self.setWindowIcon(QtGui.QIcon('gui/world.png'))
        self.show()
        self.setFixedSize(self.width(), self.height())  # disable window resizing

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

    def go_to_image(self, new_idx):
        # simplify this!
        # is_amp_checked = self.amp_radio_button.isChecked()
        # is_phs_checked = self.phs_radio_button.isChecked()
        # is_log_scale_checked = self.log_scale_checkbox.isChecked()
        # is_show_labels_checked = self.show_labels_checkbox.isChecked()
        # is_color_checked = self.color_radio_button.isChecked()
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
        self.display.image = imgs[new_idx]
        # self.display.change_image(new_idx, dispAmp=is_amp_checked, dispPhs=is_phs_checked,
        #                           logScale=is_log_scale_checked, dispLabs=is_show_labels_checked, color=is_color_checked)
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
            fname_ext = '.png'
            log = True if self.log_scale_checkbox.isChecked() else False
            color = True if self.color_radio_button.isChecked() else False

            if is_amp_checked:
                imsup.SaveAmpImage(curr_img, '{0}.png'.format(fname), log, color)
            elif is_phs_checked:
                imsup.SavePhaseImage(curr_img, '{0}.png'.format(fname), log, color)
            else:
                phs_tmp = np.copy(curr_img.amPh.ph)
                curr_img.amPh.ph = np.cos(phs_tmp)
                imsup.SavePhaseImage(curr_img, '{0}.png'.format(fname), log, color)
                curr_img.amPh.ph = np.copy(phs_tmp)
            print('Saved image as "{0}.png"'.format(fname))

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
        tmp_img_list = imsup.CreateImageListFromFirstImage(first_img)

        if curr_img.prev is not None:
            curr_img.prev.next = None
            self.go_to_prev_image()
        else:
            curr_img.next.prev = None
            self.go_to_next_image()
            if curr_idx == 0:
                self.display.image.numInSeries = 1

        del tmp_img_list[curr_idx]
        del self.display.pointSets[curr_idx]
        tmp_img_list.UpdateLinks()
        del curr_img

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
        img_list = imsup.CreateImageListFromFirstImage(curr_img)
        img_list2 = img_list[:n_to_zoom]

        for img, n in zip(img_list2, range(n_to_zoom, 2*n_to_zoom)):
            frag = zoom_fragment(img, real_sq_coords)
            img_list.insert(n, frag)
            self.display.pointSets.insert(n, [])

        img_list.UpdateLinks()
        self.go_to_last_image()
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

    def cross_corr_with_prev(self):
        curr_img = self.display.image
        if curr_img.prev is None:
            print('There is no reference image!')
            return
        img_list_to_cc = imsup.CreateImageListFromImage(curr_img.prev, 2)
        img_aligned = self.cross_corr_core(img_list_to_cc)[0]
        self.insert_img_after_curr(img_aligned)
        self.go_to_next_image()

    def cross_corr_n_images(self):
        n_to_cc = int(self.n_to_cc_input.text())
        curr_img = self.display.image
        first_img = imsup.GetFirstImage(curr_img)
        all_img_list = imsup.CreateImageListFromFirstImage(first_img)
        n_imgs = len(all_img_list)
        # n_point_sets = len(self.display.pointSets)
        # n_diff = n_imgs - n_point_sets
        # self.display.pointSets.append([] * n_diff)
        if (curr_img.numInSeries - 1) + n_to_cc > n_imgs:
            n_to_cc = n_imgs - (curr_img.numInSeries - 1)
        img_list_to_cc = imsup.CreateImageListFromImage(curr_img, n_to_cc)
        img_align_list = self.cross_corr_core(img_list_to_cc)

        ref_img = imsup.copy_am_ph_image(curr_img)
        self.insert_img_last(ref_img)
        for img in img_align_list:
            self.insert_img_last(img)
        self.go_to_last_image()
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
        n_to_ewr = int(self.n_to_ewr_input.text())
        n_iters = int(self.n_iters_input.text())
        curr_img = self.display.image
        first_img = imsup.GetFirstImage(curr_img)
        all_imgs_list = imsup.CreateImageListFromFirstImage(first_img)

        n_all_imgs = len(all_imgs_list)
        if (curr_img.numInSeries - 1) + n_to_ewr > n_all_imgs:
            n_to_ewr = n_all_imgs - (curr_img.numInSeries - 1)

        imgs_to_iwfr = imsup.CreateImageListFromImage(curr_img, n_to_ewr)

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

# --------------------------------------------------------

def LoadImageSeriesFromFirstFile(imgPath):
    imgList = imsup.ImageList()
    imgNumMatch = re.search('([0-9]+).dm3', imgPath)
    imgNumText = imgNumMatch.group(1)
    imgNum = int(imgNumText)

    while path.isfile(imgPath):
        print('Reading file "' + imgPath + '"')
        imgData, pxDims = dm3.ReadDm3File(imgPath)
        imsup.Image.px_dim_default = pxDims[0]
        imgData = np.abs(imgData)
        img = imsup.ImageExp(imgData.shape[0], imgData.shape[1], imsup.Image.cmp['CAP'], imsup.Image.mem['CPU'],
                             num=imgNum, px_dim_sz=pxDims[0])
        # img.LoadAmpData(np.sqrt(imgData).astype(np.float32))
        img.LoadAmpData(imgData.astype(np.float32))
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

    imgList.UpdateLinks()
    return imgList[0]

# --------------------------------------------------------

def cross_corr_images(img_list, n_div, frag_coords, df_min=0.0, df_max=-1.0, df_step=1.0):
    img_align_list = imsup.ImageList()
    img_list[0].shift = [0, 0]
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

    orig_width = img.width
    crop_width = np.abs(coords[2] - coords[0])
    zoom_factor = orig_width / crop_width
    zoom_img = tr.RescaleImageSki(crop_img, zoom_factor)
    zoom_img.px_dim *= zoom_factor
    zoom_img.defocus = img.defocus
    # self.insert_img_after_curr(zoom_img)
    return zoom_img

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

def CalcDispCoords(dispWidth, imgWidth, realCoords):
    factor = dispWidth / imgWidth
    dispCoords = [ (rc * factor) + const.ccWidgetDim // 2 for rc in realCoords ]
    return dispCoords

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

def RunTriangulationWindow():
    app = QtWidgets.QApplication(sys.argv)
    trWindow = TriangulateWidget()
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