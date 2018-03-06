import os
import sys
import dicom
from glob import glob 
import numpy as np
import scipy.ndimage
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Non_MA:
    """
    Background image to synthesize    
    """
    def __init__(self):
        self.old_path = None
    
    def set_img(self, path):
        if path != self.old_path:
            self.origin_non_MA = dicom.read_file(path).pixel_array.astype("float32")
            self.old_path = path

    def insert_metal(self, metal, y, x, metal_r=None, y2=None, x2=None):
        """
        Metal insert on background image
        metal, y, x : insert one metal on y, x index
        metal_r, y2, x2 : insert another metal
        """
        m_y, m_x = metal.shape
        inserted_metal = self.origin_non_MA.copy()
        inserted_metal[y:y + m_y, x:x + m_x] += metal

        if metal_r is not None:
            m_y, m_x = metal_r.shape
            inserted_metal[y2:y2 + m_y, x2:x2 + m_x] += metal_r
        
        return inserted_metal.clip(np.amin(inserted_metal), 4095)


class MA:
    """
    Extract metal image on MA image
    """
    def __init__(self):
        self.old_path = None
        self.metal_cnt = 1

    def _get_metal_range(self, metal_img):
        """
        Return square with metal parts
        """
        metal_range = np.where(metal_img != 0)
        y_min, y_max = min(metal_range[0]), max(metal_range[0])
        x_min, x_max = min(metal_range[1]), max(metal_range[1])
        return metal_img[y_min:y_max, x_min:x_max]

    def set_img(self, path, metal_cnt):
        if path != self.old_path:
            self.metal_cnt = metal_cnt
            self.old_path = path
            self.origin_MA = dicom.read_file(path).pixel_array.astype("float32")
            flaged_MA = (self.origin_MA >= 4090).astype(int)
            # got no metal
            if np.sum(flaged_MA) == 0 :
                self._metal = np.array([0.])
                return

            cliped_MA = self.origin_MA * flaged_MA

            if metal_cnt == 1:
                self._metal = self._get_metal_range(cliped_MA)
            elif metal_cnt == 2:
                # It could be occurs error, when metal on middle
                self._metal = self._get_metal_range(cliped_MA[:, 0:cliped_MA.shape[0]//2]) 
                self._metal_r = self._get_metal_range(cliped_MA[:, cliped_MA.shape[0]//2: cliped_MA.shape[0]]) 

    def _get_metal(self, metal, zoom, angle):
        if angle > 0:
            metal = scipy.ndimage.rotate(metal, angle)
        if zoom != 1:
            metal = scipy.ndimage.zoom(metal, zoom, order=0)
        return metal

    def get_metal(self, zoom=1, angle=0, zoom2=1, angle2=0):       
        if self.metal_cnt == 1:
            return self._get_metal(self._metal, zoom, angle)
        else:
            return self._get_metal(self._metal, zoom, angle), self._get_metal(self._metal_r, zoom2, angle2)


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        if os.path.exists(os.getcwd() + "\\inserted") is False:
            os.mkdir(os.getcwd() + "\\inserted")

        self.data_path = os.getcwd() + "\\inserted\\"
        files = [os.path.basename(x)[:-4].split("_") for x in glob(self.data_path + "*")]
        self.file_dict = {int(f[0] + f[1]) : {int(f[2] + f[3]) : int(f[4])} for f in files}
        
        self._inserted_metal = None
        self.non_MA = Non_MA()
        self.metal = MA()
        
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 768, 768)
        self.setWindowTitle("Metal Insertion v0.1")

        self.MA_path = QLineEdit()
        self.non_MA_path = QLineEdit()        
        self.location_edit = QLineEdit()          
        self.zoom_edit = QLineEdit()
        self.angle_edit = QLineEdit()
        
        self.location_edit2 = QLineEdit()          
        self.zoom_edit2 = QLineEdit()
        self.angle_edit2 = QLineEdit()        
        self.metal_cnt = QLineEdit()

        self.pushButton = QPushButton("Set Image")  
        self.pushButton.clicked.connect(self.input_imgs)
        self.saveButton = QPushButton("Save Image")
        self.saveButton.clicked.connect(self.save_img)

        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        upLayout = QVBoxLayout()
        upLayout.addWidget(self.canvas)
        
        downLayout = QVBoxLayout()
        downLayout.addWidget(QLabel("MA Path: "))
        downLayout.addWidget(self.MA_path)
        downLayout.addWidget(QLabel("NON MA Path: "))
        downLayout.addWidget(self.non_MA_path)
        downLayout1 = QHBoxLayout()
        downLayout1.addWidget(QLabel("Y X(0 ~ 512) : "))
        downLayout1.addWidget(self.location_edit)
        downLayout1.addWidget(QLabel("Zoom(real number): "))
        downLayout1.addWidget(self.zoom_edit)
        downLayout1.addWidget(QLabel("angle(0 ~ 360) : "))
        downLayout1.addWidget(self.angle_edit)
        downLayout1.addWidget(self.pushButton)
        downLayout1.addWidget(self.saveButton)
        downLayout2 = QHBoxLayout()
        downLayout2.addWidget(QLabel("Y2 X2(0 ~ 512) : "))
        downLayout2.addWidget(self.location_edit2)
        downLayout2.addWidget(QLabel("Zoom2(real number): "))
        downLayout2.addWidget(self.zoom_edit2)
        downLayout2.addWidget(QLabel("angle2(0 ~ 360) : "))
        downLayout2.addWidget(self.angle_edit2)
        downLayout2.addWidget(QLabel("Metal Count : "))
        downLayout2.addWidget(self.metal_cnt)
        downLayout.addStretch(1)

        layout = QVBoxLayout()        
        layout.addLayout(upLayout)
        layout.addLayout(downLayout)
        layout.addLayout(downLayout1)
        layout.addLayout(downLayout2)
        layout.setStretchFactor(upLayout, 1)
        layout.setStretchFactor(downLayout, 0)

        self.setLayout(layout)

    def _get_path(self):
        non_MA_path = self.non_MA_path.text()
        MA_path = self.MA_path.text()        
        print("Params non MA : ", non_MA_path)
        print("Params_____MA : ", MA_path)
        return non_MA_path, MA_path

    def _get_input_params(self):
        locate = (0, 0) if len(self.location_edit.text()) == 0 else [int(i) for i in self.location_edit.text().split()]
        zoom = 1.0 if len(self.zoom_edit.text()) == 0 else float(self.zoom_edit.text())
        angle = 0 if len(self.angle_edit.text()) == 0 else int(self.angle_edit.text())
        metal_cnt = 1 if len(self.metal_cnt.text()) == 0 else int(self.metal_cnt.text())
        print("Params_______ : ", locate, zoom, angle, metal_cnt)
        return locate, zoom, angle, metal_cnt

    def _get_input_params2(self):
        locate2 = (0, 0) if len(self.location_edit2.text()) == 0 else [int(i) for i in self.location_edit2.text().split()]
        zoom2 = 1.0 if len(self.zoom_edit2.text()) == 0 else float(self.zoom_edit2.text())
        angle2 = 0 if len(self.angle_edit2.text()) == 0 else int(self.angle_edit2.text())
        print("Params_______ : ", locate2, zoom2, angle2)
        return locate2, zoom2, angle2
        
    def input_imgs(self):
        non_MA_path, MA_path = self._get_path()
        """
        # For Test
        MA_path     =   r"C:\Users\zsef1\OneDrive\RPLab\MAR\metal_insert_tool\15369989_0096.DCM"
        non_MA_path =   r"C:\Users\zsef1\OneDrive\RPLab\MAR\metal_insert_tool\15858650_0030.DCM"
        """
        if self.non_MA is None or self.MA_path is None:
            return
        if os.path.exists(non_MA_path) is False or os.path.exists(MA_path) is False:
            return

        l, z, a, metal_cnt = self._get_input_params()
        self.metal.set_img(MA_path, metal_cnt)
        self.non_MA.set_img(non_MA_path)

        if metal_cnt == 1:
            metal = self.metal.get_metal(z, a)
            self._inserted_metal = self.non_MA.insert_metal(metal, l[0], l[1])
        else:
            l2, z2, a2 = self._get_input_params2()
            metal, metal2 = self.metal.get_metal(z, a, z2, a2)
            self._inserted_metal = self.non_MA.insert_metal(metal, l[0], l[1], metal2, l2[0], l2[1])

        self.ax.imshow(self._inserted_metal, cmap='gray')
        self.canvas.draw()

    def _set_file_cnt(self, non_ma_num, ma_num):
        non_ma_num = int(''.join(non_ma_num))
        ma_num = int(''.join(ma_num))
        if non_ma_num not in self.file_dict:
            self.file_dict[non_ma_num] = {ma_num:0}
        else:
            if ma_num not in self.file_dict[non_ma_num]:
                self.file_dict[non_ma_num][ma_num] = 0
            else:
                self.file_dict[non_ma_num][ma_num] += 1
        return self.file_dict[non_ma_num][ma_num]

    def save_img(self):
        non_ma_num = self.non_MA_path.text().split("\\")[-1][:-4].split("_")
        ma_num = self.MA_path.text().split("\\")[-1][:-4].split("_")

        path_cnt = self._set_file_cnt(non_ma_num, ma_num)
        path = "%s\\inserted\\%s_%s_%s_%s_%d"%(os.getcwd(), *non_ma_num, *ma_num, path_cnt)
        print("Save Img : ", path)
        np.save(path+".npy", self._inserted_metal)
        self.fig.savefig(path+".png")
        

if __name__ == "__main__":    
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()