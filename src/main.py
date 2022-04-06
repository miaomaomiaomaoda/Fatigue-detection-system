import sys
import dlib
import imutils
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QMessageBox, QFileDialog
from cv2 import cv2
from pygame import mixer
from src.UI import Ui_MainWindow
from src.utils import *
import winsound


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # 属性
        self.adjust_camera_Thread = AdjustCamera_Thread()
        self.start_Thread = Start_Thread()
        self.setupUi(self)

        # 连接槽
        self.Cam_Select.currentIndexChanged.connect(self.change_Cam_Select)
        self.Button_OpenVideo.clicked.connect(self.open_Video)
        self.Button_Start.clicked.connect(self.start)
        self.Button_End.clicked.connect(self.end)
        self.Button_AdjustCamera_Location.clicked.connect(self.adjust_camera_location)
        self.offDuty_Check.clicked.connect(self.change_OffDuty_Check_Status)
        self.offDuty_Time.valueChanged.connect(self.change_OffDuty_Value)
        self.video.clicked.connect(self.set_open_video)
        self.cam.clicked.connect(self.set_open_video)
        self.show_eye.clicked.connect(self.set_show_setting)
        self.show_head.clicked.connect(self.set_show_setting)
        self.show_mouth.clicked.connect(self.set_show_setting)
        self.show_key_point.clicked.connect(self.set_show_setting)

        self.start_Thread.msg.connect(self.show_Message)
        self.start_Thread.picture.connect(self.show_Image)
        self.start_Thread.window.connect(self.pop_window)
        self.adjust_camera_Thread.picture.connect(self.show_Image)
        self.adjust_camera_Thread.msg.connect(self.show_Message)
        self.adjust_camera_Thread.window.connect(self.pop_window)

    def set_show_setting(self):
        isChecked = self.sender().isChecked()
        if self.sender() == self.show_eye:
            self.start_Thread.set_show_eye(isChecked)
        elif self.sender() == self.show_mouth:
            self.start_Thread.set_show_mouth(isChecked)
        elif self.sender() == self.show_head:
            self.start_Thread.set_show_Head(isChecked)
        else:
            self.start_Thread.set_show_key_point(isChecked)

    def set_open_video(self):
        if self.video.isChecked():
            self.start_Thread.set_open_video(True)
        else:
            self.start_Thread.set_open_video(False)

    def change_OffDuty_Check_Status(self):
        self.start_Thread.change_OffDuty_Check_Status(self.offDuty_Check.isChecked())

    def change_OffDuty_Value(self):
        self.start_Thread.change_OffDuty_Value(self.offDuty_Time.value())

    def start(self):
        self.start_Thread.start()

    def adjust_camera_location(self):
        self.adjust_camera_Thread.start()

    def end(self):
        self.adjust_camera_Thread.close()
        self.start_Thread.close()

    def change_Cam_Select(self):
        self.adjust_camera_Thread.change_cam_select(self.Cam_Select.currentIndex())
        self.start_Thread.change_cam_select(self.Cam_Select.currentIndex())
        self.output_Window.append("切换摄像头" + str(self.Cam_Select.currentIndex()))

    def open_Video(self):
        filePath = QFileDialog.getOpenFileName(self, "打开视频文件", "", 'Video files(*.mp4)')
        self.output_Window.append("视频文件" + filePath[0] + "加载成功")
        self.start_Thread.set_filePath(filePath[0])
        self.video.setChecked(True)

    def show_Message(self, msg):
        self.output_Window.append(msg)

    def show_Image(self, image):
        height = image.shape[0]
        width = image.shape[1]
        frame = QImage(image, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def pop_window(self, info):
        QMessageBox.warning(self, "提示", info, QMessageBox.Yes)

    def exit(self):
        if self.cap is not None:
            self.cap.release()
        sys.exit(app.exec_())


class AdjustCamera_Thread(QThread):
    picture = pyqtSignal(object)
    msg = pyqtSignal(str)
    window = pyqtSignal(str)

    def __init__(self):
        super(AdjustCamera_Thread, self).__init__()
        self.predictor = None
        self.detector = None
        self.cap = None
        self.camSelect = 0
        self.isClose = False
        self.load_Model()

    def load_Model(self):
        # 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
        print("[INFO] loading facial landmark predictor...")
        # 使用dlib.get_frontal_face_detector() 获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 使用dlib.shape_predictor获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.msg.emit("脸部特征检测模型加载成功")

    def change_cam_select(self, camSelect):
        self.camSelect = camSelect

    def close(self):
        self.isClose = True

    def run(self):
        self.isClose = False
        self.window.emit("请调整摄像头位置，使人脸位于显示框内。调整后请按关闭结束")
        self.cap = cv2.VideoCapture(self.camSelect, cv2.CAP_DSHOW)
        while True:
            ret, frame = self.cap.read()
            frame = imutils.resize(frame, width=720)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 自适应直方图均衡
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # 使用detector(gray, 0) 进行脸部位置检测
            rects = self.detector(gray, 0)
            # 面部特征检测
            for rect in rects:
                # 使用predictor(gray, rect)获得脸部特征位置的信息
                shape = self.predictor(gray, rect)
                # 将脸部特征信息转换为数组array的格式
                shape = face_utils.shape_to_np(shape)
                # 获取头部姿态
                reprojectdst, euler_angle = get_head_pose(shape)
                # 取pitch（har）、yaw、roll旋转角度
                pitch = euler_angle[0, 0]
                yaw = euler_angle[1, 0]
                roll = euler_angle[2, 0]
                # 绘制正方体12轴
                for start, end in line_pairs:
                    start_point = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
                    end_point = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
                # 实时显示计算结果
                cv2.putText(frame, "pitch: {:5.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2)
                cv2.putText(frame, "yaw: {:5.2f}".format(yaw), (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "roll: {:5.2f}".format(roll), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)

            self.picture.emit(frame)
            if self.isClose:
                break
        self.cap.release()
        self.window.emit("摄像头位置调整结束")


class Start_Thread(QThread):
    picture = pyqtSignal(object)
    msg = pyqtSignal(str)
    window = pyqtSignal(str)

    def __init__(self):
        super(Start_Thread, self).__init__()
        self.offDutyTime = 0
        self.predictor = None
        self.detector = None
        self.filePath = None
        self.cap = None
        self.camSelect = 0
        self.isClose = False
        self.isOffDutyCheck = False
        self.isOpenVideo = False
        self.isShowEye = True
        self.isShowMouth = True
        self.isShowHead = False
        self.isShowKeyPoint = False
        # 加载模型
        self.load_Model()

    def load_Model(self):
        # 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
        print("[INFO] loading facial landmark predictor...")
        # 使用dlib.get_frontal_face_detector() 获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 使用dlib.shape_predictor获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.msg.emit("脸部特征检测模型加载成功")

    def set_show_eye(self, isShowEye):
        self.isShowEye = isShowEye

    def set_show_mouth(self, isShowMouth):
        self.isShowMouth = isShowMouth

    def set_show_Head(self, isShowHead):
        self.isShowHead = isShowHead

    def set_show_key_point(self, isShowKeyPoint):
        self.isShowKeyPoint = isShowKeyPoint

    def change_OffDuty_Check_Status(self, isOffDutyCheck):
        self.isOffDutyCheck = isOffDutyCheck

    def change_OffDuty_Value(self, offDutyTime):
        self.offDutyTime = offDutyTime

    def change_cam_select(self, camSelect):
        self.camSelect = camSelect

    def set_filePath(self, filePath):
        self.isOpenVideo = True
        self.filePath = filePath

    def set_open_video(self, isOpenVideo):
        self.isOpenVideo = isOpenVideo

    def close(self):
        self.isClose = True

    @staticmethod
    def playMusic():
        mixer.init()
        mixer.music.load('warning.mp3')
        mixer.music.play()

    def run(self):
        self.window.emit("开始程序")
        self.isClose = False
        # 打开相机/视频
        if self.isOpenVideo:
            if self.filePath is None:
                self.window.emit("未加载视频，请加载视频后再点击")
                return

            self.cap = cv2.VideoCapture(self.filePath)
            self.msg.emit("视频读取成功")
        else:
            self.cap = cv2.VideoCapture(self.camSelect, cv2.CAP_DSHOW)
            self.msg.emit("相机打开成功")
        # 初始化数据参数（测试次数、测试EAR、MAR和HAR的和、测试次数魔法值）
        test_time = 0
        TEST_TIMES = 100
        ear_sum = 0
        mar_sum = 0
        har_sum = 0

        Detected_TIME_LIMIT = 60
        closed_times = 0
        yawning_times = 0
        pitch_times = 0
        warning_time = 0

        # 阈值（EAR、MAR、HAR、per clos阈值）
        EAR_THRESH = 0
        MAR_THRESH = 0
        HAR_THRESH = 0
        FATIGUE_THRESH = 0.4
        PITCH_THRESH = 6
        offDutyTime = 0

        self.msg.emit("程序正在计算面部特征阈值，请您耐心等待")
        self.window.emit("程序正在计算面部特征阈值，请您耐心等待")
        # 从视频流循环帧
        while True:
            # 进行循环，读取图片，对图片做维度扩大，进行灰度化处理,以及进行拉普拉斯滤波增强对比
            ret, frame = self.cap.read()
            if not ret:
                if self.isOpenVideo:
                    self.window.emit("视频播放结束")
                print('视频结束')
                break

            frame = imutils.resize(frame, width=720)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 限制对比度自适应直方图均衡
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # 使用detector(gray, 0) 进行脸部位置检测
            rects = self.detector(gray, 0)
            # 脱岗检测
            if not rects:
                if self.isOffDutyCheck:
                    offDutyTime += 1
                    if offDutyTime >= self.offDutyTime * 30:
                        self.msg.emit("您已经脱岗，请立刻回到岗位")
                        self.window.emit("您已经脱岗，请立刻回到岗位")
                        offDutyTime = 0
            else:
                offDutyTime = 0

            # 面部特征检测
            for rect in rects:
                # 使用predictor(gray, rect)获得脸部特征位置的信息
                shape = self.predictor(gray, rect)

                if self.isShowKeyPoint:
                    # 8.2 获取关键点的坐标
                    for point in shape.parts():
                        # 每个点的坐标
                        point_position = (point.x, point.y)
                        # 8.4 绘制关键点
                        cv2.circle(frame, point_position, 3, (255, 8, 0), -1)

                # 将脸部特征信息转换为数组array的格式
                shape = face_utils.shape_to_np(shape)
                # 提取左眼、右眼坐标、嘴巴坐标
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]
                # 构造函数计算左右眼的EAR平均值、计算嘴巴MAR值
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth)
                # 获取头部姿态
                reprojectdst, euler_angle = get_head_pose(shape)
                # 取pitch（har）、yaw、roll旋转角度
                pitch = euler_angle[0, 0]
                yaw = euler_angle[1, 0]
                roll = euler_angle[2, 0]
                har = pitch

                if self.isShowHead:
                    # 绘制正方体12轴
                    for start, end in line_pairs:
                        start_point = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
                        end_point = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
                        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

                # 实时显示计算结果
                cv2.putText(frame, "ear: {}".format(ear), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "mar: {}".format(mar), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "pitch: {:5.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2)
                cv2.putText(frame, "yaw: {:5.2f}".format(yaw), (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "roll: {:5.2f}".format(roll), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)

                # 计算100次ear、mar和har数据求平均值，得到当前使用者眼部、嘴部、头部俯仰的阈值
                if test_time < TEST_TIMES:
                    test_time += 1
                    ear_sum += ear
                    mar_sum += mar
                    har_sum += har

                    if test_time == TEST_TIMES:
                        EAR_THRESH = ear_sum / TEST_TIMES
                        MAR_THRESH = mar_sum / TEST_TIMES
                        HAR_THRESH = har_sum / TEST_TIMES
                        print('眼睛长宽比ear 100次取平均的阈值:{:.2f} '.format(EAR_THRESH))
                        print('嘴部长宽比mar 100次取平均的阈值:{:.2f} '.format(MAR_THRESH))
                        print('头部俯仰角pitch 100次取平均的阈值:{:.2f} '.format(HAR_THRESH))
                        self.msg.emit('眼睛长宽比ear 100次取平均的阈值:{:.2f}'.format(EAR_THRESH))
                        self.msg.emit('嘴部长宽比mar 100次取平均的阈值:{:.2f}'.format(MAR_THRESH))
                        self.msg.emit('头部俯仰角pitch 100次取平均的阈值:{:.2f}'.format(HAR_THRESH))
                    continue

                # 画图,嘴巴、眼睛凸包标注,用矩形框标注人脸
                if self.isShowEye:
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if self.isShowMouth:
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

                '''
                    计算检测时间内，异常状态的次数
                    异常状态定义:
                    1.EAR 小于0.8标记为闭合
                    2.MAR 大于1.5倍即异常
                    3.HAR 跟阈值差大于标准值
                '''
                if Detected_TIME_LIMIT > 0:
                    Detected_TIME_LIMIT -= 1
                    if ear < 0.75 * EAR_THRESH:
                        closed_times += 1
                    if mar > 1.6 * MAR_THRESH:
                        yawning_times += 1
                    if abs(har - HAR_THRESH) > PITCH_THRESH:
                        pitch_times += 1

                else:
                    # 重置Detected_TIME_LIMIT
                    Detected_TIME_LIMIT = 60
                    isEyeTired = False
                    isYawnTired = False
                    isHeadTired = False

                    # 判断是否疲劳,大于阈值则疲劳
                    if closed_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                        self.msg.emit("闭眼时长较长")
                        isEyeTired = True

                    if yawning_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                        self.msg.emit("张嘴时长较长")
                        isYawnTired = True

                    if pitch_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                        self.msg.emit("低头时长较长")
                        isHeadTired = True

                    # 重置次数
                    closed_times = 0
                    yawning_times = 0
                    pitch_times = 0

                    isWarning = False
                    # 疲劳状态判断
                    if isEyeTired and isYawnTired:
                        warning_time += 2
                        isWarning = True
                    elif isHeadTired and isEyeTired:
                        warning_time += 2
                        isWarning = True
                    elif isEyeTired:
                        warning_time += 1
                        isWarning = True
                    elif isYawnTired:
                        warning_time += 1
                        isWarning = True
                    elif isHeadTired:
                        warning_time += 1
                        isWarning = True
                    else:
                        warning_time = 0

                    if warning_time >= 3:
                        warning_time = 0
                        self.msg.emit("您已经疲劳，请注意休息")
                        self.window.emit("您已经疲劳，请注意休息")
                        self.playMusic()
                    else:
                        if isWarning:
                            winsound.Beep(440, 1000)

            # 窗口显示 show with opencv
            self.picture.emit(frame)

            if self.isClose:
                break

        self.cap.release()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
