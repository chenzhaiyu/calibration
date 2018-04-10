# - *coding: gbk *-

import numpy as np
import cv2
import glob

# 格网交叉点总列数和总行数，和每个格子的边长(mm)
Grid_Point_Col = 5
Grid_Point_Row = 7
Grid_Size = 20

# 设定cornerSubPix方法的最大迭代次数和位置变化最小值作为迭代终止条件
criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 40, 0.001

# 屏幕上棋盘点，存储世界坐标，Z取0，每张影像对应的棋盘点坐标是一样的
chessboard_points = np.zeros((Grid_Point_Col * Grid_Point_Row, 3), np.float32)
chessboard_points[:, :2] = np.mgrid[0:Grid_Point_Col * Grid_Size: Grid_Size, 0:Grid_Point_Row * Grid_Size: Grid_Size].T.reshape(-1, 2)

# 像点和物点
image_points = []
object_points = []

# 影像路径
image_paths = glob.glob('dataset/resized/*.JPG')
image_size = None

for image_path in image_paths:
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image_gray

    # 在灰度影像上找到棋盘点角点，cvFindChessboardCorners函数试图确定输入图像是否是棋盘模式，并确定角点的位置
    is_chessboard_found, corners = cv2.findChessboardCorners(image_gray, (Grid_Point_Col, Grid_Point_Row))

    if is_chessboard_found:
        # 将角点精确到子像素级别
        cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
        # 往像点和物点矩阵中分别塞点
        image_points.append(corners)
        object_points.append(chessboard_points)
        # 绘制角点，输出
        cv2.drawChessboardCorners(image, (Grid_Point_Col, Grid_Point_Row), corners, is_chessboard_found)
        cv2.namedWindow("Chessboard Calibration", 0)
        cv2.resizeWindow("Chessboard Calibration", 1280, 1024)
        cv2.imshow('Chessboard Calibration', image)
        cv2.imwrite('processing.jpg', image)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# calibrateCamera相机检校
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size.shape[::-1], None, None)


print "\n------RMS:------\n"
print ret
print "\n------Camera Matrix:------\n"
print mtx
print "\n\n------Distortion Coefficient:------\n"
print dist
print "\n\n------Rotation Vector:------\n"
print rvecs
print "\n\n------Translation Vector:------\n"
print tvecs

# 利用获取的检校参数对拍摄影像进行纠正
image_before_undistort = cv2.imread('dataset/resized/IMG_2369.JPG')
h, w = image_before_undistort.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
image_after_undistort = cv2.undistort(image_before_undistort, mtx, dist, None, new_camera_matrix)

x, y, w, h = roi
image_after_undistort = image_after_undistort[y:y + h, x:x + w]
cv2.imwrite('result.jpg', image_after_undistort)
