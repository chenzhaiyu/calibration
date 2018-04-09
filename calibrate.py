# - *coding: gbk *-

import numpy as np
import cv2
import glob

# 格网交叉点列数和行数
Grid_Point_Col = 8
Grid_Point_Row = 6

# 设定cornerSubPix方法的最大迭代次数和位置变化最小值作为迭代终止条件
criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 40, 0.001

# 屏幕上棋盘点，存储世界坐标，Z取0，每张影像对应的棋盘点坐标是一样的
chessboard_points = np.zeros((Grid_Point_Col * Grid_Point_Row, 3), np.float32)
chessboard_points[:, :2] = np.mgrid[0:Grid_Point_Col, 0:Grid_Point_Row].T.reshape(-1, 2)

# 像点和物点
image_points = []
object_points = []

# 影像路径
image_paths = glob.glob('dataset/*.jpg')

for image_path in image_paths:
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 在灰度影像上找到棋盘点角点，cvFindChessboardCorners函数试图确定输入图像是否是棋盘模式，并确定角点的位置
    is_chessboard_found, corners = cv2.findChessboardCorners(image_gray, (Grid_Point_Col, Grid_Point_Row))

    print(corners)
    if is_chessboard_found:
        cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)
        object_points.append(chessboard_points)
        print(corners)
        cv2.drawChessboardCorners(image, (Grid_Point_Col, Grid_Point_Row), corners, is_chessboard_found)
        cv2.imshow('Chessboard Calibration', image)
        cv2.imwrite('imageCalibration.jpg', image)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_gray.shape[::-1], None, None)

print "\nCamera Matrix:\n"
print mtx
print "\n\nDistortion Coefficient:\n"
print dist
print "\n\nRotation Vector:\n"
print rvecs
print "\n\nTranslation Vector:\n"
print tvecs

# img = cv2.imread('dataset/24.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# cv2.imwrite('calibresult.jpg', dst)
