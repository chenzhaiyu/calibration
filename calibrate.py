# - *coding: gbk *-

import numpy as np
import cv2
import glob

# ���������������������������ÿ�����ӵı߳�(mm)
Grid_Point_Col = 5
Grid_Point_Row = 7
Grid_Size = 20

# �趨cornerSubPix������������������λ�ñ仯��Сֵ��Ϊ������ֹ����
criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 40, 0.001

# ��Ļ�����̵㣬�洢�������꣬Zȡ0��ÿ��Ӱ���Ӧ�����̵�������һ����
chessboard_points = np.zeros((Grid_Point_Col * Grid_Point_Row, 3), np.float32)
chessboard_points[:, :2] = np.mgrid[0:Grid_Point_Col * Grid_Size: Grid_Size, 0:Grid_Point_Row * Grid_Size: Grid_Size].T.reshape(-1, 2)

# �������
image_points = []
object_points = []

# Ӱ��·��
image_paths = glob.glob('dataset/resized/*.JPG')
image_size = None

for image_path in image_paths:
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image_gray

    # �ڻҶ�Ӱ�����ҵ����̵�ǵ㣬cvFindChessboardCorners������ͼȷ������ͼ���Ƿ�������ģʽ����ȷ���ǵ��λ��
    is_chessboard_found, corners = cv2.findChessboardCorners(image_gray, (Grid_Point_Col, Grid_Point_Row))

    if is_chessboard_found:
        # ���ǵ㾫ȷ�������ؼ���
        cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
        # �������������зֱ�����
        image_points.append(corners)
        object_points.append(chessboard_points)
        # ���ƽǵ㣬���
        cv2.drawChessboardCorners(image, (Grid_Point_Col, Grid_Point_Row), corners, is_chessboard_found)
        cv2.namedWindow("Chessboard Calibration", 0)
        cv2.resizeWindow("Chessboard Calibration", 1280, 1024)
        cv2.imshow('Chessboard Calibration', image)
        cv2.imwrite('processing.jpg', image)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# calibrateCamera�����У
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

# ���û�ȡ�ļ�У����������Ӱ����о���
image_before_undistort = cv2.imread('dataset/resized/IMG_2369.JPG')
h, w = image_before_undistort.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
image_after_undistort = cv2.undistort(image_before_undistort, mtx, dist, None, new_camera_matrix)

x, y, w, h = roi
image_after_undistort = image_after_undistort[y:y + h, x:x + w]
cv2.imwrite('result.jpg', image_after_undistort)
