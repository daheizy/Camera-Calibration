import cv2
import numpy as np
import os

# 图片在当前文件夹的位置
file_in = 'p1'   # 原始图片存放位置
file_mid = 'p2'   # 更改大小图片存放位置
file_out = 'p3'   # 最后图片的保存位置

# 棋盘格模板规格，只算内角点个数，不算最外面的一圈点
w = 9
h = 6


def ResizeImage(filein, fileout, width, height):
    """
    改变图片大小，防止图片过大引起崩溃
    :param filein: 输入图片的文件夹路径
    :param fileout: 输出图片的文件夹路径
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :return:
    """
    allImages = os.listdir(filein)
    for fname in allImages:
        img = cv2.imread('p1\\' + fname)
        out = cv2.resize(img, (width, height))
        cv2.imwrite('p2\\' + fname, out)


# 更改图片尺寸
re_w = 512
re_h = 288
ResizeImage(file_in, file_mid, re_w, re_h)


# 找棋盘格角点
# 世界坐标系中的棋盘格点，在张正友标定法中认为Z = 0
# mgrid创建了大小为8×5×2的三维矩阵，在reshape成二维以后赋给objp，objp最后为(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((w * h, 3), np.float32)   # 大小为wh×3的0矩阵
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)   # :2是因为认为Z=0
objpoints = []  # 储存在世界坐标系中的三维点
imgpoints = []  # 储存在图像平面的二维点

images = os.listdir(file_mid)   # 读入图像序列
i = 0

# 算法迭代的终止条件，第一项表示迭代次数达到最大次数时停止迭代，第二项表示角点位置变化的最小值已经达到最小时停止迭代
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for fname in images:
    img = cv2.imread(file_mid + '/' + fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # RGB转灰度
    # 找到棋盘格角点，存放角点于corners，如果找到足够点对，将其存储起来，ret为非零值
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 检测到角点后，进行亚像素级别角点检测，更新角点
    if ret == True:
        i += 1
        # 输入图像gray；角点初始坐标corners；搜索窗口为2*winsize+1；表示窗口的最小（-1.-1）表示忽略；求角点的迭代终止条件
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)   # 空间坐标
        imgpoints.append(corners)  # 角点坐标即图像坐标
        # 角点显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        # cv2.imshow('findCorners', img)
        cv2.imwrite(file_out + '/print_corners' + str(i) + '.jpg', img)
        cv2.waitKey(10)
cv2.destroyAllWindows()

"""
求解参数
输入：世界坐标系里的位置；像素坐标；图像的像素尺寸大小；
输出：
ret: 重投影误差；
mtx：内参矩阵；
dist：畸变系数；
rvecs：旋转向量 （外参数）；
tvecs：平移向量 （外参数）；
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(("ret（重投影误差）:"), ret)
print(("mtx（内参矩阵）:\n"), mtx)
print(("dist（畸变参数）:\n"), dist)  # 5个畸变参数，(k_1,k_2,p_1,p_2,k_3)
print(("rvecs（旋转向量）:\n"), rvecs)
print(("tvecs（平移向量）:\n"), tvecs)


# 优化内参数和畸变系数
# 使用相机内参mtx和畸变系数dist，并使用cv.getOptimalNewCameraMatrix()
# 通过设定自由自由比例因子alpha。
# 当alpha设为0的时候，将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
# 当alpha设为1的时候，将会返回一系个包含额外黑色像素点的内参数和畸变数，并返回一个ROI用于将其剪裁掉。
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (re_w, re_h), 0, (re_w, re_h))


# 矫正畸变
img2 = cv2.imread(file_mid + '/1.jpg')
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
cv2.imwrite(file_out + '/calibresult.jpg', dst)
print("newcameramtx（优化后相机内参）:\n", newcameramtx)

# 反投影误差total_error,越接近0，说明结果越理想。
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)   # 计算三维点到二维图像的投影
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)   # 反投影得到的点与图像上检测到的点的误差
    total_error += error
print(("total error: "), total_error / len(objpoints))   # 记平均
