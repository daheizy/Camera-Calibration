# Camera-Calibration
  张正友相机标定，使用棋盘格输入序列进行相机标定，输出相机内参；畸变参数；相机内参等。
## 开始之前：
所需环境：
cv2

numpy

os

## 输入：棋盘格
  文件中棋盘格为w = 9；h = 6
## 文件标注：
  p1： 原始图片存放位置
  p2： 更改大小图片存放位置
  p3： 最后图片的保存位置
## 输出：
  ret: 重投影误差；
  mtx：内参矩阵；
  dist：畸变系数；
  rvecs：旋转向量 （外参数）；
  tvecs：平移向量 （外参数）；
  在最后使用cv2.getOptimalNewCameraMatrix对相机内参进行优化。
## 
