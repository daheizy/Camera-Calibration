# Camera-Calibration
  张正友相机标定，使用棋盘格输入序列进行相机标定，输出相机内参；畸变参数；相机内参等。



## 开始之前：
所需环境：<br>
cv2<br>
numpy<br>
os<br>
## 输入：棋盘格
  文件中棋盘格为w = 9；h = 6

数据采集时要保证棋盘格的平整，不然会有较大的误差。

## 文件标注：
  p1： 原始图片存放位置<br>
  p2： 更改大小图片存放位置<br>
  p3： 最后图片的保存位置<br>
## 输出：
  ret: 重投影误差；<br>
  mtx：内参矩阵；<br>
  dist：畸变系数；<br>
  rvecs：旋转向量 （外参数）；<br>
  tvecs：平移向量 （外参数）；<br>
  在最后使用cv2.getOptimalNewCameraMatrix对相机内参进行优化。<br>

