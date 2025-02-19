'''
(1) 功能说明：输入图像文件或目录，获取轮廓
(2) 开发单位：电子科技大学数字媒体技术团队
(3) 作者：蔡洪斌
(4) 联系方式：QQ7849952
(5) 创建日期：2025-01-05
(6) 重要修改：
'''
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap



def edge_pix2face(edges,id_image,face_num):
    color_coded_img_rgb = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
    edge_pixels = np.argwhere(edges > 0)
    edge_face_ids = set()
    for pixel in edge_pixels:
        y, x = pixel
        color = color_coded_img_rgb[y, x]
        # 解码颜色回面ID
        h, _, _ = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
        face_id = int(h / 360 * face_num)
        edge_face_ids.add(face_id)
    return edge_face_ids

#计算边缘，并将带有边缘的图像保存和显示；获取边缘像素对应的面
def calculate_contours(image_path, id_image_path, face_num,save_file):
    """
    计算轮廓并保存结果图像。
    :param image_path: 输入图像路径
    :param save_file: 结果保存路径
    :return: 处理后的轮廓图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    id_image = cv2.imread(id_image_path)

    if image is None:
        raise ValueError("Image not found or unable to load.")

    # 转为灰度图并二值化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # 使用边缘检测
    edges = cv2.Canny(gray_image, 100, 200)

    # edges = cv2.Canny(binary_image, 100, 200)
    #找到边缘像素对应的面，然后

    edge_face_ids=edge_pix2face(edges,id_image, face_num)


    # 寻找轮廓
    contours_info = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if len(contours) == 0:
        print("No contours found.")
        return None

    # 在原图上绘制轮廓
    contour_image = image.copy()
    print(f"***contours={len(contours)}")
    num = 0
    for i, contour in enumerate(contours):
        if len(contour) < 20:  # 过滤短轮廓
            continue
        color = (
            int(((i + 10) * 50) % 256),
            int(((i + 10) * 80) % 256),
            int(((i + 10) * 120) % 256),
        )
        cv2.drawContours(contour_image, [contour], -1, color, 2)
        num += 1
        print(f"Contour {num}: Number of points = {len(contour)}")

    # 保存结果图像
    cv2.imwrite(save_file, contour_image)
    return contour_image,edge_face_ids


class ContourVisualizer(QMainWindow):
    def __init__(self, contour_image):
        super().__init__()
        self.setWindowTitle("Contour Visualization")
        self.display_image(contour_image)

    def display_image(self, img):
        # 将BGR图像转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 创建标签并显示图像
        label = QLabel(self)
        label.setPixmap(QPixmap.fromImage(q_img))

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#
#     # 输入和保存路径
#     # image_path = "images/view_2.png"
#     image_path = r"E:\pictures\picture.png"
#     output_path = r"E:\pictures"
#
#     if os.path.isfile(image_path):
#         print(f"Processing single file: {image_path}")
#         file = os.path.splitext(os.path.basename(image_path))[0]
#         save_file = os.path.join(output_path, file + "_contour.png")
#         # 计算轮廓
#         contour_image = calculate_contours(image_path, save_file)
#         if contour_image is None:
#             sys.exit("No contours found. Exiting.")
#
#         # 显示结果
#         window = ContourVisualizer(contour_image)
#         window.resize(800, 600)
#         window.show()
#         # 等待用户关闭窗口
#         app.exec_()
#
#     elif os.path.isdir(image_path):
#         print(f"Processing directory: {image_path}")
#         png_files = [f for f in os.listdir(image_path) if f.lower().endswith(".png")]
#         if not png_files:
#             print(f"No PNG files found in directory: {image_path}")
#
#         for png_file in png_files:
#             image_file = os.path.join(image_path, png_file)
#             print(f"Processing {image_file}...")
#             save_file = os.path.join(output_path, png_file.rsplit('.', 1)[0] + "_contour.png")
#
#             # 计算轮廓
#             contour_image = calculate_contours(image_file, save_file)
#             if contour_image is None:
#                 sys.exit("No contours found. Exiting.")
#
#             # 显示结果
#             window = ContourVisualizer(contour_image)
#             window.resize(800, 600)
#             window.show()
#
#
#     sys.exit()
