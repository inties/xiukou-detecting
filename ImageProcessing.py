import sys
import vtk
from pandas.compat.pickle_compat import load_newobj
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog
import numpy as np
import cv2
import time
from vtkmodules.util import numpy_support

from concurrent.futures import ThreadPoolExecutor
import os
import get_contour_dir


class VTKViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.setWindowTitle("3D Model Viewer with Edge Detection")

        # 设置主窗口的初始尺寸为500x500，并限制其最小和最大尺寸
        self.setGeometry(100, 100, 1000, 1000)
        self.setMinimumSize(1000, 1000)
        self.setMaximumSize(1000, 1000)

        # 创建VTK渲染窗口并设置其尺寸
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setMinimumSize(500, 500)  # 使用 setMinimumSize
        self.vtk_widget.setMaximumSize(500, 500)  # 使用 setMaximumSize
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)

        # 设置布局
        main_layout = QHBoxLayout()  # 使用水平布局
        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(main_layout)

        # 添加VTK渲染窗口到布局
        main_layout.addWidget(self.vtk_widget)

        # 创建垂直布局用于按钮区域
        button_layout = QVBoxLayout()

        load_newobj_button= QPushButton("Load new Obj")
        load_newobj_button.clicked.connect(self.load_new_obj)
        button_layout.addWidget(load_newobj_button)
        # 添加加载模型按钮
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        button_layout.addWidget(load_button)

        # 添加图像捕获按钮
        image_capture_button = QPushButton("Capture images and save")
        image_capture_button.clicked.connect(self.capture_images)
        button_layout.addWidget(image_capture_button)

        # 添加边缘检测按钮
        edge_detection_button = QPushButton("Detect Edges and Save")
        edge_detection_button.clicked.connect(self.detect_edges_and_save)
        button_layout.addWidget(edge_detection_button)
        # 将按钮区域添加到主布局
        main_layout.addLayout(button_layout)

        # 初始化交互
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

        self.model_path = None
        self.polydata = None

    def render_model_with_color_coding(self, file_name):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()

        polydata = reader.GetOutput()

        # 创建颜色映射
        num_cells = polydata.GetNumberOfCells()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)  # RGB
        colors.SetName("Colors")

        for i in range(num_cells):
            # 使用HSV色彩空间来生成更独特且视觉上区分度更高的颜色
            h = (i / num_cells) * 360  # 色调H根据面ID分配
            s = 1.0  # 饱和度S固定为最大值
            v = 1.0  # 亮度V固定为最大值
            hsv = np.uint8([[[h / 2, 255, 255]]])
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            color = [rgb[0][0][0], rgb[0][0][1], rgb[0][0][2]]
            colors.InsertNextTypedTuple(color)

        polydata.GetCellData().SetScalars(colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray("Colors")
        mapper.InterpolateScalarsBeforeMappingOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLighting(False)  # 禁用光照以确保颜色准确反映面ID

        # 清除之前的渲染
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()

        # 更新渲染窗口（仅一次）
        self.vtk_widget.GetRenderWindow().Render()

        # 保存polydata以便后续使用
        self.polydata = polydata
        self.face_num=num_cells

        
    def parse_obj_groups(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        current_group = 'DefaultGroup'
        face_indices = []
        group_faces = {'DefaultGroup': [], 'EdgeFaces': []}

        for line in lines:
            if line.startswith('g '):
                current_group = line.strip().split(' ')[1]
            elif line.startswith('f '):
                face_indices.append(line)
                group_faces[current_group].append(len(face_indices) - 1)

        return group_faces, face_indices

    def render_output_obj(self, file_name):
        group_faces, face_indices = self.parse_obj_groups(file_name)

        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()

        polydata = reader.GetOutput()

        # 创建一个标量数组用于存储组信息
        group_ids = vtk.vtkUnsignedCharArray()
        group_ids.SetNumberOfComponents(1)
        group_ids.SetName("GroupIDs")

        # 遍历所有面，并根据组信息设置 ID
        for i in range(polydata.GetNumberOfCells()):
            if i in group_faces.get('EdgeFaces', []):
                group_ids.InsertNextValue(1)  # EdgeFaces 组
            else:
                group_ids.InsertNextValue(0)  # DefaultGroup 组

        # 将组 ID 添加到单元格数据中
        polydata.GetCellData().SetScalars(group_ids)

        # 创建颜色映射表
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(2)
        lut.SetTableRange(0, 1)
        lut.Build()

        # 定义颜色
        edge_face_color = [1.0, 0.0, 0.0, 1.0]  # 红色
        default_group_color = [0.5, 0.5, 0.5, 1.0]  # 灰色

        # 设置颜色映射表
        lut.SetTableValue(0, *default_group_color)  # DefaultGroup
        lut.SetTableValue(1, *edge_face_color)  # EdgeFaces

        # 创建 mapper 并设置颜色映射
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOn()  # 启用标量可见性
        mapper.SetScalarModeToUseCellData()  # 使用单元格数据
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 1)  # 设置标量范围

        # 创建 actor 并设置属性
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLighting(True)  # 启用光照

        # 清除之前的渲染
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()

        # 更新渲染窗口
        self.vtk_widget.GetRenderWindow().Render()


    # def render_output_obj(self, file_name):
    #     reader = vtk.vtkOBJReader()
    #     reader.SetFileName(file_name)
    #     reader.Update()

    #     polydata = reader.GetOutput()

    #     # 创建一个标量数组用于存储组信息
    #     group_ids = vtk.vtkUnsignedCharArray()
    #     group_ids.SetNumberOfComponents(1)
    #     group_ids.SetName("GroupIDs")

    #     # 遍历所有面，并根据组信息设置 ID (0 表示 DefaultGroup，1 表示 EdgeFaces)
    #     for i in range(polydata.GetNumberOfCells()):
    #         cell = polydata.GetCell(i)
    #         if isinstance(cell, vtk.vtkTriangle):  # 假设所有面都是三角形
    #             # 获取面所在的组名
    #             group_name = polydata.GetCellData().GetArray('group_names').GetComponent(i, 0) if \
    #                 polydata.GetCellData().HasArray('group_names') else 'DefaultGroup'

    #             # 根据组名设置 ID
    #             if group_name == 'EdgeFaces':
    #                 group_ids.InsertNextValue(1)  # EdgeFaces 组
    #             else:
    #                 group_ids.InsertNextValue(0)  # DefaultGroup 组

    #     # 将组 ID 添加到单元格数据中
    #     polydata.GetCellData().SetScalars(group_ids)

    #     # 创建颜色映射表
    #     lut = vtk.vtkLookupTable()
    #     lut.SetNumberOfColors(2)
    #     lut.SetTableRange(0, 1)
    #     lut.Build()

    #     # 定义颜色
    #     edge_face_color = [1.0, 0.0, 0.0, 1.0]  # 红色
    #     default_group_color = [0.5, 0.5, 0.5, 1.0]  # 灰色

    #     # 设置颜色映射表
    #     lut.SetTableValue(0, *default_group_color)  # DefaultGroup
    #     lut.SetTableValue(1, *edge_face_color)  # EdgeFaces

    #     # 创建 mapper 并设置颜色映射
    #     mapper = vtk.vtkPolyDataMapper()
    #     mapper.SetInputData(polydata)
    #     mapper.ScalarVisibilityOn()  # 启用标量可见性
    #     mapper.SetScalarModeToUseCellData()  # 使用单元格数据
    #     mapper.SetLookupTable(lut)
    #     mapper.SetScalarRange(0, 1)  # 设置标量范围

    #     # 创建 actor 并设置属性
    #     actor = vtk.vtkActor()
    #     actor.SetMapper(mapper)
    #     actor.GetProperty().SetLighting(True)  # 启用光照

    #     # 清除之前的渲染
    #     self.vtk_renderer.RemoveAllViewProps()
    #     self.vtk_renderer.AddActor(actor)
    #     self.vtk_renderer.ResetCamera()

    #     # 更新渲染窗口
    #     self.vtk_widget.GetRenderWindow().Render()
    def render_model(self, file_name):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 0.5, 0.5)  # 设置颜色为灰色
        actor.GetProperty().SetLighting(True)  # 启用光照

        # 清除之前的渲染
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()

        # 更新渲染窗口
        self.vtk_widget.GetRenderWindow().Render()

    def capture_window_image(self, use_color_coding=False):
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.vtk_widget.GetRenderWindow())
        if use_color_coding:
            self.render_model_with_color_coding(self.model_path)
        else:
            self.render_model(self.model_path)
        window_to_image_filter.Update()  # 渲染需要时间，但这里并没有引入异步
        img_data = window_to_image_filter.GetOutput()
        # 提交图像数据转换任务给线程池，并返回 Future 对象
        future = self.executor.submit(self._convert_to_numpy, img_data,use_color_coding)
        return future  # 返回 Future 对象，允许调用者等待或附加回调

    #完成到numpy的转换，并且将得到的图片保存到文件夹中
    def _convert_to_numpy(self, img_data,use_color_coding):#use_color_coding决定保存图片文件的名称
        """在后台线程中将 VTK 图像数据转换为 NumPy 数组"""
        print("converting")
        img_data_array =numpy_support.vtk_to_numpy(img_data.GetPointData().GetScalars()).reshape(500, 500, 3)
        img_data_array = img_data_array.reshape(500,500,-1)

        img_data_array_ = cv2.cvtColor(img_data_array, cv2.COLOR_RGB2BGR)
        base_path = r"pictures"
        filename = "picture_ID.png" if use_color_coding else "picture.png"
        file_path = os.path.join(base_path, filename)
        print(file_path)
        # 保存图像到磁盘
        save_status = cv2.imwrite(file_path, img_data_array_)
        if save_status:
            print(f"Image saved successfully to {file_path}")
        else:
            print(f"Failed to save image to {file_path}")
        return img_data_array

    def _handle_future(self, future, image_type):
        try:
            print("load images")
            img = future.result()  # 在子线程中获取结果，不会阻塞主线程
            print("1")
            if image_type == 'color_coded':
                self.color_coded_img = img
                print("2")
            elif image_type == 'normal':
                self.normal_img = img
                print("3")

            # 如果两个图像都已准备好，则发出信号通知边缘检测完成
            if hasattr(self, 'color_coded_img') and hasattr(self, 'normal_img'):
                print("images OK")
                # self.edge_detection_finished.emit(self.color_coded_img, self.normal_img)
        except Exception as e:
            print(f"Error processing {image_type} image: {e}")

    # 获取面ID颜色编码的图像和正常着色的图像
    def capture_images(self):
        print("capturing images")
        color_coded_future = self.capture_window_image(use_color_coding=True)
        normal_future = self.capture_window_image(use_color_coding=False)

    #读取图片、检测边缘、保存绘制有边缘的图片并display
    def thread_detect_edges(self):
        app = QApplication(sys.argv)

        # 输入和保存路径
        # image_path = "images/view_2.png"
        image_path = r"pictures\picture.png"
        id_image_path=r"pictures\picture_ID.png"
        output_path = r"pictures"

        if os.path.isfile(image_path):
            print(f"Processing single file: {image_path}")
            file = os.path.splitext(os.path.basename(image_path))[0]
            save_file = os.path.join(output_path, file + "_contour.png")
            # 计算轮廓，返回边缘像素对应的面,以及边缘像素对应面id
            contour_image,edge_face_ids = get_contour_dir.calculate_contours(image_path, id_image_path,self.face_num, save_file)
            if contour_image is None:
                sys.exit("No contours found. Exiting.")
            print(len(edge_face_ids))
            print(self.face_num)
            obj_file_path=r"objFiles\cuff.obj"
            output_obj_file_path=r"objFiles\cuff_output.obj"
            self.mark_edge_faces_in_obj(obj_file_path, edge_face_ids,output_obj_file_path)

            # 显示结果
            window = get_contour_dir.ContourVisualizer(contour_image)
            window.resize(800, 600)
            window.show()
            # 等待用户关闭窗口
            app.exec_()
    def mark_edge_faces_in_obj(self, obj_file_path, edge_face_ids, output_obj_file_path):
        """
        修改 OBJ 文件，在其中标明哪些面是边缘面，并将同组的面放在一起，同时保留元信息。

        :param obj_file_path: 原始 OBJ 文件路径
        :param edge_face_ids: 边缘面 ID 的集合（假设是1-based）
        :param output_obj_file_path: 修改后的 OBJ 文件保存路径
        """
        with open(obj_file_path, 'r') as file:
            lines = file.readlines()

        # 分离元信息、非面定义和其他行
        metadata_lines = []
        non_face_lines = []
        face_lines = []

        is_metadata_section = True  # 标记是否在元信息部分

        for line in lines:
            if line.startswith('#'):
                metadata_lines.append(line)
            elif line.strip() == '' or (
                    is_metadata_section and not line.startswith('v ') and not line.startswith('f ')):
                metadata_lines.append(line)  # 包含空行和其他元信息行
            else:
                is_metadata_section = False
                if line.startswith('f '):  # 面定义行
                    face_lines.append(line)
                else:
                    non_face_lines.append(line)

        # 分类面为边缘面和非边缘面
        edge_faces = []
        non_edge_faces = []

        for idx, line in enumerate(face_lines, start=0):
            if idx in edge_face_ids:
                edge_faces.append(line)
            else:
                non_edge_faces.append(line)

        # 写入新的 OBJ 文件
        with open(output_obj_file_path, 'w') as file:
            # 写入元信息行
            for line in metadata_lines:
                file.write(line)

            # 写入非面定义行
            for line in non_face_lines:
                file.write(line)

            # 写入边缘面
            if edge_faces:
                file.write('g EdgeFaces\n')
                for line in edge_faces:
                    file.write(line)

            # 写入非边缘面
            if non_edge_faces:
                file.write('g DefaultGroup\n')
                for line in non_edge_faces:
                    file.write(line)
        print("save obj successfully")
    def detect_edges_and_save(self):
        print("Detecting edges and saving...")
        future = self.executor.submit(self.thread_detect_edges)

        return future

    def load_new_obj(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open 3D Model", "", "3D Model Files (*.obj);;All Files (*)",
                                                   options=options)
        if file_name:
            self.model_path = file_name
            self.render_model(file_name)
            print("Model loaded successfully.")


    def load_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open 3D Model", "", "3D Model Files (*.obj);;All Files (*)",
                                                   options=options)
        if file_name:
            self.model_path = file_name
            self.render_output_obj(file_name)
            print("Model loaded successfully.")

    def cleanup(self):
        self.executor.shutdown(wait=True)  # 关闭线程池

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = VTKViewer()
    viewer.show()
    sys.exit(app.exec_())