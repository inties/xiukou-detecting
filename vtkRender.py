import vtk

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
        color = [255*i/num_cells, 255*i/num_cells, 255*i/num_cells]  # 将cell ID编码到RGB
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
    actor.GetProperty().SetLighting(True)  # 启用光照

    # 清除之前的渲染
    self.vtk_renderer.RemoveAllViewProps()
    self.vtk_renderer.AddActor(actor)
    self.vtk_renderer.ResetCamera()

    # 更新渲染窗口（仅一次）
    self.vtk_widget.GetRenderWindow().Render()