import vtk

def read_obj_and_group_faces(file_path):
    group_faces = {0: [], 1: []}
    current_group = None
    sharp_num = 0
    line_num=0
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith('#'):
                sharp_num += 1
                # Check for group markers
                if 'edge' in stripped_line:
                    current_group = 0
                  
                elif 'default' in stripped_line:
                    current_group = 1
            elif stripped_line.startswith('f'):
                if current_group is not None:
                    # Extract face indices (assuming no texture or normal indices)
                    line_num+=1
                    indices = list(map(int, stripped_line.split()[1:]))
                    indices_sorted = sorted(indices)
                    group_faces[current_group].append(indices_sorted)
                    # group_faces[current_group].append(indices)
    print(f"组一 faces count: {len(group_faces[0])}")
    print(f"组二 faces count: {len(group_faces[1])}")  
    print(sharp_num)            
    print(line_num)
    return group_faces


def main():
    obj_file = "cuff.obj"  # Replace with your OBJ file path
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update()

    polydata = reader.GetOutput()

    group_faces = read_obj_and_group_faces(obj_file)
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetName("Colors")
    color_array.SetNumberOfComponents(3)
    color_array.SetNumberOfTuples(polydata.GetNumberOfCells())

    color_red = [255, 0, 0]   # Color for Group 一
    color_blue = [0, 0, 255]  # Color for Group 二
    
    group_1_num=0
    group_2_num=0

    # Print the vertex IDs of the first ten faces in polydata
    for i in range(min(10, polydata.GetNumberOfCells())):
        cell = polydata.GetCell(i)
        point_ids = cell.GetPointIds()
        ids = [point_ids.GetId(j) for j in range(point_ids.GetNumberOfIds())]
        print(f"Face {i} vertex IDs: {ids}")
    # Assign colors based on the group faces
    for i in range(polydata.GetNumberOfCells()):
        if i <= 175:
            color_array.SetTuple3(i, *color_red)
            group_1_num += 1
        else:
            color_array.SetTuple3(i, *color_blue)
            group_2_num += 1
      
    print(f"组一 faces count: {group_1_num}")
    print(f"组二 faces count: {group_2_num}")
    polydata.GetCellData().SetScalars(color_array)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)

    render_window.Render()
    render_window_interactor.Start()


if __name__ == "__main__":
    main()