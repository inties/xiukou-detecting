'''
(1) 功能说明：输入三维模型，获取不同视点的图像。示例代码。
(2) 开发单位：电子科技大学数字媒体技术团队
(3) 作者：蔡洪斌
(4) 联系方式：QQ7849952
(5) 创建日期：2025-01-05
(6) 重要修改：
'''
import os
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PerspectiveCameras,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
    DirectionalLights,  # 导入方向光源
    PointLights,  # 使用点光源
)
import matplotlib.pyplot as plt

# 配置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义文件路径
obj_path = "obj/cuff.obj"
output_dir = "./images"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 检查 OBJ 文件是否存在
if not os.path.exists(obj_path):
    raise FileNotFoundError(f"文件 {obj_path} 不存在！")

# 加载 3D 模型
mesh = load_objs_as_meshes([obj_path], device=device)

# 设置模型的纹理为白色
verts = mesh.verts_packed()  # 顶点
verts_rgb = torch.ones_like(verts)  # RGB 白色
verts_rgb = verts_rgb.unsqueeze(0)  # 增加批量维度
textures = TexturesVertex(verts_features=verts_rgb.to(device))
mesh.textures = textures

# 计算模型质心
center = verts.mean(dim=0).tolist()
print(f"模型质心位置: {center}")

# 设置视点偏移量
offset_x = 1.0  # 沿 X 轴的偏移量
offset_y = 1.0  # 沿 Y 轴的偏移量
offset_z = 8.0  # 沿 Z 轴的偏移量。0会出错

# 定义三个视点：质心、质心+offset、质心-offset
view_positions = [
    [center[0],            center[1] + offset_y, center[2] + offset_z],  # 质心位置
    [center[0] + offset_x, center[1] + offset_y, center[2] + offset_z],  # 沿 X 轴正向移动
    [center[0] - offset_x, center[1] + offset_y, center[2] + offset_z],  # 沿 X 轴负向移动
]

# 渲染设置
image_size = 512
raster_settings = RasterizationSettings(
    image_size=image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# 设置方向光源
lights = DirectionalLights(
    device=device,
    direction=torch.tensor([[0.0, 0.0, 1.0]], device=device),  # 光源方向，指向正 Z 轴
    ambient_color=torch.tensor([[1.0, 1.0, 1.0]], device=device),  # 环境光
    diffuse_color=torch.tensor([[1.0, 1.0, 1.0]], device=device),  # 漫反射光
)

# 设置渲染器
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, lights=lights),  # 将光源传递给shader
)

# 渲染并保存图像
for i, eye in enumerate(view_positions):
    # 使用 look_at_view_transform 创建视点
    R, T = look_at_view_transform(
        dist=3.0,
        elev=0,
        azim=0,
        at=torch.tensor([center], device=device),  # 目标点
        eye=torch.tensor([eye], device=device),  # 相机位置
    )
    camera = PerspectiveCameras(device=device, R=R, T=T)

    # 渲染图像
    images = renderer(mesh, cameras=camera)
    rendered_image = images[0, ..., :3].cpu().numpy()  # 提取 RGB 图像

    # 归一化图像，使其在 0..1 范围内
    rendered_image = (rendered_image - rendered_image.min()) / (rendered_image.max() - rendered_image.min())

    # 保存图像
    output_path = os.path.join(output_dir, f"view_{i + 1}.png")
    plt.imsave(output_path, rendered_image)
    print(f"图像已保存到: {output_path}")

    # 显示生成的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(rendered_image)
    plt.title(f"View {i + 1}")
    plt.axis("off")

    plt.show()
