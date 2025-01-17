#如下代码的运行结果只是赋予不同物体不同的颜色，材质名称是依次命名为了"Material_{i}"

import bpy
import pickle
from mathutils import Vector

# 读取 .pkl 文件
file_path = '/Users/wangzhicheng/Desktop/FSD/Blender_3D/prediction/scene-0003/1533201470448696.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 体素参数
voxel_data = data['pred_voxels']  # (256, 256, 32)
voxel_size = (0.4, 0.4, 0.25) # 每个体素的实际大小 (x, y, z)
origin = (-51.2, -51.2, -5) # 场景的原点 (对应 voxel 的 (0, 0, 0))

# # 创建材质
# materials = []
# for i in range(17):
#     mat = bpy.data.materials.new(name=f"Material_{i}")
#     mat.diffuse_color = (i / 17, (17 - i) / 17, i / 34, 1)
#     materials.append(mat)

# 物体类别和它们对应的颜色 (RGBA)
class_colors = [
    [255, 120,  50, 255],       # barrier              orange
    [255, 192, 203, 255],       # bicycle              pink
    [255, 255,   0, 255],       # bus                  yellow
    [  0, 150, 245, 255],       # car                  blue
    [  0, 255, 255, 255],       # construction_vehicle cyan
    [255, 127,   0, 255],       # motorcycle           dark orange
    [255,   0,   0, 255],       # pedestrian           red
    [255, 240, 150, 255],       # traffic_cone         light yellow
    [135,  60,   0, 255],       # trailer              brown
    [160,  32, 240, 255],       # truck                purple                
    [255,   0, 255, 255],       # driveable_surface    dark pink
    [139, 137, 137, 255],       # other_flat           dark red
    [ 75,   0,  75, 255],       # sidewalk             dark purple
    [150, 240,  80, 255],       # terrain              light green          
    [230, 230, 250, 255],       # manmade              white
    [  0, 175,   0, 255],       # vegetation           green
]

# 创建材质并设置颜色
materials = []
for i, color in enumerate(class_colors):
    mat = bpy.data.materials.new(name=f"Material_{i}")
    mat.diffuse_color = (color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255)
    materials.append(mat)


#-----------------------------------方法1:直接生成一个网格对象，而不是单独添加立方体-------------------------
# 优化：批量创建体素网格
verts = []
faces = []
face_materials = []  # 存储每个面的材质
cube_count = 0

for x in range(len(voxel_data)):
    for y in range(len(voxel_data[0])):
        for z in range(len(voxel_data[0][0])):
            value = voxel_data[x][y][z]
            if value != 0 and 0 <= value-1 < len(materials):   # 忽略值为 0 的体素
                voxel_center = Vector((        # 计算体素中心的世界坐标
                    origin[0] + x * voxel_size[0] + voxel_size[0] / 2,
                    origin[1] + y * voxel_size[1] + voxel_size[1] / 2,
                    origin[2] + z * voxel_size[2] + voxel_size[2] / 2,
                ))
                
                # 添加顶点 (立方体每个角点)
                idx = len(verts)
                size = Vector(voxel_size) / 2
                verts.extend([                   #添加立方体的 8 个顶点到 verts 列表中
                    voxel_center + Vector((-size.x, -size.y, -size.z)),
                    voxel_center + Vector(( size.x, -size.y, -size.z)),
                    voxel_center + Vector(( size.x,  size.y, -size.z)),
                    voxel_center + Vector((-size.x,  size.y, -size.z)),
                    voxel_center + Vector((-size.x, -size.y,  size.z)),
                    voxel_center + Vector(( size.x, -size.y,  size.z)),
                    voxel_center + Vector(( size.x,  size.y,  size.z)),
                    voxel_center + Vector((-size.x,  size.y,  size.z)),
                ])
                # 添加面 (立方体的 6 个面)
                faces.extend([                  #为每个立方体添加 6 个面到 faces 列表中
                    (idx, idx+1, idx+2, idx+3),
                    (idx+4, idx+5, idx+6, idx+7),
                    (idx, idx+1, idx+5, idx+4),
                    (idx+1, idx+2, idx+6, idx+5),
                    (idx+2, idx+3, idx+7, idx+6),
                    (idx+3, idx, idx+4, idx+7),
                ])
                # 为每个面分配材质
                face_materials.extend([value-1] * 6)  # 每个面使用同样的材质

                cube_count += 1

# 创建网格对象
mesh = bpy.data.meshes.new("VoxelMesh")          #创建一个新的网格对象（空网格）
mesh.from_pydata(verts, [], faces)               #使用先前生成的顶点和面数据来填充网格对象
obj = bpy.data.objects.new("VoxelObject", mesh)  #创建一个新的对象，将网格对象与新对象关联
bpy.context.collection.objects.link(obj)         #将新对象链接到当前场景的集合中

# 将所有的材质添加到对象的材质槽中。这里你逐个将materials列表中的材质添加到网格对象obj的data.materials属性中
for i, mat in enumerate(materials):
    obj.data.materials.append(mat)

# 为每个面分配对应的材质
for i, face in enumerate(mesh.polygons):
    face.material_index = face_materials[i]

print(f"Generated {cube_count} cubes.")


#-----------------------------------方法2:单独添加立方体（软件会由于voxel过多直接卡死）-------------------------
# # 遍历体素数据，生成立方体
# for x in range(len(voxel_data)):
#     for y in range(len(voxel_data[0])):
#         for z in range(len(voxel_data[0][0])):
#             value = voxel_data[x][y][z]
#             if value != 0:  # 忽略值为 0 的体素
#                 # 计算体素中心的世界坐标
#                 voxel_center = (
#                     origin[0] + x * voxel_size[0] + voxel_size[0] / 2,
#                     origin[1] + y * voxel_size[1] + voxel_size[1] / 2,
#                     origin[2] + z * voxel_size[2] + voxel_size[2] / 2,
#                 )
                
#                 # 添加立方体
#                 bpy.ops.mesh.primitive_cube_add(
#                     size=1,
#                     location=voxel_center,
#                 )
                
#                 # 缩放立方体到正确的体素大小
#                 bpy.context.object.scale = (voxel_size[0] / 2, voxel_size[1] / 2, voxel_size[2] / 2)
                
#                 # 分配材质
#                 bpy.context.object.data.materials.append(materials[value])