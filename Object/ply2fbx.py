import bpy
import sys

# 명령줄에서 인자로 받은 PLY 파일 경로와 FBX 파일 경로
input_ply = sys.argv[-2]  # PLY 파일 경로
output_fbx = sys.argv[-1]  # FBX 파일 경로

# 기존 데이터 삭제 (초기화)
bpy.ops.wm.read_factory_settings(use_empty=True)

# PLY 파일 불러오기
bpy.ops.import_mesh.ply(filepath=input_ply)

# 현재 활성 객체 가져오기
obj = bpy.context.object

# UV 맵 생성 (Smart UV Project 사용)
bpy.ops.object.mode_set(mode='EDIT')  # 편집 모드로 전환
bpy.ops.uv.smart_project()  # UV 맵 생성
bpy.ops.object.mode_set(mode='OBJECT')  # 객체 모드로 전환

# 점 색상(Vertex Color)을 재질로 변환
if obj.data.vertex_colors:
    color_layer = obj.data.vertex_colors.active  # 활성화된 점 색상 레이어

    # 새 재질 생성
    mat = bpy.data.materials.new(name="VertexColorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # 재질 노드 설정
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        color_node = nodes.new(type="ShaderNodeVertexColor")
        color_node.layer_name = color_layer.name
        links.new(color_node.outputs["Color"], bsdf.inputs["Base Color"])

    # 재질을 객체에 할당
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# FBX 파일로 내보내기
bpy.ops.export_scene.fbx(filepath=output_fbx)

print(f"UV 맵과 색상 정보가 포함된 FBX 파일로 변환 완료: {output_fbx}")

