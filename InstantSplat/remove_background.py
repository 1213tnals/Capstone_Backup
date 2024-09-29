from plyfile import PlyData, PlyElement
import numpy as np

def remove_black_points_from_ply(input_ply_path, output_ply_path):
    # PLY 파일 읽기
    ply_data = PlyData.read(input_ply_path)

    # PLY 파일의 vertices (점) 데이터 가져오기
    vertex_data = ply_data['vertex'].data

    # 각 vertex의 RGB 컬러 정보 추출
    r = vertex_data['red']
    g = vertex_data['green']
    b = vertex_data['blue']

    # 검은색인 포인트 (R=0, G=0, B=0)를 제외한 인덱스 선택
    non_black_indices = np.where((r != 0) | (g != 0) | (b != 0))[0]

    # 검은색이 아닌 포인트들만 필터링
    filtered_vertex_data = vertex_data[non_black_indices]

    # 새로운 PLY 파일 생성
    new_vertex_element = PlyElement.describe(filtered_vertex_data, 'vertex')

    # 수정된 데이터로 새로운 PLY 파일 저장
    PlyData([new_vertex_element], text=True).write(output_ply_path)

    print(f"Filtered PLY saved to {output_ply_path}. Removed {len(vertex_data) - len(filtered_vertex_data)} black points.")

# 사용 예시
input_ply = 'input.ply'
output_ply = 'output_filtered1.ply'

remove_black_points_from_ply(input_ply, output_ply)

