import os
os.environ['OMP_NUM_THREADS'] = '1'
import json
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from typing import Generator
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from openai import OpenAI
import re  # 新增正则表达式模块
def get_color_name(rgb):
    """将 RGB 值映射到颜色名称"""
    # 将 RGB 值转换为 [0, 1] 范围
    rgb = np.clip(rgb, 0, 1)

    # 从 matplotlib 的颜色库中获取预定义的颜色
    css_colors = mcolors.CSS4_COLORS
    color_names = list(css_colors.keys())
    color_values = np.array([mcolors.to_rgb(color) for color in css_colors.values()])

    # 计算与每个预定义颜色的距离
    distances = np.sqrt(np.sum((color_values - rgb) ** 2, axis=1))
    closest_color_index = np.argmin(distances)
    closest_color_name = color_names[closest_color_index]

    return closest_color_name

def get_color_feature(obj_colors):
    """提取物体的颜色特征"""
    # 使用 KMeans 聚类找到主导颜色
    kmeans = KMeans(n_clusters=1, n_init=10).fit(obj_colors)
    dominant_color = kmeans.cluster_centers_[0]  # [R,G,B]
    color_name = get_color_name(dominant_color)
    return dominant_color.tolist(), color_name

def get_bbox_feature(obj_points):
    """提取物体的包围盒特征"""
    # 计算轴对齐包围盒
    min_coords = np.min(obj_points, axis=0)
    max_coords = np.max(obj_points, axis=0)
    size = max_coords - min_coords  # [width, height, depth]
    center = (min_coords + max_coords) / 2
    return size.tolist(), center.tolist()

def detect_metallic(obj_points, obj_colors):
    """通过颜色方差检测金属材质"""
    # 金属通常有高光导致颜色方差较大
    color_std = np.std(obj_colors, axis=0).mean()  # 平均RGB通道标准差
    return "金属" if color_std > 0.15 else "非金属"

def get_material_feature(label):
    """获取物体的材质特征（示例）"""
    # 由于 ScanNet 没有提供直接的材质标签，我们可以根据物体的语义标签猜测材质
    material_dict = {
        "chair": "木质",
        "table": "木质",
        "sofa": "织物",
        "bed": "织物",
        "window": "玻璃",
        "door": "木质",
        "counter": "石材",
        "sink": "陶瓷",
        "cabinet": "木质",
        "shelving": "金属",
        # 可以根据需要添加更多映射
    }
    material = material_dict.get(label.lower(), "未知材质")
    return material

def visualize_object(points, colors, title="物体点云"):
    """可视化单个物体的点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=title)

# 几何复杂度特征
def calculate_geometric_complexity(obj_points):
    """计算点云表面曲率复杂度"""
    from sklearn.neighbors import NearestNeighbors
    # 查找每个点的最近邻
    nbrs = NearestNeighbors(n_neighbors=10).fit(obj_points)
    distances, indices = nbrs.kneighbors(obj_points)

    # 计算局部平面拟合残差
    residuals = []
    for i in range(len(obj_points)):
        neighbors = obj_points[indices[i]]
        centroid = neighbors.mean(axis=0)
        cov = (neighbors - centroid).T @ (neighbors - centroid)
        _, s, _ = np.linalg.svd(cov)
        residuals.append(s[2] / (s[0] + 1e-6))  # 最小特征值占比

    return np.mean(residuals)  # 值越大表面越复杂

# LLM推断物体功能属性
def get_functional_property(label, description):
    """通过LLM推断物体功能属性（新版API调用方式）"""
    client = OpenAI(
        api_key="#",  # 替换为实际API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    prompt = f"""
    根据物体类型'{label}'和场景描述'{description}'，
    推断该物体的功能属性，从以下选项中选择最贴近的, 必须做出选择(最多选五项)然后严格按照输出格式输出(不要输出多余的东西,包括解释说明)：
    [承重, 储物, 支撑, 分隔物理空间, 照明, 信息显示, 可放置物品, 装饰, 娱乐工具, 医疗功能, 文化象征]
    输出格式：逗号分隔, 如: 承重,储物,支撑
    """
    completion = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[
            {"role": "system", "content": "You are inferring about the functional properties of an object"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion.choices[0].message.content.split(",")

def estimate_light_reflectivity(obj_colors):
    """估计物体表面反光特性"""
    # 假设颜色亮度方差反映反光程度
    brightness = obj_colors.mean(axis=0).dot([0.299, 0.587, 0.114])  # RGB转亮度
    brightness_var = np.var(obj_colors.dot([0.299, 0.587, 0.114]))
    return {
        "reflectivity": float(brightness * 0.5 + brightness_var * 0.5),
        "type": "高反光" if brightness > 0.6 else "低反光"
    }

def calculate_spatial_relations(scene_objects, target_obj_id):
    """计算与其他物体的空间关系"""
    target_center = scene_objects[target_obj_id]['bbox_center']
    relations = []

    for obj_id, obj in scene_objects.items():
        if obj_id == target_obj_id:
            continue

        other_center = obj['bbox_center']
        offset = np.array(other_center) - np.array(target_center)
        distance = np.linalg.norm(offset)

        # 方向关系
        direction = ""
        if offset[0] > 1.0:  # X轴方向
            direction += "右侧" if offset[0] > 0 else "左侧"
        if offset[2] > 0.5:  # Z轴（高度）
            direction += "上方" if offset[2] > 0 else "下方"

        relations.append({
            "neighbor_id": obj_id,
            "distance": distance,
            "direction": direction,
            "label": obj['label']
        })

    return sorted(relations, key=lambda x: x['distance'])[:3]  # 返回最近的三个
def process_single_scene(scene_id, data_root, scanrefer_data):
    """处理单个场景的核心逻辑"""
    scene_path = os.path.join(data_root, scene_id)
    if not os.path.exists(scene_path):
        print(f"场景路径不存在：{scene_path}")
        return None

    # ------------ 加载点云数据 ------------
    ply_filename = f"{scene_id}_vh_clean_2.ply"
    ply_path = os.path.join(scene_path, ply_filename)
    if not os.path.exists(ply_path):
        print(f"点云文件不存在：{ply_path}")
        return None

    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        all_points = np.asarray(pcd.points)
        all_colors = np.asarray(pcd.colors)
    except Exception as e:
        print(f"加载点云失败：{str(e)}")
        return None

    # ------------ 加载标注和分割数据 ------------
    agg_filename = f"{scene_id}.aggregation.json"
    agg_path = os.path.join(scene_path, agg_filename)
    seg_filename = f"{scene_id}_vh_clean_2.0.010000.segs.json"
    seg_path = os.path.join(scene_path, seg_filename)

    if not all(os.path.exists(p) for p in [agg_path, seg_path]):
        print(f"缺少标注文件，跳过场景 {scene_id}")
        return None

    try:
        with open(agg_path, 'r') as f:
            agg_data = json.load(f)
        with open(seg_path, 'r') as f:
            segs_data = json.load(f)
        seg_indices = np.array(segs_data["segIndices"])
    except Exception as e:
        print(f"加载标注文件失败：{str(e)}")
        return None

    # ------------ 处理物体属性提取 ------------
    scene_objects = []
    scene_min = np.min(all_points, axis=0)
    scene_max = np.max(all_points, axis=0)
    scene_size = scene_max - scene_min

    for obj in agg_data["segGroups"]:
        obj_id = int(obj["objectId"])
        label = obj["label"]

        target_segments = obj["segments"]
        mask = np.isin(seg_indices, target_segments)
        obj_points = all_points[mask]
        obj_colors = all_colors[mask]

        if len(obj_points) == 0:
            continue

        # 提取各种特征（保持原有逻辑）
        dominant_color, color_name = get_color_feature(obj_colors)
        size, bbox_center = get_bbox_feature(obj_points)
        relative_position = (np.array(bbox_center) - scene_min) / scene_size
        material = get_material_feature(label)
        is_metallic = detect_metallic(obj_points, obj_colors)
        surface_complexity = calculate_geometric_complexity(obj_points)
        reflectivity = estimate_light_reflectivity(obj_colors)

        scene_objects.append({
            "object_id": obj_id,
            "label": label,
            "color": {"rgb": dominant_color, "name": color_name},
            "size": size,
            "bbox_center": bbox_center,
            "material": material,
            "position": relative_position.tolist(),
            "is_metallic": is_metallic,
            "surface_complexity": surface_complexity,
            "reflectivity": reflectivity,
            "points": obj_points.tolist(),
            "colors": obj_colors.tolist()
        })

    # 构建物体字典
    scene_objects_dict = {obj["object_id"]: obj for obj in scene_objects}

    # ------------ 关联ScanRefer描述 ------------
    scene_references = [ann for ann in scanrefer_data if ann["scene_id"] == scene_id]
    cnt = 0
    for ann in scene_references:
        target_obj_id = int(ann["object_id"])
        target_obj = scene_objects_dict.get(target_obj_id)
        cnt = cnt+1
        if target_obj:
            try:
                functional_properties = get_functional_property(
                    target_obj["label"],
                    ann["description"]
                )
                ann["attributes"] = {
                    "label": target_obj["label"],
                    "color": target_obj["color"],
                    "size": target_obj["size"],
                    "bbox_center": target_obj["bbox_center"],
                    "material": target_obj["material"],
                    "is_metallic": target_obj["is_metallic"],
                    "surface_complexity": target_obj["surface_complexity"],
                    "functional_properties": functional_properties,
                    "reflectivity": target_obj["reflectivity"],
                    "position": target_obj["position"]
                }
                print(f"下面进行到第 {cnt} 个物品")
                print(functional_properties)
            except Exception as e:
                print(f"处理{scene_id}的{target_obj_id}时出错：{str(e)}")
                continue

    return scene_references


def main():
    # ------------ 配置路径 ------------
    data_root = "./scannet/scans"
    output_dir = "./val_output"
    os.makedirs(output_dir, exist_ok=True)

    # ------------ 获取所有场景ID ------------
    scene_ids = [d for d in os.listdir(data_root)
                 if os.path.isdir(os.path.join(data_root, d))
                 and re.match(r'scene\d{4}_\d{2}', d)]  # 正则匹配场景格式

    print(f"找到 {len(scene_ids)} 个待处理场景")

    # ------------ 加载ScanRefer数据 ------------
    try:
        with open("ScanRefer_filtered_val.json", 'r') as f:
            scanrefer_data = json.load(f)
        # 预处理object_id为整数
        for ann in scanrefer_data:
            ann["object_id"] = int(ann["object_id"])
        print("ScanRefer数据加载成功，总条目数:", len(scanrefer_data))
    except Exception as e:
        print(f"加载ScanRefer数据失败: {str(e)}")
        return

    # ------------ 遍历处理所有场景 ------------
    for scene_id in scene_ids:
        # 构造输出文件路径
        output_filename = f"{scene_id}_annotated_scanrefer.json"
        output_path = os.path.join(output_dir, output_filename)

        # 检查文件是否已存在
        if os.path.exists(output_path):
            print(f"\n场景 {scene_id} 已处理，跳过...")
            continue

        print(f"\n正在处理场景 {scene_id}...")

        processed_data = process_single_scene(
            scene_id=scene_id,
            data_root=data_root,
            scanrefer_data=scanrefer_data
        )

        if not processed_data:
            continue

        # 保存结果
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"已保存 {len(processed_data)} 条标注到 {output_path}")
        except Exception as e:
            print(f"保存失败: {str(e)}")
            # 删除可能不完整的文件
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == "__main__":
    main()