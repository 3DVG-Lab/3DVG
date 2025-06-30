import os

import numpy as np
from typing import List, Dict
import json
from openai import OpenAI
from tqdm import tqdm


def reset_attributes(temp):
    temp_val = temp.get("label")
    del temp["label"]
    temp["标签"] = temp_val

    temp_val = temp.get("color")
    del temp["color"]
    temp["颜色"] = temp_val

    temp_val = temp.get("material")
    del temp["material"]
    temp["材质"] = temp_val

    temp_val = temp.get("functional_properties")
    del temp["functional_properties"]
    temp["功能"] = temp_val

    temp_val = temp.get("size")
    del temp["size"]
    temp["尺寸"] = temp_val

    temp_val = temp.get("bbox_center")
    del temp["bbox_center"]
    temp["包围盒中心"] = temp_val

    temp_val = temp.get("is_metallic")
    del temp["is_metallic"]
    temp["是否为金属"] = temp_val

    temp_val = temp.get("surface_complexity")
    del temp["surface_complexity"]
    temp["曲率复杂度"] = temp_val

    temp_val = temp.get("reflectivity")
    del temp["reflectivity"]
    temp["反射率"] = temp_val

    temp_val = temp.get("position")
    del temp["position"]
    temp["位置"] = temp_val

    return temp


def reset_attributes2(temp):
    key_mapping = {
        "标签": "label",
        "颜色": "color",
        "材质": "material",
        "功能": "functional_properties",
        "尺寸": "size",
        "包围盒中心": "bbox_center",
        "是否为金属": "is_metallic",
        "曲率复杂度": "surface_complexity",
        "反射率": "reflectivity",
        "位置": "position"
    }
    for ch_key, en_key in key_mapping.items():
        if ch_key in temp:  # 避免KeyError
            temp[en_key] = temp.pop(ch_key)  # 原子操作：删除旧键+插入新键

    return temp


def get_attributes(file):
    attributes = []
    for item in file:
        temp = reset_attributes(item.get('attributes'))
        attributes.append({"object_id": item.get('object_id'), "attributes": temp})
    return attributes


def read_json(i=0, j=1):
    file_path = "scene{:04d}_{:02d}_annotated_scanrefer.json".format(i, j)
    with open(file_path, 'r', encoding='utf-8') as file:
        file = json.load(file)
        attributes = get_attributes(file)
        return attributes


def build_object_prompt(query: str, objects: Dict) -> str:
    object_desc = "\n".join([f"- {k}: {v}" for k, v in objects.get('attributes').items()])

    # return f"""请根据查询需求，为该物体的各个属性分配重要性权重（0-10）
    # 任务要求：
    # 1. 仔细分析用户查询的语义重点
    # 2. 结合特征描述判断该物体各特征与查询的相关性
    # 3. 输出0-10范围的权重值，数值越大表示该特征对定位越重要
    #
    # 查询语句："{query}"
    #
    # 物体属性列表：
    # {object_desc}
    #
    # 请严格按以下JSON格式输出：
    # {{
    #     "reasoning": "分析理由...",
    #     "weights": {{
    #         "attr1": 8, "attr2": 5
    #     }}
    # }}"""
    return f"""根据查询语义重点，为物体所有属性分配检索权重（0-10）
    数值越大表示该特征对查询的物体定位越重要

    查询："{objects.get('description')}"

    待分析属性：
    {object_desc}

    请严格按以下JSON格式输出：
    {{
      "weights": {{
        "颜色":9, "尺寸":7, "材质":3
      }}
    }}
"""


def parse_multiobject_output(response: str, expected_objects: List[str]) -> Dict[str, Dict[str, float]]:
    """
    带完整性校验的多物体权重解析
    """
    try:
        data = json.loads(response.strip())
        weights = data["weights"]

        # 校验完整性
        missing_objects = set(expected_objects) - set(weights.keys())
        if missing_objects:
            raise ValueError(f"缺少物体权重: {missing_objects}")

        # 校验属性完整性
        for obj_id, attrs in weights.items():
            if not isinstance(attrs, dict):
                raise ValueError(f"物体 {obj_id} 权重格式错误")

        return weights

    except Exception as e:
        print(f"解析错误: {str(e)}")
        # 生成默认权重
        return {obj_id: {k: 1.0 for k in obj["attributes"]} for obj in expected_objects}


def batch_softmax(weights: Dict[str, Dict[str, float]], mode: str = "object") -> Dict[str, Dict[str, float]]:
    """
    多层级权重归一化
    Args:
        mode:
            'object' - 每个物体内部属性做softmax
            'global' - 全局所有属性一起做softmax
    """
    if mode == "object":
        return {
            obj_id: {
                k: float(np.exp(v) / sum(np.exp(list(attrs.values()))))
                for k, v in attrs.items()
            }
            for obj_id, attrs in weights.items()
        }
    elif mode == "global":
        all_values = np.array([v for attrs in weights.values() for v in attrs.values()])
        exp_values = np.exp(all_values)
        global_probs = exp_values / exp_values.sum()

        index = 0
        normalized = {}
        for obj_id, attrs in weights.items():
            normalized[obj_id] = {}
            for k in attrs.keys():
                normalized[obj_id][k] = float(global_probs[index])
                index += 1
        return normalized
    else:
        raise ValueError("Invalid normalization mode")

def call_llm_api(prompt: str, i = 0, j = 1) -> str:
    client = OpenAI(
        api_key="sk-08b082a66cd74572a03b42ba2267b82c",  # 替换为实际API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    prompt = prompt
    completion = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[
            {"role": "system", "content": "你正在推断物体的属性权重, 请按照json格式输出。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scene{:04d}_{:02d}_monitor.txt".format(i, j))
    with open(output_path, 'a', encoding='utf-8') as file:
        file.write(completion.choices[0].message.content)
        file.write("\n")
    return completion.choices[0].message.content


def multiobject_weighting(query: str, objects: List[Dict], x = 0, y = 1) -> dict:
    # 生成提示
    prompt = [build_object_prompt(query, object) for object in objects]

    # 调用LLM（示例）
    # jsons = []
    # for _ in prompt:
    #     llm_response = call_llm_api(_)
    #     jsons.append(llm_response)

    # reasons = ""
    # reasons2 = ""
    weights = {}
    for i in tqdm(range(len(prompt))):
        llm_response = json.loads(call_llm_api(prompt[i], x, y))
        weights[objects[i]["object_id"]] = llm_response["weights"]
        # reasons += str(objects[i]["object_id"]) + ' ' + str(llm_response["reasoning"])
        # reasons2 += str(objects[i]["object_id"]) + ' ' + str(llm_response["reasoning"]) + '\n'
    llm_response = f"""{{
    "weights":
        {json.dumps(weights, indent=8, ensure_ascii=False)}
}}"""

    # 解析响应
    expected_ids = [str(obj["object_id"]) for obj in objects]
    raw_weights = parse_multiobject_output(llm_response, expected_ids)

    # 双重归一化
    object_norm = batch_softmax(raw_weights, mode="object")
    global_norm = batch_softmax(raw_weights, mode="global")

    # reasons2 = reasons2.split("\n")[:-1]
    # reasons2 = {reason.split(' ')[0]: reason.split(' ')[1] for reason in reasons2}

    return {
        "object_level_weights": object_norm,
        "global_weights": global_norm,
        "raw_weights": raw_weights,
        "llm_response": llm_response
        # "reasons": reasons2
    }


def work(i=0, j=1):
    objects = read_json(i, j)

    result = multiobject_weighting(
        query="坑位，后面用描述填充该项。",
        objects=objects,
        x=i,
        y=j
    )

    # print("物体级归一化权重：")
    # for obj_id, attrs in result["object_level_weights"].items():
    #     print(f"{obj_id}:")
    #     for attr, w in attrs.items():
    #         print(f"  {attr}: {w:.4f}")
    #
    # print("\n全局归一化权重：")
    # for obj_id, attrs in result["global_weights"].items():
    #     print(f"{obj_id}:")
    #     for attr, w in attrs.items():
    #         print(f"  {attr}: {w:.4f}")

    res = []
    for obj_id, attrs in tqdm(result["object_level_weights"].items()):
        res.append({
            "scene_id": "scene{:04d}_{:02d}_weights.json".format(i, j),
            "object_id": obj_id,
            # "reason": result["reasons"][obj_id],
            "weights": reset_attributes2(result["object_level_weights"][obj_id])
        })

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scene{:04d}_{:02d}_weights.json".format(i, j))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)


# 使用示例
if __name__ == "__main__":
    cnt = 0
    id1 = 1000
    id2 = 10
    # for i in range(id1):
    #     for j in range(id2):
    #         path = "scene{:04d}_{:02d}_annotated_scanrefer.json".format(i, j)
    #         if os.path.exists(path):
    #             cnt += 1
    #             print(f"{cnt} / {551}", "now scene{:04d}_{:02d}_annotated_scanrefer.json".format(i, j))
    #             work(i, j)
    work()