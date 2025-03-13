import json
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from pathlib import Path
from typing import Dict, Any
import Config
from transformers import BertModel, BertTokenizer

class ScanReferDataset(Dataset):
    def __init__(self, config: Any):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.bert.eval()
        self.scene_data: Dict[str, list] = {}
        self.valid_indices: list = []
        self.attribute_maps: Dict[str, Dict[str, int]] = {}
        self._build_attribute_meta()
        self._load_annotations()
        self._load_scene_data()
        self._build_attribute_maps()
        self._filter_valid_samples()
    def _build_attribute_meta(self):
        self.attribute_meta = self.config.attribute_meta
        self.continuous_stats = {
            attr: {'values': []}
            for attr, meta in self.attribute_meta.items()
            if meta['type'] == 'regression'
        }
    def _load_annotations(self):
        path = getattr(self.config, "scanrefer_train", self.config.scanrefer_train)
        with open(path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
    def _load_scene_data(self):
        scannet_base = Path(self.config.scannet_base)
        scene_dirs = [d for d in scannet_base.iterdir() if d.is_dir()]
        for scene_dir in tqdm(scene_dirs, desc="加载场景数据"):
            try:
                objects = self._process_scene(scene_dir.name)
                if objects:
                    self.scene_data[scene_dir.name] = objects
                    self._collect_continuous_stats(objects)
            except Exception as e:
                print(f"场景 {scene_dir.name} 加载失败: {str(e)}")
        for attr in self.continuous_stats:
            values = np.array(self.continuous_stats[attr]['values'])
            self.config.normalize_stats[attr] = {
                'mean': values.mean(axis=0).tolist(),
                'std': values.std(axis=0).tolist()
            }
    def _process_scene(self, scene_id: str) -> list:
        scene_path = Path(self.config.scannet_base) / scene_id
        objects = []
        try:
            pcd = o3d.io.read_point_cloud(str(scene_path / f"{scene_id}_vh_clean_2.ply"))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            with open(scene_path / f"{scene_id}_vh_clean_2.0.010000.segs.json", 'r') as f:
                seg_data = json.load(f)
            with open(scene_path / f"{scene_id}.aggregation.json", 'r') as f:
                agg_data = json.load(f)
            segments = np.array(seg_data["segIndices"])
            seg_dict = {}
            for idx, seg_id in enumerate(segments):
                seg_dict.setdefault(seg_id, []).append(idx)
            for group in agg_data["segGroups"]:
                obj_id = str(group["objectId"])
                mask = np.zeros(len(segments), dtype=bool)
                for seg_id in group["segments"]:
                    if seg_id in seg_dict:
                        mask[seg_dict[seg_id]] = True
                if np.sum(mask) < 50:
                    continue
                obj_points = points[mask]
                if len(obj_points) == 0:
                    continue
                centroid = np.mean(obj_points, axis=0)
                avg_color = np.mean(colors[mask], axis=0)
                objects.append({
                    "id": obj_id,
                    "points": self._sample_points(obj_points - centroid),
                    "centroid": centroid,
                    "color": avg_color,
                    "bbox": np.array([np.min(obj_points, axis=0), np.max(obj_points, axis=0)])
                })
        except Exception as e:
            print(f"处理场景 {scene_id} 出错: {str(e)}")
            return []
        return objects
    def _collect_continuous_stats(self, objects: list):
        for obj in objects:
            if 'color' in self.continuous_stats:
                self.continuous_stats['color']['values'].append(obj['color'])
            if 'position' in self.continuous_stats:
                self.continuous_stats['position']['values'].append(obj['centroid'])
    def _build_attribute_maps(self):
        attr_values = {attr: set() for attr in self.attribute_meta}
        for ann in tqdm(self.annotations, desc="构建属性映射"):
            attributes = self._parse_attributes(ann)
            for attr in self.attribute_meta:
                value = attributes.get(attr)
                if value is None:
                    continue
                meta = self.attribute_meta[attr]
                if meta['type'] == 'single':
                    attr_values[attr].add(str(value))
                elif meta['type'] == 'multi':
                    attr_values[attr].update(map(str, value))
                elif meta['type'] == 'binary':
                    attr_values[attr].update(['True', 'False'])
        for attr in self.attribute_meta:
            sorted_values = sorted(attr_values[attr])
            self.attribute_maps[attr] = {v: idx for idx, v in enumerate(sorted_values)}
            self.attribute_meta[attr]["num_classes"] = len(sorted_values)
    def _parse_attributes(self, ann: dict) -> dict:
        scene_id = ann["scene_id"]
        obj_id = str(ann["object_id"])
        text = ann["description"].lower()
        attributes = {}
        target_obj = next((
            o for o in self.scene_data.get(scene_id, [])
            if o["id"] == obj_id
        ), None)
        if target_obj:
            attributes['color'] = target_obj['color'].tolist()
            attributes['position'] = target_obj['centroid'].tolist()
            attributes['is_metallic'] = any(kw in text for kw in ['metal', 'steel'])
            attributes['reflectivity'] = 0.8 if attributes['is_metallic'] else 0.2  # 直接存储数值
        attributes.update(self._parse_discrete_attributes(text))
        return attributes
    def _parse_discrete_attributes(self, text: str) -> dict:
        return {
            'material': self._parse_material(text),
            'functional_properties': self._parse_functional(text)
        }
    def _parse_material(self, text: str) -> str:
        materials = ['wood', 'plastic', 'metal', 'fabric', 'glass']
        return next((m for m in materials if m in text), 'unknown')
    def _parse_functional(self, text: str) -> list:
        return [p for p in ['movable', 'adjustable', 'stackable'] if p in text]
    def _encode_attributes(self, attributes: dict) -> dict:
        encoded = {}
        for attr, meta in self.attribute_meta.items():
            value = attributes.get(attr)
            if meta['type'] == 'regression':
                stats = self.config.normalize_stats[attr]
                norm_value = (np.array(value) - stats['mean']) / stats['std']
                encoded[attr] = torch.tensor(norm_value, dtype=torch.float32)
            elif meta['type'] == 'binary':
                encoded[attr] = torch.tensor(float(value), dtype=torch.float32)
            elif meta['type'] == 'single':
                idx = self.attribute_maps[attr].get(str(value), 0)
                encoded[attr] = torch.tensor(idx, dtype=torch.long)
            elif meta['type'] == 'multi':
                indices = [self.attribute_maps[attr][str(v)] for v in value]
                multi_hot = torch.zeros(meta['num_classes'])
                multi_hot[indices] = 1
                encoded[attr] = multi_hot
        return encoded
    def _filter_valid_samples(self):
        self.valid_indices = [
            idx for idx, ann in enumerate(self.annotations)
            if ann["scene_id"] in self.scene_data and
               any(obj["id"] == str(ann["object_id"])
                   for obj in self.scene_data[ann["scene_id"]])
        ]
    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        num_points = self.config.points_per_object
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
        else:
            indices = np.random.choice(len(points), num_points, replace=True)
        return points[indices].astype(np.float32)
    def __len__(self) -> int:
        return len(self.valid_indices)
    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[self.valid_indices[idx]]
        scene_objects = self.scene_data[ann["scene_id"]]
        target_obj = next(obj for obj in scene_objects if obj['id'] == str(ann["object_id"]))
        true_bbox = target_obj['bbox']
        inputs = self.tokenizer(
            ann["description"],
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.bert(**inputs)
            text_features = outputs.last_hidden_state.mean(dim=1)
        attributes = self._parse_attributes(ann)
        encoded_attrs = self._encode_attributes(attributes)
        scene_objects = self.scene_data[ann["scene_id"]]
        target_idx = next(i for i, obj in enumerate(scene_objects) if obj['id'] == str(ann["object_id"]))
        return {
            'scene_id': ann["scene_id"],
            'object_id': str(ann["object_id"]),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'text_features': text_features.squeeze(0),
            'attributes': encoded_attrs,
            'labels': torch.tensor(target_idx, dtype=torch.long),
            'true_bbox': torch.from_numpy(true_bbox).float()
        }

if __name__ == '__main__':
    dataset = ScanReferDataset(Config)
    sample = dataset[0]
    print("样本结构:")
    print(f"输入ID形状: {sample['input_ids'].shape}")
    print(f"颜色属性: {sample['attributes']['color']}")
    print(f"材质属性: {sample['attributes'].get('material', 'N/A')}")