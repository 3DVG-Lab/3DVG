import os
import json
import torch
import numpy as np
import open3d as o3d
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import math

class VisualTrainConfig:
    base_dir = "../scannet/scans"
    feature_dim = 256
    batch_size = 32
    lr = 1e-4
    epochs = 30
    temperature = 0.1
    validation_split = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = "./improved_visual_models"
    log_dir = "./visual_logs"

class ScanNetDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        self.label_map = {}
        self.load_all_scenes()
        print(f"加载完成，总实例数: {len(self.data)}")
    def load_all_scenes(self):
        scene_dirs = [d for d in os.listdir(self.config.base_dir)
                      if os.path.isdir(os.path.join(self.config.base_dir, d))]
        print(f"发现场景数: {len(scene_dirs)}")
        for scene_dir in scene_dirs:
            scene_path = os.path.join(self.config.base_dir, scene_dir)
            self.load_scene(scene_path, scene_dir)
    def load_scene(self, scene_path, scene_id):
        try:
            ply_file = next(f for f in os.listdir(scene_path) if f.endswith('.ply'))
            pcd = o3d.io.read_point_cloud(os.path.join(scene_path, ply_file))
            points = np.asarray(pcd.points)
            seg_file = next(f for f in os.listdir(scene_path) if f.endswith('.segs.json'))
            with open(os.path.join(scene_path, seg_file)) as f:
                seg_data = json.load(f)
            seg_indices = np.array(seg_data['segIndices'])
            agg_file = next(f for f in os.listdir(scene_path) if f.endswith('.aggregation.json'))
            with open(os.path.join(scene_path, agg_file)) as f:
                agg_data = json.load(f)
            object_count = 0
            for group in agg_data['segGroups']:
                object_id = f"{scene_id}_{object_count}"
                mask = np.isin(seg_indices, group['segments'])
                if np.sum(mask) < 50:
                    continue
                obj_points = points[mask]
                self.data.append((obj_points, object_id))
                object_count += 1
        except Exception as e:
            print(f"加载场景 {scene_path} 出错: {str(e)}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        points, label = self.data[idx]
        if len(points) >= 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
        else:
            indices = np.random.choice(len(points), 1024, replace=True)
        sampled_points = points[indices]
        sampled_points -= sampled_points.mean(axis=0)
        return torch.FloatTensor(sampled_points), 0
class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.key_conv = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.value_conv = nn.Conv1d(in_channel, in_channel, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        energy = torch.bmm(proj_query.transpose(1, 2), proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x)
        out = torch.bmm(proj_value, attention)
        out = self.gamma * out + x
        return out

class ImprovedPointNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=8,
            in_channel=3,
            mlp=[32, 64]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=32,
            radius=0.4,
            nsample=16,
            in_channel=64 + 3,
            mlp=[64, 128]
        )
        self.local_attention = nn.Sequential(
            nn.Conv1d(128, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 128, 1),
            nn.Softmax(dim=-1)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    def forward(self, x):
        B = x.size(0)
        xyz1, points1 = self.sa1(x, None)
        xyz2, points2 = self.sa2(xyz1, points1)
        attn_weights = self.local_attention(points2)
        attended_points = torch.sum(points2 * attn_weights, dim=-1)
        global_feat = self.global_mlp(attended_points)
        local_feat = self.local_mlp(points2.mean(dim=-1))
        combined = torch.cat([global_feat, local_feat], dim=-1)
        fused = self.fusion(combined)
        return F.normalize(fused, dim=-1)
def square_distance(src, dst):
    B, N, _ = src.shape
    _, S, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, S)
    return dist
def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    C = points.shape[2]
    idx_shape = idx.shape
    idx_flatten = idx.reshape(B, -1)
    idx_expand = idx_flatten.unsqueeze(-1).expand(-1, -1, C)
    points_gathered = torch.gather(points, 1, idx_expand)
    new_points = points_gathered.reshape(*idx_shape, C)
    return new_points
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].reshape(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat([B, S, 1])
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].clone()
    invalid_mask = group_first == N
    group_first[invalid_mask] = 0
    group_first = group_first.unsqueeze(-1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp = nn.Sequential()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp.append(nn.BatchNorm2d(out_channel))
            self.mlp.append(nn.ReLU())
            last_channel = out_channel
    def forward(self, xyz, points):
        if self.group_all:
            if points is not None:
                new_points = torch.cat([xyz.permute(0, 2, 1), points], dim=1)
            else:
                new_points = xyz.permute(0, 2, 1)
            new_points = new_points.unsqueeze(-1)
            new_points = self.mlp(new_points)
            new_points = torch.max(new_points, 2, keepdim=True)[0]
            new_points = new_points.squeeze(-1).squeeze(-1)
            return None, new_points
        else:
            B, N, C = xyz.shape
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            new_points = self._sample_and_group(new_xyz, xyz, points)
            new_points = self.mlp(new_points)
            new_points = torch.max(new_points, -1)[0]
            return new_xyz, new_points
    def _sample_and_group(self, new_xyz, xyz, points):
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(2)
        if points is not None:
            points = points.permute(0, 2, 1)
            grouped_points = index_points(points, idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_points.permute(0, 3, 1, 2)
def data_augmentation(points):
    device = points.device
    B, N, _ = points.shape
    angle = torch.rand(B, device=device) * 2 * math.pi
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zeros = torch.zeros_like(cos)
    ones = torch.ones_like(cos)
    rot_mat = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=1)
    rot_mat = rot_mat.view(B, 3, 3)
    points = torch.bmm(points, rot_mat)
    scale = torch.rand(B, 1, 1, device=device) * 0.4 + 0.8
    points *= scale
    jitter = torch.randn(B, 1, 3, device=device) * 0.02
    points += jitter
    flip_mask = torch.rand(B, 1, 1, device=device) < 0.5
    points = torch.where(flip_mask, points * torch.tensor([[-1, 1, 1]], device=device), points)
    return points
def contrastive_loss(features, temperature):
    batch_size = features.size(0) // 2
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
    features = F.normalize(features, dim=1)
    similarity = torch.mm(features, features.mT) / temperature
    logits_mask = torch.ones_like(similarity) - torch.eye(features.size(0), device=features.device)
    labels = labels * logits_mask
    exp_logits = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    loss = - (labels * log_prob).sum(dim=1) / (labels.sum(dim=1) + 1e-8)
    return loss.mean()
def train_visual_model():
    config = VisualTrainConfig()
    os.makedirs(config.save_dir, exist_ok=True)
    dataset = ScanNetDataset(config)
    train_size = len(dataset) - int(len(dataset) * config.validation_split)
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)
    model = ImprovedPointNet(config.feature_dim).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader))
    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            points = batch[0].to(config.device)
            aug1 = data_augmentation(points)
            aug2 = data_augmentation(points)
            combined = torch.cat([aug1, aug2], dim=0)
            features = model(combined)
            loss = contrastive_loss(features, config.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                points = batch[0].to(config.device)
                aug1 = data_augmentation(points)
                aug2 = data_augmentation(points)
                features = model(torch.cat([aug1, aug2]))
                val_loss += contrastive_loss(features, config.temperature).item()
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pth"))
        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
    print("训练完成，最佳验证损失: {:.4f}".format(best_loss))

def validate_features(model_path):
    config = VisualTrainConfig()
    model = ImprovedPointNet(config.feature_dim).to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dataset = ScanNetDataset(config)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    similarities = []
    with torch.no_grad():
        for points, _ in loader:
            points = points.to(config.device)
            aug1 = data_augmentation(points)
            aug2 = data_augmentation(points)
            feat1 = model(aug1)
            feat2 = model(aug2)
            sim = F.cosine_similarity(feat1, feat2)
            similarities.extend(sim.cpu().numpy())
    print("\n特征质量报告:")
    print("正样本相似度均值: {:.4f} ± {:.4f}".format(np.mean(similarities), np.std(similarities)))
    print("最小/最大相似度: {:.4f}/{:.4f}".format(np.min(similarities), np.max(similarities)))
if __name__ == "__main__":
    train_visual_model()
    validate_features("./improved_visual_models/best_model.pth")