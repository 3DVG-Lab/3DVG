import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from Config import Config
from ScanReferDataset import ScanReferDataset
from Evaluation import calculate_iou
from functools import partial

class CrossModalProjection(nn.Module):
    def __init__(self, visual_dim, text_dim, config):
        super().__init__()
        self.config = config
        self.attribute_meta = config.attribute_meta
        self.attr_projectors = nn.ModuleDict()
        for attr, meta in self.attribute_meta.items():
            if meta['type'] == 'regression':
                proj = nn.Linear(meta['output_dim'], 128)
            elif meta['type'] == 'single':
                proj = nn.Embedding(meta['num_classes'], 128)
            elif meta['type'] == 'multi':
                proj = nn.Linear(meta['num_classes'], 128)
            self.attr_projectors[attr] = proj
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        self.text_fusion = nn.Sequential(
            nn.Linear(text_dim + 128, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))
    def forward(self, visual_feats, text_feats, attributes):
        B, N, _ = visual_feats.shape
        v_proj = self.visual_proj(visual_feats)
        attr_embeds = []
        for attr in self.attr_projectors:
            attr_data = attributes[attr]
            meta = self.attribute_meta[attr]
            if meta['type'] == 'regression':
                attr_embed = self.attr_projectors[attr](attr_data)
            elif meta['type'] == 'single':
                attr_embed = self.attr_projectors[attr](attr_data.long())
            elif meta['type'] == 'multi':
                attr_embed = self.attr_projectors[attr](attr_data.float())
            attr_embeds.append(attr_embed)
        attr_concat = torch.stack(attr_embeds, dim=1).mean(dim=1)
        t_fused = self.text_fusion(torch.cat([text_feats, attr_concat], dim=1))
        sim_matrix = torch.einsum('bnd,bd->bn', v_proj, t_fused)
        return sim_matrix * torch.exp(self.temperature)

class GroundingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        self.cross_modal = CrossModalProjection(256, 256, config)
    def forward(self, batch):
        objects = batch['objects']
        B, N, P, _ = objects.shape
        objects = objects.view(B * N, P, 3).permute(0, 2, 1)
        visual_feats = self.visual_encoder(objects).view(B, N, -1)
        text_feats = self.text_encoder(batch['text_features'])
        logits = self.cross_modal(visual_feats, text_feats, batch['attributes'])
        return logits

def custom_collate(batch, dataset):
    processed_batch = {
        'text_features': torch.stack([item['text_features'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'objects': [],
        'attributes': {attr: [] for attr in dataset.config.attribute_meta},
        'labels': [],
        'sampled_bboxes': [],
        'true_bboxes': torch.stack([item['true_bbox'] for item in batch]),
        'scene_objects': [],
        'sampled_indices': []
    }
    for item in batch:
        scene_id = item['scene_id']
        obj_id = item['object_id']
        scene_objects = dataset.scene_data[scene_id]
        try:
            target_idx = next(i for i, obj in enumerate(scene_objects) if obj['id'] == obj_id)
        except StopIteration:
            raise ValueError(f"未找到对象 {obj_id} 在场景 {scene_id} 中")
        sampled_indices = np.random.choice(
            len(scene_objects),
            dataset.config.max_objects,
            replace=len(scene_objects) < dataset.config.max_objects
        )
        if target_idx not in sampled_indices:
            replace_idx = np.random.randint(0, len(sampled_indices))
            sampled_indices[replace_idx] = target_idx
        np.random.shuffle(sampled_indices)
        new_label = np.where(sampled_indices == target_idx)[0][0]
        processed_batch['labels'].append(new_label)
        processed_batch['sampled_bboxes'].append([obj['bbox'] for obj in scene_objects])
        processed_batch['scene_objects'].append(scene_objects)
        processed_batch['sampled_indices'].append(sampled_indices)
        sampled_points = [torch.from_numpy(scene_objects[i]['points']).float() for i in sampled_indices]
        processed_batch['objects'].append(torch.stack(sampled_points))
        for attr in dataset.config.attribute_meta:
            processed_batch['attributes'][attr].append(item['attributes'][attr])
    processed_batch['objects'] = torch.stack(processed_batch['objects'])
    processed_batch['labels'] = torch.LongTensor(processed_batch['labels'])
    for attr in dataset.config.attribute_meta:
        dtype = torch.long if dataset.config.attribute_meta[attr]['type'] == 'single' else torch.float
        processed_batch['attributes'][attr] = torch.stack(processed_batch['attributes'][attr]).to(dtype)
    return processed_batch

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        inputs = {
            'objects': batch['objects'].to(device),
            'text_features': batch['text_features'].to(device),
            'attributes': {k: v.to(device) for k, v in batch['attributes'].items()}
        }
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
    return avg_loss

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    correct_25 = 0
    correct_50 = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            inputs = {
                'objects': batch['objects'].to(device),
                'text_features': batch['text_features'].to(device),
                'attributes': {k: v.to(device) for k, v in batch['attributes'].items()}
            }
            labels = batch['labels'].numpy()
            logits = model(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            for i in range(len(preds)):
                scene_objects = batch['scene_objects'][i]
                sampled_indices = batch['sampled_indices'][i]
                true_bbox = batch['true_bboxes'][i].numpy()
                pred_idx = sampled_indices[preds[i]]
                pred_bbox = scene_objects[pred_idx]['bbox']
                iou = calculate_iou(pred_bbox, true_bbox)
                if preds[i] == labels[i]:
                    correct += 1
                total += 1
                # 统计IoU阈值准确率
                correct_25 += (iou >= 0.25)
                correct_50 += (iou >= 0.5)
    accuracy = correct / total if total > 0 else 0.0
    acc_25 = correct_25 / total if total > 0 else 0.0
    acc_50 = correct_50 / total if total > 0 else 0.0
    return accuracy, acc_25, acc_50

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = ScanReferDataset(config)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    collate_fn = partial(custom_collate, dataset=full_dataset)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    model = GroundingModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()
    best_val_iou = 0.0
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, acc_25, acc_50 = validate(model, val_loader, device)
        scheduler.step()
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"IoU@0.25: {acc_25:.4f} | IoU@0.5: {acc_50:.4f}")
        if (acc_25 + acc_50) / 2 > best_val_iou:  # 也可以选择用其中一个指标
            best_val_iou = (acc_25 + acc_50) / 2
            torch.save(model.state_dict(), "best_model.pth")
            print(f"保存新最佳模型，IoU@0.25: {acc_25:.4f}, IoU@0.5: {acc_50:.4f}")

if __name__ == "__main__":
    main()