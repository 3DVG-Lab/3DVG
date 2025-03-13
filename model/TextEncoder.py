import os
import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F

class TextTrainConfig:
    annotated_dir = "../val_output"
    weights_dir = "../val_weights"
    annotated_suffix = "_annotated_scanrefer.json"
    weights_suffix = "_weights.json"
    pretrained_model = 'bert-base-uncased'
    feature_dim = 256
    freeze_bert_layers = 6
    hidden_dropout_prob = 0.1
    batch_size = 64
    lr = 2e-5
    epochs = 40
    max_length = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = "./text_models_v4"
    log_dir = "./text_runs_v4"
    def __init__(self):
        self.attribute_meta = {
            'color': {'type': 'regression', 'dim': 3},
            'position': {'type': 'regression', 'dim': 3},
            'reflectivity': {'type': 'binary'},
            'material': {'type': 'single', 'num_classes': 0},
            'functional_properties': {'type': 'multi', 'num_classes': 0},
            'is_metallic': {'type': 'binary'}
        }
        self.normalize_stats = {
            'color': {'mean': [0.4239914365384501, 0.37878322154134075, 0.3337124479552827],
                      'std': [0.22587656612433255, 0.21733586974894836, 0.20950749236477106]},
            'position': {'mean': [0.49804436870432706, 0.5049521487704115, 0.32232527579756676],
                         'std': [0.24632094592570444, 0.2510434754244024, 0.16253498342388417]}
        }
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(vars(self), f)
    @classmethod
    def load(cls, path):
        config = cls()
        with open(path) as f:
            data = json.load(f)
            for k, v in data.items():
                setattr(config, k, v)
        return config

class TextClassificationDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.scene_ids = self._get_scene_ids()
        self.data = self._load_data()
        self._build_mappings()
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    def _get_scene_ids(self):
        return [f.replace(self.config.annotated_suffix, "")
                for f in os.listdir(self.config.annotated_dir)
                if f.endswith(self.config.annotated_suffix)]
    def _load_data(self):
        all_data = []
        for scene_id in self.scene_ids:
            annotated_path = os.path.join(self.config.annotated_dir, f"{scene_id}{self.config.annotated_suffix}")
            weights_path = os.path.join(self.config.weights_dir, f"{scene_id}{self.config.weights_suffix}")
            with open(annotated_path, 'r', encoding='utf-8') as f:
                annotated_data = json.load(f)
            with open(weights_path, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
            all_data.extend([{**ann, 'weights': wt['weights']}
                             for ann, wt in zip(annotated_data, weights_data)])
        return all_data
    def _build_mappings(self):
        self.label_map = {}
        self.attribute_maps = {
            attr: {} for attr, meta in self.config.attribute_meta.items()
            if meta['type'] in ['single', 'multi', 'binary']
        }
        all_labels = set()
        attr_values = {
            attr: set() for attr, meta in self.config.attribute_meta.items()
            if meta['type'] in ['single', 'multi', 'binary']
        }
        self.continuous_stats = {
            attr: {'values': []} for attr, meta in self.config.attribute_meta.items()
            if meta['type'] == 'regression'
        }
        for item in self.data:
            all_labels.add(item['attributes']['label'])
            for attr, meta in self.config.attribute_meta.items():
                value = item['attributes'].get(attr)
                if meta['type'] == 'regression':
                    if attr == 'color':
                        self.continuous_stats[attr]['values'].append(value['rgb'])
                    elif attr == 'position':
                        self.continuous_stats[attr]['values'].append(value)
                elif meta['type'] == 'binary':
                    if attr == 'reflectivity':
                        binary_value = 'high' if value['reflectivity'] > 0.5 else 'low'
                        attr_values[attr].add(binary_value)
                    else:
                        if value is not None:
                            attr_values[attr].add(str(value))
                elif meta['type'] == 'single':
                    if value is not None:
                        attr_values[attr].add(str(value))
                elif meta['type'] == 'multi':
                    if isinstance(value, list):
                        attr_values[attr].update(map(str, value))
        self.label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.config.num_classes = len(self.label_map)
        for attr in attr_values:
            sorted_values = sorted(attr_values[attr])
            self.attribute_maps[attr] = {v: idx for idx, v in enumerate(sorted_values)}
            if self.config.attribute_meta[attr]['type'] == 'single':
                self.config.attribute_meta[attr]['num_classes'] = len(sorted_values)
            elif self.config.attribute_meta[attr]['type'] == 'multi':
                self.config.attribute_meta[attr]['num_classes'] = len(sorted_values)
        for attr in self.continuous_stats:
            values = np.array(self.continuous_stats[attr]['values'])
            self.config.normalize_stats[attr]['mean'] = values.mean(axis=0).tolist()
            self.config.normalize_stats[attr]['std'] = values.std(axis=0).tolist()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item['description'],
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label = self.label_map[item['attributes']['label']]
        attributes = {}
        weights = {}
        for attr, meta in self.config.attribute_meta.items():
            raw_value = item['attributes'].get(attr)
            weight = item['weights'].get(attr, 0.0)
            if meta['type'] == 'regression':
                if attr == 'color':
                    value = raw_value['rgb']
                else:
                    value = raw_value
                stats = self.config.normalize_stats[attr]
                value = (np.array(value) - stats['mean']) / stats['std']
                attributes[attr] = torch.tensor(value, dtype=torch.float32)
                weights[attr] = torch.tensor(weight)
            elif meta['type'] == 'binary':
                if attr == 'reflectivity':
                    binary_value = 'high' if raw_value['reflectivity'] > 0.5 else 'low'
                    value = 1 if binary_value == 'high' else 0
                else:
                    value = 1 if raw_value else 0
                attributes[attr] = torch.tensor(value)
                weights[attr] = torch.tensor(weight)
            elif meta['type'] == 'single':
                value = self.attribute_maps[attr].get(str(raw_value), 0)
                attributes[attr] = torch.tensor(value)
                weights[attr] = torch.tensor(weight)
            elif meta['type'] == 'multi':
                values = [self.attribute_maps[attr][str(v)] for v in raw_value] if raw_value else []
                multi_hot = torch.zeros(self.config.attribute_meta[attr]['num_classes'])
                if values:
                    multi_hot[values] = 1
                attributes[attr] = multi_hot
                weights[attr] = torch.tensor(weight)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label),
            'attributes': attributes,
            'weights': weights
        }

class TextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self._freeze_bert_layers(config.freeze_bert_layers)
        self.feature_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, config.feature_dim),
            nn.LayerNorm(config.feature_dim)
        )
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)
        self.regression_heads = nn.ModuleDict()
        self.discrete_heads = nn.ModuleDict()
        for attr, meta in config.attribute_meta.items():
            if meta['type'] == 'regression':
                self.regression_heads[attr] = nn.Sequential(
                    nn.Linear(config.feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, meta['dim'])
                )
            else:
                if meta['type'] == 'single':
                    self.discrete_heads[attr] = nn.Linear(config.feature_dim, meta['num_classes'])
                elif meta['type'] == 'multi':
                    self.discrete_heads[attr] = nn.Linear(config.feature_dim, meta['num_classes'])
                elif meta['type'] == 'binary':
                    self.discrete_heads[attr] = nn.Linear(config.feature_dim, 1)
        self.task_weights = nn.ParameterDict({
            attr: nn.Parameter(torch.ones(1)) for attr in config.attribute_meta
        })
        self.main_weight = nn.Parameter(torch.ones(1))
    def _freeze_bert_layers(self, num_freeze):
        for layer in self.bert.encoder.layer[:num_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        features = self.feature_projection(pooled)
        features = F.normalize(features, p=2, dim=-1)
        main_logits = self.classifier(features)
        regression_outputs = {}
        discrete_logits = {}
        for attr in self.regression_heads:
            regression_outputs[attr] = self.regression_heads[attr](features)
        for attr in self.discrete_heads:
            logits = self.discrete_heads[attr](features)
            if self.config.attribute_meta[attr]['type'] == 'binary':
                logits = torch.sigmoid(logits)
            discrete_logits[attr] = logits
        return main_logits, discrete_logits, regression_outputs, features, self.main_weight, self.task_weights

class WeightedMultiLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_fns = {
            'main': nn.CrossEntropyLoss(),
            'regression': nn.SmoothL1Loss(),
            'single': nn.CrossEntropyLoss(),
            'multi': nn.BCEWithLogitsLoss(),
            'binary': nn.BCEWithLogitsLoss()
        }
    def forward(self, outputs, targets):
        main_logits, discrete_logits, reg_outputs, _, main_weight, task_weights = outputs
        losses = {}
        main_loss = self.loss_fns['main'](main_logits, targets['label'])
        losses['main'] = main_loss * torch.exp(-main_weight) + main_weight
        for attr in reg_outputs:
            pred = reg_outputs[attr]
            target = targets['attributes'][attr].to(pred.device)
            weight = targets['weights'][attr].mean()
            stats = self.config.normalize_stats[attr]
            device = pred.device
            pred = pred * torch.tensor(stats['std'], device=device) + torch.tensor(stats['mean'], device=device)
            target = target * torch.tensor(stats['std'], device=device) + torch.tensor(stats['mean'], device=device)
            loss = self.loss_fns['regression'](pred, target)
            task_weight = task_weights[attr]
            losses[attr] = loss * weight * torch.exp(-task_weight) + task_weight
        for attr in discrete_logits:
            meta = self.config.attribute_meta[attr]
            weight = targets['weights'][attr].mean()
            if meta['type'] == 'binary':
                loss = self.loss_fns['binary'](
                    discrete_logits[attr].squeeze(),
                    targets['attributes'][attr].float()
                )
            elif meta['type'] == 'single':
                loss = self.loss_fns['single'](
                    discrete_logits[attr],
                    targets['attributes'][attr].long()
                )
            elif meta['type'] == 'multi':
                loss = self.loss_fns['multi'](
                    discrete_logits[attr],
                    targets['attributes'][attr].float()
                )
            task_weight = task_weights[attr]
            losses[attr] = loss * weight * torch.exp(-task_weight) + task_weight
        return sum(losses.values())

def train_text_model():
    config = TextTrainConfig()
    os.makedirs(config.save_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    full_dataset = TextClassificationDataset(config)
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, len(full_dataset) - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    model = TextClassifier(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.epochs)
    criterion = WeightedMultiLoss(config)
    best_val_acc = 0.0
    print("\n动态生成的配置参数：")
    for attr, meta in config.attribute_meta.items():
        if meta['type'] in ['single', 'multi']:
            print(f"{attr}: {meta['num_classes']} classes")
        elif meta['type'] == 'regression':
            print(f"{attr}: {meta['dim']}D regression")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = {k: v.to(config.device) for k, v in batch.items() if k not in ['attributes', 'weights']}
            attributes = {k: v.to(config.device) for k, v in batch['attributes'].items()}
            weights = {k: v.to(config.device) for k, v in batch['weights'].items()}
            optimizer.zero_grad()
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            loss = criterion(outputs, {'label': inputs['label'], 'attributes': attributes, 'weights': weights})
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(config.device) for k, v in batch.items() if k not in ['attributes', 'weights']}
                attributes = {k: v.to(config.device) for k, v in batch['attributes'].items()}
                weights = {k: v.to(config.device) for k, v in batch['weights'].items()}
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                loss = criterion(outputs, {'label': inputs['label'], 'attributes': attributes, 'weights': weights})
                val_loss += loss.item()
                main_logits = outputs[0]
                _, predicted = torch.max(main_logits, 1)
                correct += (predicted == inputs['label']).sum().item()
                total += inputs['label'].size(0)
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pth'))
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("=" * 60)
    writer.close()
    print(f"Training Complete. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_text_model()