import torch

class Config:
    seed = 42
    max_length = 64
    pretrained_model = 'bert-base-uncased'
    freeze_bert_layers = 6
    hidden_dropout_prob = 0.1
    use_layer_norm = True
    attribute_meta = {
        'color': {'type': 'regression', 'output_dim': 3},
        'position': {'type': 'regression', 'output_dim': 3},
        'reflectivity': {'type': 'binary'},
        'material': {'type': 'single', 'num_classes': 6},
        'functional_properties': {'type': 'multi', 'num_classes': 27},
        'is_metallic': {'type': 'binary'}
    }
    normalize_stats = {
        'color': {'mean': [0.4239914365384501, 0.37878322154134075, 0.3337124479552827],
                  'std': [0.22587656612433255, 0.21733586974894836, 0.20950749236477106]},
        'position': {'mean': [0.49804436870432706, 0.5049521487704115, 0.32232527579756676],
                     'std': [0.24632094592570444, 0.2510434754244024, 0.16253498342388417]}
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scannet_base = "../scannet/scans"
    scanrefer_train = "../ScanRefer_filtered_train.json"
    scanrefer_val = "../ScanRefer_filtered_val.json"
    visual_feature_dim = 256
    text_feature_dim = 256
    max_objects = 265
    points_per_object = 1024
    batch_size = 24
    lr = 1e-4
    epochs = 80
    weight_decay = 0.01
    visual_model_path = "./improved_visual_models/best_model.pth"
    text_model_path = "./text_models_v2/best_text_model.pth"
    save_dir = "./grounding_models"
    debug_mode = True
    overfit_batch = 5
    weights_dir = "../train_weights"
    weights_suffix = "_weights.json"
    num_classes = 265
    num_colors = 10
    num_materials = 8
    num_positions = 6
    feature_dim = 256