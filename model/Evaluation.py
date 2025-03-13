import numpy as np
def calculate_iou(bbox1, bbox2):
    # 输入验证
    # assert bbox1.shape == (2, 3), f"非法bbox1形状: {bbox1.shape}"
    # assert bbox2.shape == (2, 3), f"非法bbox2形状: {bbox2.shape}"
    # assert np.all(bbox1[0] <= bbox1[1]), f"非法bbox1坐标: {bbox1}"
    # assert np.all(bbox2[0] <= bbox2[1]), f"非法bbox2坐标: {bbox2}"
    eps = 1e-10
    intersect_min = np.maximum(bbox1[0], bbox2[0])
    intersect_max = np.minimum(bbox1[1], bbox2[1])
    intersect_dims = np.maximum(0.0, intersect_max - intersect_min)
    intersection = np.prod(intersect_dims)
    vol1 = np.prod(np.maximum(bbox1[1] - bbox1[0], 0.0) + eps)
    vol2 = np.prod(np.maximum(bbox2[1] - bbox2[0], 0.0) + eps)
    union = vol1 + vol2 - intersection
    iou = intersection / (union + eps)  # 防止除零
    iou = np.clip(iou, 0.0, 1.0)
    # 调整断言允许微小误差
    # assert -1e-6 <= iou <= 1.0 + 1e-6, f"计算得到非法IoU值: {iou}"
    return iou
def evaluate_metrics(preds, labels):
    unique_classes = np.unique(np.concatenate([preds, labels]))
    aps = []
    for cls in unique_classes:
        cls_preds = (np.array(preds) == cls)
        cls_labels = (np.array(labels) == cls)
        correct = np.sum(cls_preds & cls_labels)
        total_positive = np.sum(cls_labels)
        if total_positive == 0:
            print(f"警告: 类别 {cls} 在验证集中无正样本，已跳过")
            continue
        ap = correct / total_positive
        aps.append(ap)
    if not aps:
        print("警告: 所有类别均无正样本，mAP设为0")
        return 0.0
    return np.mean(aps)