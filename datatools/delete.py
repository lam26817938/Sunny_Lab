import os

# 数据集的路径
dataset_paths = {
    'train': 'datasets/PB/train/',
    'val': 'datasets/PB/valid/',
    'test': 'datasets/PB/test/'  
}

def is_file_empty(file_path):
    """检查文件是否为空或仅包含glass-bottle标注"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if not lines:  # 文件为空
            return True
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id != 0:  # 如果存在非glass-bottle的标注，则文件不为空
                return False
    return True  # 所有标注都是glass-bottle或文件实际为空

# 遍历数据集目录
for dataset_type, path in dataset_paths.items():
    labels_path = os.path.join(path, 'labels')
    images_path = os.path.join(path, 'images')
    
    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        if is_file_empty(label_file_path):
            os.remove(label_file_path)  # 删除标注文件
            # 构造并删除对应的图片文件
            image_file = label_file.replace('.txt', '.jpg')  # 假设图片格式为jpg，如果不是，请相应修改
            image_file_path = os.path.join(images_path, image_file)
            if os.path.exists(image_file_path):
                os.remove(image_file_path)
            print(f"Deleted {label_file_path} and {image_file_path}")

print("完成删除操作。")