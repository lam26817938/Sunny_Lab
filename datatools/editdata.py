import os

# 设置您的数据集路径
dataset_paths = ['datasets/PB/train/labels', 'datasets/PB/valid/labels', 'datasets/PB/test/labels']

# 要保留的类别ID
class_id_to_keep = 1

for path in dataset_paths:
    for filename in os.listdir(path):
        if filename.endswith('.txt'):  # 确保处理的是标注文件
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            with open(file_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    if class_id == class_id_to_keep:
                        file.write(line)

print("Done")