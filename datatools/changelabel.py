import os

# 设置数据集的路径
label_dirs = ['datasets/PB/train/labels', 'datasets/PB/valid/labels', 'datasets/PB/test/labels']

for label_dir in label_dirs:
    for label_file in os.listdir(label_dir):
        # 构建标注文件的完整路径
        label_path = os.path.join(label_dir, label_file)
        # 读取标注文件
        with open(label_path, 'r') as file:
            lines = file.readlines()
        # 修改类别标签从1到0
        with open(label_path, 'w') as file:
            for line in lines:
                parts = line.strip().split()
                if parts[0] == "1":  # 检查类别标签是否为1
                    parts[0] = "0"  # 将类别标签改为0
                # 重新写入修改后的行
                file.write(" ".join(parts) + "\n")

print("已完成标注文件中类别标签的修改。")