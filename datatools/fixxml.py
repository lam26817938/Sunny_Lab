import os
import xml.etree.ElementTree as ET

def fix_bounding_boxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    
    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # 修正座標
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax >= width: xmax = width-1
        if ymax >= height: ymax = height-1
        
        bndbox.find('xmin').text = str(xmin)
        bndbox.find('ymin').text = str(ymin)
        bndbox.find('xmax').text = str(xmax)
        bndbox.find('ymax').text = str(ymax)
    
    tree.write(xml_file)  # 直接覆蓋原檔案

def fix_dataset_bounding_boxes(dataset_directory):
    for filename in os.listdir(dataset_directory):
        if filename.endswith('.xml'):
            xml_file = os.path.join(dataset_directory, filename)
            fix_bounding_boxes(xml_file)
            print(f"Fixed bounding boxes in {filename}")

# 使用示例，確保替換'/path/to/your/dataset'為您的資料集資料夾路徑
dataset_directory = 'datasets/PBXML/images'
fix_dataset_bounding_boxes(dataset_directory)