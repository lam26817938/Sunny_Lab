import os
import xml.etree.ElementTree as ET
import shutil  


dataset_directory = 'datasets\PBXML\images'


for filename in os.listdir(dataset_directory):
    if filename.endswith('.xml'):
        file_path = os.path.join(dataset_directory, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 標記是否找到指定標注
        found_plastic_bottle = False
        
        # 收集所有object元素以便之後可能刪除
        objects_to_remove = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label != 'plastic-bottle':
                objects_to_remove.append(obj)
            else:
                found_plastic_bottle = True
        
        # 如果找到非plastic-bottle標注，則刪除這些標注
        if found_plastic_bottle and objects_to_remove:
            for obj in objects_to_remove:
                root.remove(obj)
            tree.write(file_path)  # 寫回修改後的XML到檔案
            print(f'Updated {file_path} to only include plastic-bottle annotations.')
        elif not found_plastic_bottle:
            # 如果檔案中沒有plastic-bottle標注，可以選擇刪除或保留檔案
            os.remove(file_path)  # 如果選擇刪除檔案，取消這行註釋
            print(f'No plastic-bottle found in {file_path}, file untouched.')