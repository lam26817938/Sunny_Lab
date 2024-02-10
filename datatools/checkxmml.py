import os
import xml.etree.ElementTree as ET

# 指定包含XML文件的資料夾路徑
dataset_directory = 'datasets\PBXML\images'  # 請將這裡改為你的資料夾路徑

def validate_and_delete(file_path):
    try:
        xml_tree = ET.parse(file_path)
        root = xml_tree.getroot()

        assert root.tag == 'annotation' or root.attrib.get('verified') == 'yes', "PASCAL VOC does not contain a root element 'annotation' or attribute 'verified=yes'"
        assert len(root.findtext('filename')) > 0, "XML file does not contain a 'filename'"
        assert len(root.findtext('path')) > 0, "XML file does not contain 'path' element"
        assert root.find('source') is not None and len(root.find('source').findtext('database')) > 0, "XML file does not contain 'source' element with a 'database'"
        assert root.find('size') is not None and list(root.find('size')) == 3, "XML file does not contain 'size' element with three children"
        assert all(x.text.isdigit() for x in root.find('size') if x.tag in ['width', 'height', 'depth']), "XML file's 'size' element does not contain valid 'width', 'height', or 'depth'"
        assert root.find('segmented').text in ['0', '1'], "'segmented' element is not '0' or '1'"
        assert len(root.findall('object')) > 0, "XML file contains no 'object' element"

        required_objects = ['name', 'pose', 'truncated', 'difficult', 'bndbox']
        for obj in root.findall('object'):
            for ro in required_objects[:-1]:  # Exclude 'bndbox' for this check
                assert obj.find(ro) is not None and obj.findtext(ro) != '', f"Object does not contain a parameter '{ro}'"
            assert int(obj.findtext('truncated')) in [0, 1], "Object's 'truncated' is not 0 or 1"
            assert int(obj.findtext('difficult')) in [0, 1], "Object's 'difficult' is not 0 or 1"
            bndbox = obj.find('bndbox')
            assert bndbox is not None, "Object does not contain a parameter 'bndbox'"
            for dim in ['xmin', 'ymin', 'xmax', 'ymax']:
                assert bndbox.find(dim) is not None and bndbox.findtext(dim).isdigit(), f"'{dim}' value for the bounding box is missing or not a digit"

        # 如果執行到這裡，表示檔案格式正確，不需要刪除
        print(f'{os.path.basename(file_path)} is in the correct PASCAL VOC format.')

    except AssertionError as e:
        print(f'Error in {os.path.basename(file_path)}: {e}')
        os.remove(file_path)
        print(f'{os.path.basename(file_path)} has been deleted due to errors.')

# 遍歷資料夾中的所有XML文件
for filename in os.listdir(dataset_directory):
    if filename.endswith('.xml'):
        file_path = os.path.join(dataset_directory, filename)
        validate_and_delete(file_path)

print('Finished checking all files.')