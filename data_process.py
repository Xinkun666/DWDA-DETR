import json
import os
import shutil


def json_to_txt(json_dir, txt_dir):
    # 确保输出目录存在
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    # 遍历json_dir目录下的所有json文件
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            # 构建完整的json文件路径
            json_path = os.path.join(json_dir, json_file)
            # 读取json文件内容
            with open(json_path, 'r') as file:
                data = json.load(file)

            # 构建对应的txt文件路径
            txt_path = os.path.join(txt_dir, os.path.splitext(json_file)[0] + '.txt')

            # 打开txt文件准备写入
            with open(txt_path, 'w') as file:
                # 遍历shapes中的每个元素
                for shape in data['shapes']:
                    label = shape['label']
                    # 提取points属性的第一个和第二个元素
                    (x1, y1), (x2, y2) = shape['points']
                    # 写入到txt文件
                    file.write(f'{label} {x1} {y1} {x2} {y2}\n')

def update_json_coordinates(txt_dir, json_dir):
    """
    更新JSON文件中的坐标信息，基于对应的TXT文件中的坐标数据。

    参数:
    txt_dir (str): 包含TXT文件的目录路径。
    json_dir (str): 包含JSON文件的目录路径，这些JSON文件应该与TXT文件对应。

    每个TXT文件中的行应该包含 [标签、置信度、x1,y1,x2,y2]，x1,y1表示矩形框的左上角点坐标，
    x2,y2表示右下角点坐标。每个JSON文件中的shapes属性应该包含与TXT文件中的标签对应的标签，
    并将坐标信息更新为TXT文件中的坐标。
    """

    # 遍历txt目录中的所有文件
    for txt_filename in os.listdir(txt_dir):
        if txt_filename.endswith('.txt'):
            # 构建对应的json文件路径
            json_filename = txt_filename.replace('.txt', '.json')
            json_path = os.path.join(json_dir, json_filename)

            # 读取txt文件
            txt_path = os.path.join(txt_dir, txt_filename)
            with open(txt_path, 'r') as f:
                txt_lines = f.readlines()

            # 读取json文件
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                # 遍历txt文件中的每一行
                for line in txt_lines:
                    parts = line.strip().split(' ')
                    label, x1, y1, x2, y2 = parts[0], parts[2], parts[3], parts[4], parts[5]

                    # 在json的shapes中找到对应的label并替换坐标
                    for shape in json_data['shapes']:
                        if shape['label'] == label:
                            shape['points'][0] = [float(x1), float(y1)]
                            shape['points'][1] = [float(x2), float(y2)]

                # 将修改后的json数据保存回文件
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)


def convert_and_update_txt(txt_dir, json_dir, output_dir):
    """
    根据对应的JSON文件中的图像尺寸，转换TXT文件中的坐标，并保存到新的目录。

    参数:
    txt_dir (str): 原始TXT文件的目录路径。
    json_dir (str): 对应JSON文件的目录路径。
    output_dir (str): 修改后的TXT文件保存的目录路径。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历TXT文件
    for txt_filename in os.listdir(txt_dir):
        if txt_filename.endswith('.txt'):
            txt_path = os.path.join(txt_dir, txt_filename)
            json_path = os.path.join(json_dir, txt_filename.replace('.txt', '.json'))
            output_path = os.path.join(output_dir, txt_filename)

            # 读取JSON文件以获取图片尺寸
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                img_width = json_data['imageWidth']
                img_height = json_data['imageHeight']

            # 读取并处理TXT文件
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            with open(output_path, 'w') as f_out:
                for line in lines:
                    parts = line.strip().split(' ')
                    label, x, y, w, h, confidence = int(parts[0]), float(parts[1]), float(parts[2]), float(
                        parts[3]), float(parts[4]), float(parts[5])
                    # 坐标转换
                    # label += 1  # 标签加1
                    x1 = (x - w / 2) * img_width
                    y1 = (y - h / 2) * img_height
                    x2 = (x + w / 2) * img_width
                    y2 = (y + h / 2) * img_height
                    # 保存修改后的行到新文件
                    f_out.write(f"{label} {confidence} {x1} {y1} {x2} {y2}\n")


def process_json_files(src_dir, dest_txt_dir, dest_img_dir, img_src_dir):
    if not os.path.exists(dest_txt_dir):
        os.makedirs(dest_txt_dir)
    if not os.path.exists(dest_img_dir):
        os.makedirs(dest_img_dir)

    for json_file in os.listdir(src_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(src_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_width, image_height = data['imageWidth'], data['imageHeight']
                shapes = data['shapes']

                txt_filename = json_file.replace('.json', '.txt')
                txt_path = os.path.join(dest_txt_dir, txt_filename)

                with open(txt_path, 'w') as txt_f:
                    for shape in shapes:
                        label = shape['label']
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        x_center = (x1 + x2) / 2 / image_width
                        y_center = (y1 + y2) / 2 / image_height
                        width = abs(x2 - x1) / image_width
                        height = abs(y2 - y1) / image_height
                        txt_f.write(f"{label} {x_center} {y_center} {width} {height}\n")

            # Copy corresponding image file
            img_filename = data['imagePath']
            src_img_path = os.path.join(img_src_dir, img_filename)
            dest_img_path = os.path.join(dest_img_dir, img_filename)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dest_img_path)


def process_json_files2(json_dir):
    # 遍历dir/json目录下的所有json文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_file_path = os.path.join(json_dir, filename)

            # 读取json文件内容
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            imageWidth = data['imageWidth']
            imageHeight = data['imageHeight']
            # 遍历shapes列表
            for h,shape in enumerate(data['shapes']):
                # 遍历points列表
                for i, point in enumerate(shape['points']):
                    # 如果点的任何坐标值为负数，将其置为0
                    for j, coord in enumerate(point):
                        if coord < 0:
                            data['shapes'][h]['points'][i][j] = 0
                        if j == 0 and coord > imageWidth:
                            data['shapes'][h]['points'][i][j] = imageWidth
                        elif j == 1 and coord > imageHeight:
                            data['shapes'][h]['points'][i][j] = imageHeight

            # 保存修改后的json文件内容
            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=4)

def get_max_map(log_file):
    max_map = float('-inf')
    epoch = 0
    # 读取log文件内容
    with open(log_file, 'r') as file:
        for line in file:
            # 解析json格式的每一行
            data = json.loads(line)
            # 获取当前epoch的map值
            epoch_map = data.get('map', 0)
            # 更新最大map值
            if epoch_map > max_map:
                max_map = epoch_map
                epoch = data.get('epoch')

    return max_map,epoch


def copy_jpg_to_val_img(json_dir, img_dir, val_img_dir):
    """
    将val目录下所有的json文件对应的jpg文件复制到val_img目录中。

    参数：
    json_dir (str): 包含json文件的目录路径。
    img_dir (str): 包含jpg文件的目录路径。
    val_img_dir (str): 将jpg文件复制到的目标目录路径。
    """
    # 确保目标目录存在
    os.makedirs(val_img_dir, exist_ok=True)

    # 遍历val目录下的所有json文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            # 获取json文件名（不含扩展名）
            json_filename = os.path.splitext(filename)[0]

            # 构建对应的jpg文件路径
            img_filename = json_filename + '.jpg'
            img_path = os.path.join(img_dir, img_filename)

            # 如果jpg文件存在，则复制到val_img目录下
            if os.path.exists(img_path):
                shutil.copy(img_path, val_img_dir)

def convert_to_coco_format(json_dir, image_dir, output_file):
    """
    将给定目录下的JSON标注文件转换为COCO数据集格式的JSON文件。

    参数：
    json_dir (str): 包含JSON标注文件的目录路径。
    image_dir (str): 包含图像文件的目录路径。
    output_file (str): 输出的COCO数据集格式的JSON文件路径。
    """
    # 初始化Coco数据集格式字典
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加categories信息（这里假设已知的类别标签为1和2）
    coco_data["categories"].append({"id": 1, "name": "head", "supercategory": "object"})
    coco_data["categories"].append({"id": 2, "name": "tail", "supercategory": "object"})

    # 遍历json目录下的所有json文件
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_file_path = os.path.join(json_dir, filename)

            # 读取json文件内容
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # 添加image信息
            image_info = {
                "file_name": data["imagePath"],
                "height": data["imageHeight"],
                "width": data["imageWidth"],
                "id": int(os.path.splitext(filename)[0])
            }
            coco_data["images"].append(image_info)

            # 添加annotation信息
            for shape in data["shapes"]:
                bbox = [min(shape["points"][0][0], shape["points"][1][0]),
                        min(shape["points"][0][1], shape["points"][1][1]),
                        abs(shape["points"][1][0] - shape["points"][0][0]),
                        abs(shape["points"][1][1] - shape["points"][0][1])]

                annotation = {
                    "segmentation": [],
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "image_id": image_info["id"],
                    "bbox": bbox,
                    "category_id": int(shape["label"]),
                    "id": len(coco_data["annotations"]) + 1
                }
                coco_data["annotations"].append(annotation)

    # 将数据集保存为JSON文件
    with open(output_file, 'w') as outfile:
        json.dump(coco_data, outfile, indent=4)







if __name__ == '__main__':
    # json_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish/val'
    # txt_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/groundtruths'
    # json_to_txt(json_dir=json_dir,txt_dir=txt_dir)

    # txt_dir = '/Users/liuxinkun/all_projects/PycharmProjects/yolov5/runs/detect/exp6/labels'
    # output_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/yolov5l_detect'
    # json_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish/val'
    # convert_and_update_txt(txt_dir=txt_dir,output_dir=output_dir,json_dir=json_dir)

    # Process train and val json files
    # src_train_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish/train'
    # src_val_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish/val'
    # img_src_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish'
    # train_txt_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/yolov/labels/train'
    # val_txt_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/yolov/labels/val'
    # train_img_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/yolov/images/train'
    # val_img_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/yolov/images/val'
    #
    # process_json_files(src_train_dir, train_txt_dir, train_img_dir, img_src_dir)
    # process_json_files(src_val_dir, val_txt_dir, val_img_dir, img_src_dir)

    # 调用函数
    # process_json_files2("/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish/val")

    # 读取log文件并获取最大map值
    # log_file = '/Users/liuxinkun/Downloads/log (3).txt'
    # max_map,epoch = get_max_map(log_file)
    #
    # print(f"最大的map值为:{max_map},epoch:{epoch}")

    # 调用函数
    json_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish/train'
    img_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish'
    val_img_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/coco_fish/train2017'

    copy_jpg_to_val_img(json_dir, img_dir, val_img_dir)

    # 调用函数
    # json_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish/val'
    # image_dir = '/Users/liuxinkun/Downloads/fishdata/fish_data/all_fish'
    # output_file = '/Users/liuxinkun/Downloads/fishdata/fish_data/instances_val2017.json'
    #
    # convert_to_coco_format(json_dir, image_dir, output_file)