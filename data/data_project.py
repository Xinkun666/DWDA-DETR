import os
import glob
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor,ToPILImage
import torch.nn.functional as F
import numpy as np
import random
random.seed(42)


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        """Initializes Albumentations class for optional data augmentation in YOLOv5 with specified input size."""
        self.transform = None
        try:
            import albumentations as A
            T = [
                A.Blur(p=0.3),
                A.MedianBlur(p=0.3),
                A.ToGray(p=0.3),
                A.CLAHE(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
                A.ImageCompression(quality_lower=75, p=0.3),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=90, p=0.3)
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            pass

    def __call__(self, im, labels, p=0.5):
        """Applies transformations to an image and labels with probability `p`, returning updated image and labels."""
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels

def get_json_files(directory: str) -> List[str]:
    """
    获取指定目录下所有的 JSON 文件路径。

    参数：
    - directory (str): 目录路径字符串。

    返回：
    - List[str]: 包含所有 JSON 文件路径的列表。
    """
    # 存储 JSON 文件名的列表
    json_files = []

    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件名是否以 .json 结尾
            if file.endswith('.json'):
                # 如果是 JSON 文件，则将其加入列表
                json_files.append(file)

    return json_files

def get_jpg_files(directory: str) -> List[str]:
    """
    获取指定目录下所有的 JPG 文件路径。

    参数：
    - directory (str): 目录路径字符串。

    返回：
    - List[str]: 包含所有 JPG 文件路径的列表。
    """
    # 存储 JPG 文件名的列表
    jpg_files = []

    # 遍历指定目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件名是否以 .jpg 结尾
            if file.endswith('.jpg'):
                # 如果是 JPG 文件，则将其加入列表
                jpg_files.append(file)

    return jpg_files

def filter_json_files(json_files,data_path):
    """
    检查 JSON 文件列表中的每个文件，保留包含 'shapes' 属性的文件，删除不包含的文件。

    参数：
    - json_files (List[str]): JSON 文件列表。
    - file_path (Str): JSON 文件的包含目录

    返回：
    - List[str]: 过滤后的 JSON 文件路径列表。
    """
    filtered_files = []
    for json_file in json_files:
        file_path = os.path.join(data_path,json_file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data['shapes'] != []:
                    filtered_files.append(json_file)
                else:
                    print(f'{json_file} is none')
    return filtered_files

def display_images(images):
    """
    显示一张或多张图片。

    :param images: PIL.Image.Image对象或PIL.Image.Image对象的列表
    """
    if not isinstance(images, list):
        # 如果images不是列表，将其转换为单元素列表
        images = [images]

    # 计算子图的布局
    num_images = len(images)
    if num_images == 1:
        # 只有一张图片时
        plt.imshow(images[0])
        plt.axis('off')
    else:
        # 多张图片时，计算合适的网格大小
        cols = int(num_images ** 0.5)
        rows = (num_images + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axs = axs.flatten()  # 将多维的axs数组扁平化处理

        for ax, img in zip(axs, images):
            ax.imshow(img)
            ax.axis('off')

        # 隐藏空余的子图
        for ax in axs[len(images):]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def draw_rectangles(image, coordinates_list, labels,img_id=None,data_path=None,mode=0):
    """
    为一个或多个PIL图像绘制矩形框。

    :param images: PIL图像列表。
    :param coordinates_list: 包含矩形框左上角和右下角坐标的列表，可以是两个点的列表或四个坐标的列表。
    :param labels: 标签列表，用于为不同的矩形框选择不同的颜色。
    """
    # 预定义的颜色列表，足够10个不同的标签
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0)
    ]

    if image == None and img_id != None:
        img_id_str = '{:06d}.jpg'.format(img_id)
        img_path = os.path.join(data_path,img_id_str)
        assert data_path != None,'no data path'
        if mode == 0:
            json_path = os.path.join(data_path,'train',img_id_str.replace('jpg','json'))
        elif mode == 1:
            json_path = os.path.join(data_path, 'val', img_id_str.replace('jgp','json'))
        image = Image.open(img_path)
        if mode != 2:
            with open(json_path, 'r') as f:
                label = json.load(f)
                for l in label['shapes']:
                    box = [l['points'][0][0],l['points'][0][1],l['points'][1][0],l['points'][1][1]]
                    box_label = int(l['label'])
                    coordinates_list.append(box)
                    labels.append(box_label)
        else:
            w,h = image.size
            for i,box in enumerate(coordinates_list):
                coordinates_list[i][:] = [box[0]*w,box[1]*h,box[2]*w,box[3]*h]

    draw = ImageDraw.Draw(image)
    for coordinates, label in zip(coordinates_list, labels):
        # 判断坐标形式并提取x1, y1, x2, y2
        if all(isinstance(coord, list) for coord in coordinates):  # 判断是否为[[x1,y1],[x2,y2]]形式
            (x1, y1), (x2, y2) = coordinates
        elif len(coordinates) == 4:  # 判断是否为[x1,y1,x2,y2]形式
            x1, y1, x2, y2 = coordinates
        else:
            raise ValueError("Coordinates must be in [[x1,y1],[x2,y2]] or [x1,y1,x2,y2] format.")

        # 根据标签选择颜色
        color = predefined_colors[int(label)]

        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    return image

def build_dataset(image_set='train',args=None):
    if image_set=='train':
        dataset = FishData(args,0)
    elif image_set == 'val':
        dataset = FishData(args,1)
    elif image_set == 'test':
        dataset = FishData(args,2)
    return dataset

def rect_to_xywh(rect):
    """
    Convert a rectangle from corner coordinates to center coordinates plus width and height,
    with values rounded to 6 decimal places.

    Args:
    rect (list of lists): The rectangle specified by the coordinates of its top-left and bottom-right corners.
                          Expected format: [[x1, y1], [x2, y2]].

    Returns:
    list: A list containing the x and y coordinates of the rectangle's center, followed by its width and height.
          Each value is rounded to 6 decimal places.
          Format: [x_center, y_center, width, height].
    """
    # Unpack the top-left and bottom-right coordinates
    x1, y1 = rect[0]
    x2, y2 = rect[1]

    # Calculate the center coordinates and round to 6 decimal places
    x_center = round((x1 + x2) / 2, 6)
    y_center = round((y1 + y2) / 2, 6)

    # Calculate width and height and round to 6 decimal places
    width = round(abs(x2 - x1), 6)
    height = round(abs(y2 - y1), 6)

    return [x_center, y_center, width, height]

def xywh_to_rect(xywh):
    """
    将矩形框的中心点坐标和宽高转换为左上角和右下角坐标。

    参数：
    xywh (list): 包含矩形框中心点坐标和宽高的列表，格式为 [x_center, y_center, w, h]。

    返回值：
    list: 包含矩形框左上角和右下角坐标的列表，格式为 [x1, y1, x2, y2]。

    示例：
    >>> rectangle = [3, 4, 2, 3]
    >>> xywh_to_rect(rectangle)
    [2.0, 2.5, 4.0, 5.5]
    """
    x_center, y_center, w, h = xywh
    x1 = round(x_center - w / 2, 6)
    y1 = round(y_center - h / 2, 6)
    x2 = round(x_center + w / 2, 6)
    y2 = round(y_center + h / 2, 6)
    return [x1, y1, x2, y2]

class FishData(Dataset):
    def __init__(self,args=None,mode=0):
        if args != None:
            self.root_path = args.data_path
            if mode == 0:
                self.json_path = os.path.join(self.root_path,'train')
            elif mode == 1:
                self.json_path = os.path.join(self.root_path,'val')
            elif mode == 2:
                self.json_path = None
                self.data_path = args.data_path
                self.data_file = get_jpg_files(self.data_path)
        else :
            self.root_path = '../images/all_fish'
            if mode == 0:
                self.json_path = os.path.join(self.root_path,'train')
            elif mode == 1:
                self.json_path = os.path.join(self.root_path,'val')
            elif mode == 2:
                self.json_path = None
                self.data_path = args.data_path
                self.data_file = get_jpg_files(self.data_path)
        if self.json_path != None:
            self.json_file = get_json_files(self.json_path)
            self.json_file = filter_json_files(self.json_file,self.json_path)
        self.mode = mode
        self.A = Albumentations()

    def __getitem__(self, item):
        if self.mode != 2:
            json_file = self.json_file[item]
            json_path = os.path.join(self.json_path,json_file)

            with open(json_path,'r') as f:
                label = json.load(f)
                #获取图片
                image_file = label['imagePath']
                image_path = os.path.join(self.root_path,image_file)
                image = Image.open(image_path)
                image_copy = image.copy()
                img_id = int(image_file.split(".")[0])
                if self.mode == 0:
                    boxes = []
                    for shape in label['shapes']:
                        box = []
                        box.append(int(shape['label']))
                        box.extend(rect_to_xywh([shape['points'][0],shape['points'][1]]))
                        box[1] = box[1]/label['imageWidth']
                        box[2] = box[2]/label['imageHeight']
                        box[3] = box[3] / label['imageWidth']
                        box[4] = box[4] / label['imageHeight']
                        boxes.append(box)

                    image,boxes = self.A(np.array(image),np.array(boxes))
                    image = Image.fromarray(image)
                    boxes = boxes.tolist()
                    for i,box in enumerate(boxes):
                        box[1] = box[1] * label['imageWidth']
                        box[2] = box[2] * label['imageHeight']
                        box[3] = box[3] * label['imageWidth']
                        box[4] = box[4] * label['imageHeight']
                        x1,y1,x2,y2 = xywh_to_rect(box[1:])
                        label['shapes'][i]['points'][0] = [x1,y1]
                        label['shapes'][i]['points'][1] = [x2,y2]

                    coordinates_list = []
                    labels = []
                    for shape in label['shapes']:
                        labels.append(int(shape['label']))
                        coordinates_list.append(shape['points'])
                    # draw_image = draw_rectangles(image,coordinates_list,labels)
                    # display_images([image_copy,draw_image])
                #数据预处理
                image_trans,label = self.data_transforms(image.copy(),label)

                target = {}
                boxes = []
                labels = []
                area = []
                orig_size = [label['imageWidth'], label['imageHeight']]
                iscrowd = []
                image_id = [img_id]
                size = [label['newimageWidth'], label['newimageHeight']]

                for l in label['shapes']:

                    boxes.append(l['points_resized_norm_xywh'])
                    labels.append(int(l['label']))
                    iscrowd.append(0)
                    area.append((l['points'][1][0]-l['points'][0][0])*(l['points'][1][1]-l['points'][0][1]))

                boxes = torch.tensor(boxes)
                labels = torch.tensor(labels)
                image_id = torch.tensor(image_id)
                area = torch.tensor(area)
                iscrowd = torch.tensor(iscrowd)
                orig_size = torch.tensor(orig_size)
                size = torch.tensor(size)

                target["boxes"] = boxes
                target["labels"] = labels
                target["image_id"] = image_id
                target["area"] = area
                target["iscrowd"] = iscrowd
                target["orig_size"] = orig_size
                target["size"] = size

                return image_trans,target
        else:
            img_file = self.data_file[item]
            img_path = os.path.join(self.data_path,img_file)
            img_id = int(img_file.split('.')[0])
            image = Image.open(img_path)
            image_trans = self.data_transforms(image.copy(), None)
            return image_trans,img_id

    def __len__(self):
        if self.mode !=2:
            return len(self.json_file)
        else:
            return len(self.data_file)

    def data_transforms(self,image,label,resize=224):
        if label != None:
            original_width, original_height = image.size
            # 计算最接近的112整数倍的新宽度和新高度
            new_width = (original_width + resize-1) // resize * resize
            new_height = (original_height + resize-1) // resize * resize
            # 缩放图片
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            image_tensor = ToTensor()(resized_image).unsqueeze(0)
            # 最大池化下采样
            _,_,h,w = image_tensor.size()
            h_ratio,w_ratio = h //resize,w//resize
            kernel_size = (h_ratio,w_ratio)
            stride = (h_ratio,w_ratio)
            pooled_tensor = F.max_pool2d(image_tensor,kernel_size=kernel_size,stride=stride)
            pooled_tensor = torch.squeeze(pooled_tensor)
            pooled_image = ToPILImage()(pooled_tensor.squeeze(0))
            resized_points = []
            resized_norm_points = []
            resized_norm_points_xywh = []
            scaled_x = resize / original_width
            scaled_y = resize / original_height
            # 更新标签检测框坐标
            for shape in label["shapes"]:
                original_point = [shape["points"][0],shape["points"][1]]
                resized_point = [[original_point[0][0]*scaled_x,original_point[0][1]*scaled_y],[original_point[1][0]*scaled_x,original_point[1][1]*scaled_y]]
                resized_norm_point = [[original_point[0][0]*scaled_x/resize,original_point[0][1]*scaled_y/resize],[original_point[1][0]*scaled_x/resize,original_point[1][1]*scaled_y/resize]]
                resized_norm_point_xywh = rect_to_xywh(resized_norm_point)
                resized_points.append(resized_point)
                resized_norm_points.append((resized_norm_point))
                resized_norm_points_xywh.append(resized_norm_point_xywh)

            for index,shape in enumerate(label["shapes"]):
                label["shapes"][index]["points_resized"] = resized_points[index]
                label["shapes"][index]["points_resized_norm"] = resized_norm_points[index]
                label["shapes"][index]["points_resized_norm_xywh"] = resized_norm_points_xywh[index]

            label["newimageWidth"] = resize
            label["newimageHeight"] = resize

            return pooled_tensor,label
        else:
            original_width, original_height = image.size
            # 计算最接近的112整数倍的新宽度和新高度
            new_width = (original_width + resize - 1) // resize * resize
            new_height = (original_height + resize - 1) // resize * resize
            # 缩放图片
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            image_tensor = ToTensor()(resized_image).unsqueeze(0)
            # 最大池化下采样
            _, _, h, w = image_tensor.size()
            h_ratio, w_ratio = h // resize, w // resize
            kernel_size = (h_ratio, w_ratio)
            stride = (h_ratio, w_ratio)
            pooled_tensor = F.max_pool2d(image_tensor, kernel_size=kernel_size, stride=stride)
            pooled_tensor = torch.squeeze(pooled_tensor)
            return pooled_tensor


        # 绘制边界框
        # points = []
        # labels = []
        # for shape in label["shapes"]:
        #     points.append(shape["points"])
        #     labels.append(shape["label"])
        # image_draw = draw_rectangles(image.copy(),points,labels)
        # pooled_image_draw = draw_rectangles(pooled_image.copy(),resized_points,labels)
        #
        # display_images([image_draw,pooled_image_draw])


