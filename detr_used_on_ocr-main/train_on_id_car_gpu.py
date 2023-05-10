#===========第一步刷数据.把voc数据刷成我们要的
import sys
epoch=100 # 至少是1.
batch_size = 3
sys.path.append("..")
import xml.etree.ElementTree as ET
# coding=utf-8
# project
import os.path as osp
# PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
#
# DATA_PATH = osp.join(PROJECT_PATH, 'data')
# DATA_PATH = "C:/Users/Administrator/PycharmProjects/untitled1/VOCdevkit/VOC_idcard"
DATA_PATH = "/ext/ocr/VOCdevkit/VOC_idcard"



import os

# print(os.name) # 以后的全部代码是用这个来进行 平台兼容性. 也就是自己电脑是nt, 服务器是posix
# if os.name!='nt':
#     a=(os.path.dirname(os.path.dirname(__file__))+'/ocr_business_license')
#     DATA_PATH=a
# print(DATA_PATH,33333333333)
MODEL_TYPE = {
    "TYPE": "Mobilenet-YOLOv4"
}  # YOLO type:YOLOv4, Mobilenet-YOLOv4 or Mobilenetv3-YOLOv4

CONV_TYPE = {"TYPE": "DO_CONV"}  # conv type:DO_CONV or GENERAL
#是否加注意力.
ATTENTION = {"TYPE": "NONE"}  # attention type:SEnet、CBAM or NONE

# train
TRAIN = {
    "DATA_TYPE": "VOC",  # DATA_TYPE: VOC ,COCO or Customer
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 1,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "YOLO_EPOCHS": 50,
    "Mobilenet_YOLO_EPOCHS": 120,
    "NUMBER_WORKERS": 0,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2,  # or None
    "showatt": False
}


# val
VAL = {
    "TEST_IMG_SIZE": 416,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.7,           # 干鸡毛呢这么低.
    "NMS_THRESH": 0.45,
    "MULTI_SCALE_VAL": False,
    "FLIP_VAL": False,
    "Visual": False,
    "showatt": False
}

Customer_DATA = {
    "NUM": 3,  # your dataset number
    "CLASSES": ["unknown", "person", "car"],  # your dataset class
}

VOC_DATA = {
    "NUM": 20,
    "CLASSES": [
        "life",
        "name",
        "idn",
        "front",
        "back",

    ],
}
#重新自动刷新数量
VOC_DATA["NUM"]=len(VOC_DATA["CLASSES"])

import os
from tqdm import tqdm


def parse_voc_annotation(
    data_path, file_type, anno_path, use_difficult_bbox=False
):
    """
    phase pascal voc annotation, eg:[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: eg: VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: eg: 'trainval''train''val'
    :param anno_path: path to ann file
    :param use_difficult_bbox: whither use different sample
    :return: batch size of data set
    """
    if TRAIN["DATA_TYPE"] == "VOC":
        classes = VOC_DATA["CLASSES"]
    # elif TRAIN["DATA_TYPE"] == "COCO":
    #     classes = COCO_DATA["CLASSES"]
    else:
        classes = Customer_DATA["CLASSES"]
    # img_inds_file = os.path.join(
    #     data_path, "ImageSets", "Main", file_type + ".txt"
    # )
    # with open(img_inds_file, "r") as f:
    #     lines = f.readlines()
    #     image_ids = [line.strip() for line in lines]
    import glob
    image_path = os.path.join(  # 找id对应的图片
        data_path, "JPEGImages"
    )
    train_list = glob.glob(os.path.join(image_path, '*.jpg'))+glob.glob(os.path.join(image_path, '*.png'))

    image_ids=train_list
    with open(anno_path, "w") as f:
        for image_id in tqdm(image_ids):
            new_str = ''
            # image_path = os.path.join(   # 找id对应的图片
            #     data_path, "JPEGImages", image_id + ".jpg"
            # )
            image_path=image_id
            annotation = image_path
            label_path = os.path.join(  # 找id对应的xml
                data_path, "Annotations", image_id + ".xml"
            )
            label_path=image_path.replace("JPEGImages","Annotations").replace("jpg","xml")
            root = ET.parse(label_path).getroot() # 解析一个xml文件.
            objects = root.findall("object")# objects是当前图片里面所有的物体.
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                # if (not use_difficult_bbox) and (
                #     int(difficult) == 1
                # ):  # difficult 表示是否容易识别，0表示容易，1表示困难
                #     continue
                bbox = obj.find("bndbox")
                class_id = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find("xmin").text.strip()
                ymin = bbox.find("ymin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymax = bbox.find("ymax").text.strip()
                new_str += " " + ",".join(
                    [xmin, ymin, xmax, ymax, str(class_id)] #这个地方要记住物理含义!!!!!!!!!!!!!!!!
                )
            if new_str == '':
                continue
            annotation += new_str
            annotation += "\n"
            # print(annotation)
            f.write(annotation)
    return len(image_ids)
import os
if not os.path.exists('train_annotation.txt'):
    if __name__ == "__main__":
        # train_set :  VOC2007_trainval 和 VOC2012_trainval
        train_data_path_2007 = os.path.join(
            DATA_PATH
        )
        # train_data_path_2012 = os.path.join(
        #     cfg.DATA_PATH, "voc_test_data", "VOCdevkit", "VOC2012"
        # )
        # 新建一个标签txt
        train_annotation_path = os.path.join( "train_annotation.txt")
        if os.path.exists(train_annotation_path):
            os.remove(train_annotation_path)

        # # val_set   : VOC2007_test
        # test_data_path_2007 = os.path.join(
        #     DATA_PATH, "voc_test_data", "VOCdevkit", "VOC2007"
        # )
        # test_annotation_path = os.path.join("../data", "test_annotation.txt")
        # if os.path.exists(test_annotation_path):
        #     os.remove(test_annotation_path)

        len_train = parse_voc_annotation(
            train_data_path_2007,
            "trainval",
            train_annotation_path,
            use_difficult_bbox=False,
        )
        # + parse_voc_annotation(
        #     train_data_path_2012,
        #     "trainval",
        #     train_annotation_path,
        #     use_difficult_bbox=False,
        # )
        # len_test = parse_voc_annotation(
        #     test_data_path_2007,
        #     "test",
        #     test_annotation_path,
        #     use_difficult_bbox=False,
        # )

        print(
            "The number of images for train and test are :train : {0} | test : {0}".format(
                len_train
            )
        )
print(1)



anno_path="train_annotation.txt"




















from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests
#============2021-06-18,16点39  先写到这里, 先去搞语音.回来再仔细debug一遍.


import os

import torch
from torch.utils.data  import  Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import numpy as np
from transformers import BertTokenizer
import linecache

from torch.utils.data import DataLoader, SubsetRandomSampler

print("1111111111111111")

class DealDataset2(Dataset):

    def __init__(self):


        with open(anno_path, "r") as f:
            self.annotations = list(filter(lambda x: len(x) > 0, f.readlines()))

    def __getitem__(self, index):
        a=self.annotations[index].strip()
        a=a.split(' ',1) # 只切分一次.
        pic=a[0]
        anno=a[1]
        #Qquestion = '时间'

        return pic,anno

    def __len__(self):
        return len(self.annotations)





# 下面是true_shuju:
dealDataset = DealDataset2()

train_loader = DataLoader(dataset=dealDataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,
                          )


#
# url = '0001.jpg'
# image = Image.open(url)

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50') # 这个加载的是预处理的配置文件

#==========改变分类数量.======模型名字永远不要动,后面配自己的参数,经过自定义即可改变网络结构.
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50',config='config_for_id_card.json')
# coco:box: "bbox": [x,y,width,height],左上角和长款.

try:
    model=torch.load('1.pth')
    print("加载了模型1.pth")
except:
    print("没有加载模型")
#==========输入xmin,xmax,ymin,ymax

model.cuda()


#=============写train
class A():
    pass


args = A()
args.learning_rate = 3e-5
args.adam_epsilon = 1e-8
args.weight_decay = 0
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
from transformers import AdamW
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
# 开启finetune模式 ,,,,,,,C:\Users\Administrator\.PyCharm2019.3\system\remote_sources\-456540730\-337502517\transformers\data\processors\squad.py 从这个里面进行抄代码即可.
model.zero_grad()
model.train()
print('start_train')
      # 看梯度的锁[i[1].requires_grad for i in list(model.named_parameters())]
      # 看变量名[i[0] for i in list(model.named_parameters())]



#=========是否进行前多少层的冻结.
if 0:
    a=[i[1] for i in list(model.named_parameters())[:318]] # ======直接最暴力的方法,全锁上,除了最后一层.的classify和bbox
    for i in a:
        i.requires_grad=False


import cv2



for _ in range(epoch):
    print('当前的epoch是',_)
    for dex4,i in enumerate(train_loader):
            tmp = i
            print(1)
            images=[Image.open(url) for url in tmp[0]]
            # images=[torch.tensor(url).cuda() for url in images]
            annos=[j for j in tmp[1]]
            annos2=[]
            for index,k in enumerate(annos):
                k= k.split(' ')
                annos3=[] # 这个是每个图片的标注数组.
                for kk in k:

                    box=[int(jjj) for jjj in kk.split(',')[:-1]]
                    category=int(kk.split(',')[-1])
                    x1x2y1y2 = [box[0], box[2], box[1], box[3]]
                    coco_data = [x1x2y1y2[0], x1x2y1y2[2], x1x2y1y2[1] - x1x2y1y2[0], x1x2y1y2[3] - x1x2y1y2[2]]
                    area2 = (x1x2y1y2[1] - x1x2y1y2[0]) * (x1x2y1y2[3] - x1x2y1y2[2])
                    # coco_data=torch.tensor(coco_data).cuda()
                    # category=torch.tensor(category).cuda()
                    # area2=torch.tensor(area2).cuda()


                    annos3.append( {"bbox": coco_data, "area": area2, "iscrowd": 0, "category_id": category})
                    # a={"annotations": [
                    #    ,
                    # ], "image_id": index}
                annos2.append({'annotations':annos3,'image_id':index})
            # print("转化为的coco标签:",annos2)


            print('=================')

            #
            # inputs = feature_extractor(images=[image, image], annotations=[
            #     {"annotations": [
            #         {"bbox": coco_data, "area": area2, "iscrowd": 0, "category_id": 1},
            #     ], "image_id": 0}
            #     ,
            #     {"annotations": [
            #         {"bbox": coco_data, "area": area2, "iscrowd": 0, "category_id": 1},
            #     ], "image_id": 1}
            #
            # ],
            #                            return_tensors="pt")

            inputs = feature_extractor(images=images, annotations=annos2,
                                       return_tensors="pt")








            # annotation里面是多个物体.  , 输入box 是左上点和宽高   跟coco的格式是一样的. 转化完后inputs里面的box是  (center_x, center_y, width, height)
            inputs.data['pixel_values']=inputs.data['pixel_values'].cuda()
            inputs.data['pixel_mask']=inputs.data['pixel_mask'].cuda()
            for dex3 in range(len(inputs.data['labels'])):
                for nn in inputs.data['labels'][dex3]:
                    inputs.data['labels'][dex3][nn]= inputs.data['labels'][dex3][nn].cuda() #这个地方要赋值才行. .cuda不是传地址.
            # for nn in inputs.data['labels'][1]:
            #     inputs.data['labels'][1][nn]=inputs.data['labels'][1][nn].cuda()
            # image_id对应images里面图片的索引
            # inputs.cuda()
            outputs = model(**inputs)
            loss = outputs[0]
            print(loss)
            loss.backward()
            optimizer.step()

            model.zero_grad()


            #===========每一轮训练之前我们测试一下.
            # if dex4==0:
            #     id2label= {
            #         "0": "N/A",
            #         "1": "code",
            #         "2": "bicycle",
            #         "3": "car",
            #         "4": "motorcycle",}
            #     image=images[0]
            #     logits = outputs.logits[0]
            #     bboxes = outputs.pred_boxes[0]
            #     numbeijing=VOC_DATA["NUM"]
            #     yuzhi=0.7
            #     usedex = logits.max(2)[1] != numbeijing  # ========91是背景分类, 表示空物体.
            #     usedex2 = logits.max(2)[0] > yuzhi
            #     logits_hat = logits[(logits.max(2)[1] != numbeijing) & (logits.max(2)[0] > yuzhi)]
            #     box_hat = bboxes[(logits.max(2)[1] != numbeijing) & (logits.max(2)[0] > yuzhi)]
            #     classify_hat = logits_hat.argmax(-1)  # box的分类结果
            #     classify_hat = [id2label[str(int(i))] for i in classify_hat]
            #     gailv = logits_hat.softmax(-1).max(-1)[0].tolist()
            #     print("识别到的物体是", classify_hat)
            #     print("概率是", gailv)
            #     import numpy as np
            #     import matplotlib
            #
            #     matplotlib.use('agg')
            #
            #     import matplotlib.pyplot as plt
            #
            #     # ========根据box, 画图片可视化结果.  网络输出出来的结果是, box 的中心点x,y 和 w ,h 的百分比.
            #     # lx, ly , rx ,ry    image.size   box_hat
            #     outbox = []
            #     for i in box_hat:
            #         lx = (i[0] - i[2] / 2).clamp(0, 1) * image.size[
            #             0]  # ==========一定要根据我的写法理解宽高和矩阵里面反过来.这里面要看好0,1 哪个是宽,哪个是高.
            #         rx = (i[0] + i[2] / 2).clamp(0, 1) * image.size[0]
            #         ly = (i[1] - i[3] / 2).clamp(0, 1) * image.size[1]
            #         ry = (i[1] + i[3] / 2).clamp(0, 1) * image.size[1]
            #         lx = lx.item()
            #         rx = rx.item()
            #         ly = ly.item()
            #         ry = ry.item()
            #         outbox.append((lx, rx, ly, ry))
            #     # print(outbox)
            #     from PIL import ImageDraw
            #
            #     image = image  # 打开一张图片
            #     draw = ImageDraw.Draw(image)  # 在上面画画
            #
            #     for dex, i in enumerate(outbox):  # 注意画图需要的坐标顺序!!!!!!!!!!!!
            #         draw.rectangle([i[0], i[2], i[1], i[3]], outline=(255, 0, 0))  # [左上角x，左上角y，右下角x，右下角y]，outline边框颜色
            #         draw.text((i[1], i[2]), classify_hat[dex], fill=(255, 0, 0))
            #     image.save("tmp.png")
            # image.show()

#===========然后玩loss就行了.
# model predicts bounding boxes and corresponding COCO classes
torch.save(model,'1.pth')
print("模型存在了1.pth")

outputs=outputs


print("train_over")
#===========画图部分
#===========然后玩loss就行了.
# model predicts bounding boxes and corresponding COCO classes

#
# id2label= {
#     "0": "N/A",
#     "1": "code",
#     "2": "bicycle",
#     "3": "car",
#     "4": "motorcycle",
#     "5": "airplane",
#     "6": "bus",
#     "7": "train",
#     "8": "truck",
#     "9": "boat",
#     "10": "traffic light",
#     "11": "fire hydrant",
#     "12": "N/A",
#     "13": "stop sign",
#     "14": "parking meter",
#     "15": "bench",
#     "16": "bird",
#     "17": "cat",
#     "18": "dog",
#     "19": "horse",
#     "20": "sheep",
#     "21": "cow",
#     "22": "elephant",
#     "23": "bear",
#     "24": "zebra",
#     "25": "giraffe",
#     "26": "N/A",
#     "27": "backpack",
#     "28": "umbrella",
#     "29": "N/A",
#     "30": "N/A",
#     "31": "handbag",
#     "32": "tie",
#     "33": "suitcase",
#     "34": "frisbee",
#     "35": "skis",
#     "36": "snowboard",
#     "37": "sports ball",
#     "38": "kite",
#     "39": "baseball bat",
#     "40": "baseball glove",
#     "41": "skateboard",
#     "42": "surfboard",
#     "43": "tennis racket",
#     "44": "bottle",
#     "45": "N/A",
#     "46": "wine glass",
#     "47": "cup",
#     "48": "fork",
#     "49": "knife",
#     "50": "spoon",
#     "51": "bowl",
#     "52": "banana",
#     "53": "apple",
#     "54": "sandwich",
#     "55": "orange",
#     "56": "broccoli",
#     "57": "carrot",
#     "58": "hot dog",
#     "59": "pizza",
#     "60": "donut",
#     "61": "cake",
#     "62": "chair",
#     "63": "couch",
#     "64": "potted plant",
#     "65": "bed",
#     "66": "N/A",
#     "67": "dining table",
#     "68": "N/A",
#     "69": "N/A",
#     "70": "toilet",
#     "71": "N/A",
#     "72": "tv",
#     "73": "laptop",
#     "74": "mouse",
#     "75": "remote",
#     "76": "keyboard",
#     "77": "cell phone",
#     "78": "microwave",
#     "79": "oven",
#     "80": "toaster",
#     "81": "sink",
#     "82": "refrigerator",
#     "83": "N/A",
#     "84": "book",
#     "85": "clock",
#     "86": "vase",
#     "87": "scissors",
#     "88": "teddy bear",
#     "89": "hair drier",
#     "90": "toothbrush"
#   }
# yuzhi=0.7
# logits = outputs.logits.softmax(-1)
# bboxes = outputs.pred_boxes
#
# usedex=logits.max(2)[1]!=91
# usedex2=logits.max(2)[0]>yuzhi
# logits_hat=logits[(logits.max(2)[1]!=91) & (logits.max(2)[0]>yuzhi)]
# box_hat=bboxes[(logits.max(2)[1]!=91) & (logits.max(2)[0]>yuzhi)]
# classify_hat=logits_hat.argmax(-1) # box的分类结果
# classify_hat=[id2label[str(int(i))] for i in classify_hat] # 翻译 对应分类物体名字.
# gailv =logits_hat.max(-1)[0].tolist()
# print("识别到的物体是",classify_hat)
# print("概率是",gailv)
# import numpy as np
# import matplotlib
# matplotlib.use('agg')
#
# import matplotlib.pyplot as plt
#
#
#
# # lx, ly , rx ,ry    image.size   box_hat
# outbox=[]
# for i in box_hat:
#     lx=(i[0]-i[2]/2).clamp(0,1)*image.size[0]
#     rx=(i[0]+i[2]/2).clamp(0,1)*image.size[0]
#     ly=(i[1]-i[3]/2).clamp(0,1)*image.size[1]
#     ry=(i[1]+i[3]/2).clamp(0,1)*image.size[1]
#     lx=lx.item()
#     rx=rx.item()
#     ly=ly.item()
#     ry=ry.item()
#     outbox.append((lx,rx,ly,ry))
# # print(outbox)
# from PIL import ImageDraw
# image = image # 打开一张图片
# draw = ImageDraw.Draw(image) # 在上面画画
#
# for dex,i in enumerate(outbox):  # 注意画图需要的坐标顺序!!!!!!!!!!!!
#     draw.rectangle([i[0],i[2],i[1],i[3]], outline=(255,0,0)) # [左上角x，左上角y，右下角x，右下角y]，outline边框颜色
#     draw.text((i[1], i[2]), classify_hat[dex], fill=(255, 0, 0))
# image.save("tmp.png")
# # image.show()
#
#
#
# #   logits.max(2)  看分类.
#
# # 分类表:
