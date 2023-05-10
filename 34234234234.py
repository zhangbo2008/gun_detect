from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes



import torch
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
#==========
# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
        )






#===========画图.
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt


#========根据box, 画图片可视化结果.  网络输出出来的结果是, box 的中心点x,y 和 w ,h 的百分比.
# lx, ly , rx ,ry    image.size   box_hat
outbox=[]
box_hat=results["boxes"]
for i in box_hat:
    # lx=(i[0]-i[2]/2).clamp(0,1) #==========一定要根据我的写法理解宽高和矩阵里面反过来.这里面要看好0,1 哪个是宽,哪个是高.
    # rx=(i[0]+i[2]/2).clamp(0,1)
    # ly=(i[1]-i[3]/2).clamp(0,1)
    # ry=(i[1]+i[3]/2).clamp(0,1)
    lx=i[0].item()
    ly=i[1].item()
    rx=i[2].item()
    ry=i[3].item()
    outbox.append((lx,rx,ly,ry))#========这个坐标很乱, 自己对应一下.我也调了半天.
# print(outbox)
from PIL import ImageDraw
image = image # 打开一张图片
draw = ImageDraw.Draw(image) # 在上面画画

for dex,i in enumerate(outbox):  # 注意画图需要的坐标顺序!!!!!!!!!!!!
    draw.rectangle([i[0],i[2],i[1],i[3]], outline=(255,0,0)) # [左上角x，左上角y，右下角x，右下角y]，outline边框颜色
    draw.text((i[1], i[2]),model.config.id2label[results["labels"][dex].item()]+"  "+str(results["scores"][dex].item()), fill=(255, 0, 0))







print('图片保存为tmp.png')
image.save("tmp.png")










if 0:
    print(outputs)


    #===========下面我们画出来这个图.


    #========完全就是detr的配置.



    logits = outputs.logits
    bboxes = outputs.pred_boxes
    print(logits,bboxes)
    usedex=logits.max(2)[1]!=91
    usedex2=logits.max(2)[0]>0.7
    usedex3=logits[(logits.max(2)[1]!=91) & (logits.max(2)[0]>0.7)]
    box=bboxes[(logits.max(2)[1]!=91) & (logits.max(2)[0]>0.7)]




    raise
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


    numbeijing=VOC_DATA["NUM"]
    id2label= {
        "0": "life",
        "1": "name",
        "2": "idn",
        "3": "front",
        "4": "back",}
    usedex=logits.max(2)[1]!=numbeijing  # ========91是背景分类, 表示空物体.
    usedex2=logits.max(2)[0]>yuzhi
    logits_hat=logits[(logits.max(2)[1]!=numbeijing) & (logits.max(2)[0]>yuzhi)]
    box_hat=bboxes[(logits.max(2)[1]!=numbeijing) & (logits.max(2)[0]>yuzhi)]
    classify_hat=logits_hat.argmax(-1) # box的分类结果
    classify_hat=[id2label[str(int(i))] for i in classify_hat]
    gailv =logits_hat.softmax(-1).max(-1)[0].tolist()
    print("识别到的物体是",classify_hat)
    print("概率是",gailv)
    import numpy as np
    import matplotlib
    matplotlib.use('agg')

    import matplotlib.pyplot as plt


    #========根据box, 画图片可视化结果.  网络输出出来的结果是, box 的中心点x,y 和 w ,h 的百分比.
    # lx, ly , rx ,ry    image.size   box_hat
    outbox=[]
    for i in box_hat:
        lx=(i[0]-i[2]/2).clamp(0,1)*image.size[0] #==========一定要根据我的写法理解宽高和矩阵里面反过来.这里面要看好0,1 哪个是宽,哪个是高.
        rx=(i[0]+i[2]/2).clamp(0,1)*image.size[0]
        ly=(i[1]-i[3]/2).clamp(0,1)*image.size[1]
        ry=(i[1]+i[3]/2).clamp(0,1)*image.size[1]
        lx=lx.item()
        rx=rx.item()
        ly=ly.item()
        ry=ry.item()
        outbox.append((lx,rx,ly,ry))
    # print(outbox)
    from PIL import ImageDraw
    image = image # 打开一张图片
    draw = ImageDraw.Draw(image) # 在上面画画

    for dex,i in enumerate(outbox):  # 注意画图需要的坐标顺序!!!!!!!!!!!!
        draw.rectangle([i[0],i[2],i[1],i[3]], outline=(255,0,0)) # [左上角x，左上角y，右下角x，右下角y]，outline边框颜色
        draw.text((i[1], i[2]), classify_hat[dex], fill=(255, 0, 0))
    image.save("tmp.png")
    # image.show()







    #=========根据box_hat画出来坐标



    #=========解析box   (center_x, center_y, width, height)








    #   logits.max(2)  看分类.

    # 分类表:
    """
    "id2label": {
        "0": "N/A",
        "1": "person",
        "2": "bicycle",
        "3": "car",
        "4": "motorcycle",
        "5": "airplane",
        "6": "bus",
        "7": "train",
        "8": "truck",
        "9": "boat",
        "10": "traffic light",
        "11": "fire hydrant",
        "12": "N/A",
        "13": "stop sign",
        "14": "parking meter",
        "15": "bench",
        "16": "bird",
        "17": "cat",
        "18": "dog",
        "19": "horse",
        "20": "sheep",
        "21": "cow",
        "22": "elephant",
        "23": "bear",
        "24": "zebra",
        "25": "giraffe",
        "26": "N/A",
        "27": "backpack",
        "28": "umbrella",
        "29": "N/A",
        "30": "N/A",
        "31": "handbag",
        "32": "tie",
        "33": "suitcase",
        "34": "frisbee",
        "35": "skis",
        "36": "snowboard",
        "37": "sports ball",
        "38": "kite",
        "39": "baseball bat",
        "40": "baseball glove",
        "41": "skateboard",
        "42": "surfboard",
        "43": "tennis racket",
        "44": "bottle",
        "45": "N/A",
        "46": "wine glass",
        "47": "cup",
        "48": "fork",
        "49": "knife",
        "50": "spoon",
        "51": "bowl",
        "52": "banana",
        "53": "apple",
        "54": "sandwich",
        "55": "orange",
        "56": "broccoli",
        "57": "carrot",
        "58": "hot dog",
        "59": "pizza",
        "60": "donut",
        "61": "cake",
        "62": "chair",
        "63": "couch",
        "64": "potted plant",
        "65": "bed",
        "66": "N/A",
        "67": "dining table",
        "68": "N/A",
        "69": "N/A",
        "70": "toilet",
        "71": "N/A",
        "72": "tv",
        "73": "laptop",
        "74": "mouse",
        "75": "remote",
        "76": "keyboard",
        "77": "cell phone",
        "78": "microwave",
        "79": "oven",
        "80": "toaster",
        "81": "sink",
        "82": "refrigerator",
        "83": "N/A",
        "84": "book",
        "85": "clock",
        "86": "vase",
        "87": "scissors",
        "88": "teddy bear",
        "89": "hair drier",
        "90": "toothbrush"
    },
    """




