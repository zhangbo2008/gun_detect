import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image=Image.open('guns/Images/1.jpeg')
model.eval()
if 1:
    texts = [["a photo of gun", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt") # 配置文件里面写了16. 每一个分类最长用16个token表示. 绝对够了.
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    #返回超多的结果, 一共返回patch_size多个, 576个.
    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold: #我们只要大于0.1的.
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

print('下面编写finetune代码')

model.train()
criterion = nn.CrossEntropyLoss()#=========这个损失跟论文都不太对.#先跑通代码然后再改细节.
criterion=nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(10):
        optimizer.zero_grad()

        
        gt=torch.tensor([76 ,45 ,146, 87]).float()
        texts = [["a photo of gun", "a photo of a dog"]]
        inputs = processor(text=texts, images=image, return_tensors="pt") # 配置文件里面写了16. 每一个分类最长用16个token表示. 绝对够了.
        outputs = model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
        loss=0
        for  item in outputs['pred_boxes']:
            loss += criterion(item, gt)
        print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, 1, loss.item()))
        loss.backward()
        optimizer.step()
        
model.eval()
if 1:
    texts = [["a photo of gun", "a photo of a dog"]]
    inputs = processor(text=texts, images=image, return_tensors="pt") # 配置文件里面写了16. 每一个分类最长用16个token表示. 绝对够了.
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    #返回超多的结果, 一共返回patch_size多个, 576个.
    score_threshold = 0.1
    print(111111111111111111111111)
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold: #我们只要大于0.1的.
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

#==================检测的loss需要参考detr里面匈牙利匹配算法来写..............................


#===========
sequence_output = outputs[0]
if 1:
        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DetrHungarianMatcher(
                class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = DetrLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs

            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)