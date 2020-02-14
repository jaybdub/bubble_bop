import ctypes
import torch
import numpy as np
from torch2trt import TRTModule, torch2trt
import trt_pose.models
from trt_pose.parse_objects import ParseObjects    
from trt_pose.draw_objects import DrawObjects
import PIL.Image
import json
import tensorrt as trt
import trt_pose.coco
import os
import subprocess
import cv2
import torchvision.transforms as transforms


MODEL_URLS = {
    'resnet18_baseline_att': 'https://nvidia.box.com/shared/static/5c5d7bq7bvm4ic8vx7xasvf5ol70o2r9.pth',
    'densenet121_baseline_att': 'https://nvidia.box.com/shared/static/mn7f3a8le9bn8cwihl0v6s9wlm5damaq.pth'
}


COCO_CATEGORY = {
        "supercategory": "person", 
        "id": 1, 
        "name": "person", 
        "keypoints": [
            "nose", 
            "left_eye", 
            "right_eye", 
            "left_ear", 
            "right_ear", 
            "left_shoulder", 
            "right_shoulder", 
            "left_elbow", 
            "right_elbow", 
            "left_wrist", 
            "right_wrist", 
            "left_hip", 
            "right_hip", 
            "left_knee", 
            "right_knee", 
            "left_ankle", 
            "right_ankle", 
            "neck"
        ], 
        "skeleton": [
            [16, 14], 
            [14, 12], 
            [17, 15], 
            [15, 13], 
            [12, 13], 
            [6, 8], 
            [7, 9], 
            [8, 10], 
            [9, 11], 
            [2, 3], 
            [1, 2], 
            [1, 3], 
            [2, 4], 
            [3, 5], 
            [4, 6], 
            [5, 7], 
            [18, 1], 
            [18, 6], 
            [18, 7], 
            [18, 12], 
            [18, 13]
        ]
}


TOPOLOGY = trt_pose.coco.coco_category_to_topology(COCO_CATEGORY)


class PoseDetector(object):
    
    def __init__(self,
                 torch_model='resnet18_baseline_att',
                 input_shape=(224, 224),
                 dtype=torch.float32,
                 device=torch.device('cuda'),
                 torch2trt_kwargs={'max_workspace_size': 1<<25, 'fp16_mode': True}):
        
        self.dtype = dtype
        self.device = device
        self.input_shape = input_shape
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).type(dtype)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device).type(dtype)
        
        model_path = torch_model + '_trt.pth'
        if not os.path.exists(model_path):
            
            # download model
            download_path = torch_model + '_torch.pth'
            subprocess.call(['wget', MODEL_URLS[torch_model], '-O', download_path])
            
            # load downloaded model
            model = trt_pose.models.MODELS[torch_model](len(COCO_CATEGORY['keypoints']), len(COCO_CATEGORY['skeleton'])*2).eval().to(device)
            model.load_state_dict(torch.load(download_path))
            
            # optimize with TensorRT
            data = torch.randn((1, 3) + input_shape).to(device).type(dtype)
            self.model = torch2trt(model, [data], **torch2trt_kwargs)
            torch.save(self.model.state_dict(), model_path)
            
        else:
            
            # load model
            self.model = TRTModule()
            self.model.load_state_dict(torch.load(model_path))
            
        self._topology = trt_pose.coco.coco_category_to_topology(COCO_CATEGORY)
        self.parse_objects = ParseObjects(TOPOLOGY)
        
    def _preprocess(self, image):
        image = cv2.resize(image, self.input_shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device).type(self.dtype)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]
    
    def __call__(self, image):
        with torch.no_grad():
            data = self._preprocess(image)
            cmap, paf = self.model(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)
            
        people = []
        
        for person_idx in range(int(counts[0])):
            
            person = {}
            
            for part_idx in range(len(COCO_CATEGORY['keypoints'])):
                
                idx = objects[0, person_idx, part_idx]
                
                if idx >= 0:
                    yx = peaks[0, part_idx, idx]
                    person[COCO_CATEGORY['keypoints'][part_idx]] = (float(yx[1]), float(yx[0]))
                    
            people.append(person)
            
        return people