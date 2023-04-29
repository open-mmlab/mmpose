import os
import cv2
import random
import numpy as np
from PIL import Image, ImageMath
from mmpose.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

def change_background(img, mask, bg):
        ow, oh = img.size
        bg = bg.resize((ow, oh)).convert('RGB')
        
        imcs = list(img.split())
        bgcs = list(bg.split())
        maskcs = list(mask.split())
        fics = list(Image.new(img.mode, img.size).split())
        
        for c in range(len(imcs)):
            negmask = maskcs[c].point(lambda i: 1 - i / 255)
            posmask = maskcs[c].point(lambda i: i / 255)
            fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
        out = Image.merge(img.mode, tuple(fics))
        
        return out


@TRANSFORMS.register_module()
class CopyPaste6D(BaseTransform):
    """change the background"""
    def __init__(self,
                 background_path = None,
    ):
        background_path_list = []
        for filename in os.listdir(background_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                background_path_list.append(os.path.join(background_path, filename))
        self.background_path_list = background_path_list
        
    def transform(self, results:dict) -> dict:
        ## data augmentation
        img = results['img']
        maskpath = results.get('mask_path')
        
        if maskpath is not None:
            bgpath = random.choice(self.background_path_list)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mask = Image.open(maskpath)
            bg = Image.open(bgpath)
            img = change_background(img, mask, bg)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            results['img'] = img
        
        return results
