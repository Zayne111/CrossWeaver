import torch
from torch import nn
from torch.nn import functional as F
import yaml
from fvcore.nn import FlopCountAnalysis
from semseg.models.heads import SegFormerHead
from semseg.models.base import BaseModel    


class CrossWeaver(BaseModel):
    def __init__(self, backbone: str = 'CrossWeaver-B0', num_classes: int = 19, modals: list = ['img', 'depth', 'event', 'lidar']):
        super().__init__(backbone, num_classes, modals)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        #y = y.mean(dim=0, keepdim=True)
        return y

    def init_pretrained(self, pretrained: str = None) -> None:
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        msg = self.backbone.load_state_dict(checkpoint, strict=False)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)
        print(msg)





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modals = ['img', 'event', 'lidar']
    model = CrossWeaver(backbone='CrossWeaver-B0', num_classes=11, modals=modals).to(device)
    model_path = 'output/CrossWeaver_mcubes.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")),strict=False)
    

