from vit_pytorch import ViT

from horizon.base.model import BaseModel


class VIT(BaseModel):

    def __init__(self):
        super().__init__()
        self.v = ViT(image_size=32,
                     patch_size=4,
                     num_classes=100,
                     dim=256,
                     depth=6,
                     heads=16,
                     mlp_dim=512,
                     dropout=0.0,
                     emb_dropout=0.0)

    def forward(self, image):
        return self.v(image)
