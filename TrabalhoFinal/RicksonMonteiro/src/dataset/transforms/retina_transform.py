import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image
import torch

class RetinaTransform:
    def __init__(self, size=640):
        self.size = size
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, image: Image.Image, target):
        # salvar tamanho original (h, w)
        orig_w, orig_h = image.size  # PIL: (width, height)
        target["orig_size"] = torch.tensor([orig_h, orig_w])

        # resize para (size, size) -- nota: distorce, mas é consistente
        image = F.resize(image, [self.size, self.size])

        # to tensor + normalize
        image = F.to_tensor(image)
        image = self.normalize(image)

        return image, target
