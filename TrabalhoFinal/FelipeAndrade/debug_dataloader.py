import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets.mbgv2_crops_dataset import Mbgv2CropsDataset


def main():
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),  # [0,1], shape [C,H,W]
    ])

    dataset = Mbgv2CropsDataset(split="train", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    batch = next(iter(loader))
    imgs, labels = batch
    print("Batch imgs shape:", imgs.shape)   # esperado: [32, 3, 128, 128]
    print("Batch labels shape:", labels.shape)
    print("Primeiros labels:", labels[:10])


if __name__ == "__main__":
    main()
