
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloaders.dataset import TrainDataset, TestDataset
from models.swin import SwinTransformer6Channels
from trainer.train import train
from inference.test import infer_and_save
from utils.helpers import set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate Swin Transformer on Landsat data")
    parser.add_argument('--mode', choices=['train', 'infer'], required=True, help='Choose whether to train or infer')

    parser.add_argument('--train_data', type=str, help='Path to training cube directory')
    parser.add_argument('--train_meta', type=str, help='Path to training metadata CSV')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test cube directory')
    parser.add_argument('--test_meta', type=str, required=True, help='Path to test metadata CSV')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=11255)
    parser.add_argument('--save_model_path', type=str, default='swin-with-landsat-cubes.pth')
    parser.add_argument('--submission_path', type=str, default='submission_swintransformer.csv')
    args = parser.parse_args()

    set_seed(69)
    device = get_device()

    transform = transforms.Compose([transforms.ToTensor()])

    model = SwinTransformer6Channels(args.num_classes).to(device)

    if args.mode == 'train':
        assert args.train_data and args.train_meta, "Training data and metadata must be provided in train mode"

        train_meta = pd.read_csv(args.train_meta)
        train_ds = TrainDataset(args.train_data, train_meta, subset="train", transform=transform)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=25, verbose=True)

        print(f"Training for {args.epochs} epochs on {device}")
        train(model, train_loader, optimizer, scheduler, device, args.epochs, pos_weight_factor=1.0)

        torch.save(model.state_dict(), args.save_model_path)

    elif args.mode == 'infer':
        print(f"Loading model from {args.save_model_path}")
        model.load_state_dict(torch.load(args.save_model_path, map_location=device))
        model.eval()

    test_meta = pd.read_csv(args.test_meta)
    test_ds = TestDataset(args.test_data, test_meta, subset="test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    infer_and_save(model, test_loader, device, output_path=args.submission_path)

if __name__ == "__main__":
    main()