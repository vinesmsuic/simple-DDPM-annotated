import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import os

from UNet import SimpleUNet
from ddpm import DDPM
from my_dataset import MyDataset

from utils import seed_everything


def parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-s', "--source", required=True,
                        help="filepath to your dataset image folder.")
    parser.add_argument('-b', "--batch_size", type=int, default=256,
                        help="batch size.")
    parser.add_argument('-ims', "--imsize", type=int, default=64,
                        help="image size.")
    parser.add_argument('-T', "--timesteps", type=int, default=1000,
                        help="timesteps.")
    parser.add_argument('-ep', "--epoch", type=int, default=500,
                        help="epochs. 500 is enough to make a clear images")
    parser.add_argument('-sd', "--seed", type=int, default=42,
                        help="seed number. Default is 42 for reproducible result")
    parser.add_argument('-d', "--dest", type=str, default="log",
                        help="Destination folder path for saving results.")
    parser.add_argument('-l', "--loss", type=str, default="MSE",
                        help="Use of loss function, either 'l1' or 'MSE' ")
    return parser.parse_args()


def get_transform(imsize):
    data_transforms = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # turn from Numpy array HWC => tensor CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ])
    return data_transforms


def main(args):
    seed_everything(args.seed)
    transform = get_transform(args.imsize)
    timesteps = args.timesteps
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    ddpm = DDPM(eps_model=SimpleUNet(),
                timesteps=timesteps,
                criterion=criterion)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm.to(device)
    optimizer = Adam(ddpm.parameters(), lr=0.001)

    my_dataset = MyDataset(root=args.source, transform=transform)
    dataloader = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dest_folder = args.dest
    if not (os.path.exists(dest_folder)):
        os.mkdir(dest_folder)
    epochs = args.epoch

    step = 0
    for epoch in range(epochs):
        for idx, batch_imgs in enumerate(dataloader):
            ddpm.train()
            optimizer.zero_grad()
            batch_imgs = batch_imgs.to(device)

            # Create a random time 
            t = torch.randint(low=0, high=timesteps, size=(batch_imgs.shape[0],), device=device).long()
            # Cal Loss
            loss = ddpm.get_loss(batch_imgs, t)
            loss.backward()

            step += 1
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

            optimizer.step()

        # Save Every 5 epochs and last epoch
        if epoch % 5 == 0 or epoch % epochs == 1:
            # Visualize training result
            ddpm.eval()
            with torch.no_grad():
                gen = ddpm.p_sample_loop((16, 3, args.imsize, args.imsize), device=device)
                grid = make_grid(gen, normalize=True, value_range=(-1, 1), nrow=4)
                save_image(grid, os.path.join(dest_folder, f"epoch_{epoch:03d}.png"))

                # save model
                torch.save(ddpm.state_dict(), os.path.join(dest_folder, f"ddpm_weight.pth"))


if __name__ == "__main__":
    args = parser()
    main(args)
