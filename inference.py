import torch
from torch.optim import Adam
from torch import nn
import argparse
import os
from torchvision.utils import save_image, make_grid

from UNet import SimpleUNet
from ddpm import DDPM
from utils import seed_everything


def parser():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('-w',"--weight",required=True,
						help="filepath to your weight file.")
	parser.add_argument('-n', "--num_samples", type=int, default=10,
						help="num_samples.")
	parser.add_argument('-ims', "--imsize", type=int, default=64,
						help="image size.")
	parser.add_argument('-T', "--timesteps", type=int, default=1000,
						help="timesteps.")
	parser.add_argument('-sd', "--seed", type=int, default=42,
						help="seed number. Default is 42 for reproducible result")
	parser.add_argument('-d', "--dest", type=str, default= "log_infer",
						help="Destination folder path for saving results.")
	return parser.parse_args()

def load_checkpoint(model, path, device):
	print("=> Loading checkpoint")
	if (os.path.isfile(path)):
		checkpoint = torch.load(path, map_location=device)
		for key in checkpoint:
			print(key)
		model.load_state_dict(checkpoint)
		print("checkpoint file " + str(path) + " loaded.")
	else:
		raise Exception("checkpoint file " + str(path) + " not found.") 

def main(args):
	seed_everything(args.seed)
	timesteps = args.timesteps
	device = "cuda" if torch.cuda.is_available() else "cpu"
	ddpm = DDPM(eps_model=SimpleUNet(),
				timesteps=timesteps,
				criterion=nn.L1Loss()).to(device)
	load_checkpoint(ddpm, path=args.weight, device=device)
	dest_folder = args.dest
	if not (os.path.exists(dest_folder)):
		os.mkdir(dest_folder)
	ddpm.eval()
	with torch.no_grad():
		for i in range(args.num_samples):
			gen = ddpm.p_sample_loop((1, 3, args.imsize, args.imsize), device=device)
			save_image(gen, os.path.join(dest_folder, f"{i:03d}.png"))


if __name__ == "__main__":
	args = parser()
	main(args)