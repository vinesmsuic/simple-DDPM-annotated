import torch
from torch import nn
from scheduler import get_ddpm_schedules
from tqdm import tqdm

class DDPM(nn.Module):
	def __init__(
		self,
		eps_model: nn.Module,
		timesteps,
		criterion: nn.Module = nn.MSELoss(),
				) -> None:
		super().__init__()
		self.eps_model = eps_model # The noise predictor model

		# register_buffer allows us to freely access these tensors by name. It helps device placement.
		# from https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/ddpm.py
		for k, v in get_ddpm_schedules(timesteps).items():
			self.register_buffer(k, v)
		self.criterion = criterion
		self.timesteps = timesteps

	# Get the indexed term from the list.
	# https://github.com/pytorch/pytorch/issues/15245 
	# Gather backward is faster than integer indexing on GPU
	def extract(self, a, t, x_shape):
		batch_size = t.shape[0]
		out = a.gather(-1, t)
		return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

	# forward diffusion (using the nice property of alphas)
	# Adding noise to the inital image according to the schedule
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		# \sqrt{\bar\alpha_t}
		sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
		# \sqrt{(1-\bar\alpha_t)}
		sqrt_one_minus_alphas_cumprod_t = self.extract(
			self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
		)
		# \mathcal{N}\left(x_{t}; \sqrt{\bar\alpha_t} x_{0}, (1-\bar\alpha_t) I\right)
		# N(mean, var) * (1-alpha_cumprod) = N(mean, (1-alpha_cumprod) * var)
		return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

	# reverse diffusion 
	# Sample from the model. Mean is predicted, Variance is fixed in this example
	@torch.no_grad() # need not to keep track of the gradients
	def p_sample(self, x, t, t_index):
		betas_t = self.extract(self.betas, t, x.shape)
		sqrt_one_minus_alphas_cumprod_t = self.extract(
			self.sqrt_one_minus_alphas_cumprod, t, x.shape
		)
		sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
		
		# Use our model (noise predictor) to predict the mean
		model_mean = sqrt_recip_alphas_t * (
			x - betas_t * self.eps_model(x, t) / sqrt_one_minus_alphas_cumprod_t
		)

		if t_index == 0:
			return model_mean
		else:
			posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
			noise = torch.randn_like(x)
			# x_{t-1} sample is generated
			image = model_mean + torch.sqrt(posterior_variance_t) * noise 
			return image

	# reverse diffusion but chain all the steps until the noise become a meaningful image
	@torch.no_grad() # need not to keep track of the gradients
	def p_sample_loop(self, shape, device):
		b = shape[0]
		# start from pure noise (for each example in the batch)
		img = torch.randn(shape, device=device)

		for i in tqdm(reversed(range(0, self.timesteps)), desc='reverse diffusion loop time step', total=self.timesteps):
			img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
		return img

	def get_loss(self, x_start, t):
		# Forward to get noisy image x_t
		x_noisy, noise = self.q_sample(x_start, t)
		# Predict the noise added at timestep t
		predicted_noise = self.eps_model(x_noisy, t)
		# Cal loss function of noise and predicted noise
		loss = self.criterion(noise, predicted_noise)
		return loss


#==========================================================================
if __name__ == "__main__":

	# Demo usage of the code
	from UNet import SimpleUNet
	unet = SimpleUNet()
	crit = nn.MSELoss()
	timesteps=1000
	ddpm = DDPM(eps_model=unet, timesteps=timesteps, criterion=crit)

	from PIL import Image
	import requests
	import matplotlib.pyplot as plt
	from torchvision import transforms
	import numpy as np
	
	from utils import seed_everything
	seed_everything(42)

	def img2tensor(pil_img, imsize):
		transform = transforms.Compose([
			transforms.Resize(imsize),
			transforms.CenterCrop(imsize),
			transforms.ToTensor(), # turn into Numpy array of shape HWC, divide by 255
			transforms.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1]
		])
		im_tensor = transform(pil_img) # torch.Size([3, 128, 128])
		im_tensor_4d = im_tensor.unsqueeze(0) # torch.Size([1, 3, 128, 128])
		return im_tensor_4d

	def tensor2img(im_tensor_4d):
		reverse_transform = transforms.Compose([
			transforms.Lambda(lambda t: (t + 1) / 2),
			transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
			transforms.Lambda(lambda t: t * 255.),
			transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
			transforms.ToPILImage(),
		])
		pil_img = reverse_transform(im_tensor_4d.squeeze())
		return pil_img

	def get_noisy_image_pil(ddpm, img, t, imsize=128):
		# turn img into 4d tensor
		x_start = img2tensor(img, imsize)
		# add noise through forward diffusion
		x_noisy, noise = ddpm.q_sample(x_start, t=t)
		# turn back into PIL image
		noisy_image_pil = tensor2img(x_noisy)
		return noisy_image_pil
	
	# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
	def plot(imgs, row_title=None, **imshow_kwargs):
		if not isinstance(imgs[0], list):
			# Make a 2d grid even if there's just 1 row
			imgs = [imgs]

		num_rows = len(imgs)
		num_cols = len(imgs[0])
		fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
		for row_idx, row in enumerate(imgs):
			for col_idx, img in enumerate(row):
				ax = axs[row_idx, col_idx]
				ax.imshow(np.asarray(img), **imshow_kwargs)
				ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
		if row_title is not None:
			for row_idx in range(num_rows):
				axs[row_idx, 0].set(ylabel=row_title[row_idx])

		plt.tight_layout()
		plt.axis('off')
		plt.show()

	def get_img_from_url(url):
		return Image.open(requests.get(url, stream=True).raw)

	def get_img_from_path(path):
		return Image.open(path)

	# Get image from url (alternatively, from path in your local dir)
	img = get_img_from_url("https://i.imgur.com/M0SZxzd.jpg")
	#img = get_img_from_path("xxx.png")

	# Try different values yourself
	predefined_times = [0, 50, 100, 150, 200, 250, 500] 

	noisy_images = [get_noisy_image_pil(ddpm, img, torch.tensor([t])) for t in predefined_times]
	plot(noisy_images)
