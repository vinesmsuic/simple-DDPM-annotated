import torch
import torch.nn.functional as F
from scheduler import linear_beta_schedule, betas_to_alphas_cumprod

timesteps = 250
# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)
# calculate alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# Get the indexed term from the list
# https://github.com/pytorch/pytorch/issues/15245 
# Gather backward is faster than integer indexing on GPU
def extract(a, t, x_shape):
	batch_size = t.shape[0]
	out = a.gather(-1, t.cpu())
	return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property of alphas)
def q_sample(x_start, t, noise=None):
	if noise is None:
		noise = torch.randn_like(x_start)
	# \sqrt{\bar\alpha_t}
	sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
	# \sqrt{(1-\bar\alpha_t)}
	sqrt_one_minus_alphas_cumprod_t = extract(
		sqrt_one_minus_alphas_cumprod, t, x_start.shape
	)
	# \mathcal{N}\left(x_{t}; \sqrt{\bar\alpha_t} x_{0}, (1-\bar\alpha_t) I\right)
	# N(mean, var) * (1-alpha_cumprod) = N(mean, (1-alpha_cumprod) * var)
	return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# reverse diffusion 
# Sample from the model. Mean is predicted, Variance is fixed in this example
@torch.no_grad()
def p_sample(model, x, t, t_index):
	betas_t = extract(betas, t, x.shape)
	sqrt_one_minus_alphas_cumprod_t = extract(
		sqrt_one_minus_alphas_cumprod, t, x.shape
	)
	sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
	
	# Use our model (noise predictor) to predict the mean
	model_mean = sqrt_recip_alphas_t * (
		x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
	)

	if t_index == 0:
		return model_mean
	else:
		posterior_variance_t = extract(posterior_variance, t, x.shape)
		noise = torch.randn_like(x)
		# x_{t-1} sample is generated
		image = model_mean + torch.sqrt(posterior_variance_t) * noise 
		return image