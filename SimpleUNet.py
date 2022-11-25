import torch
from torch import nn
import math
import einops

#========================================================
# Position embeddings 
# Since the network need to know which timestep it is in, the transformer Sinusoidal Embedding is used.
#========================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device # getting info about whether the variable 'time' is using CPU or GPU
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#========================================================
# Simple blocks (Used by Simplified UNet) 
# this blocks only contain Convs while original paper implementation includes ResNet blocks and Attn blocks.
#========================================================
class SimpleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_channel)
        if up:
            self.conv1 = nn.Conv2d(2*in_channel, out_channel, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
            self.transform = nn.Conv2d(out_channel, out_channel, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bnorm = nn.BatchNorm2d(out_channel)
        self.relu  = nn.ReLU()
        
    def forward(self, x, timestep):
        # First Conv
        h = self.bnorm(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(timestep))
        # Extend last 2 dimensions
        time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

#========================================================
# Simplified UNet 
# We use a simple form of a UNet for to predict the noise in the image
# The input is a noisy image, the output is the noise in the image <---- Important
# Note that the output shape of UNet should be the same as input shape
#========================================================
class SimpleUNet(nn.Module):
    def __init__(self, dim=64, channels=3, dim_mults=(1,2,4,8), time_emb_dim=32) -> None:
        super().__init__()

        # You can customize the time embedding mlp layers
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim), # (B,) => (B, 32)
            nn.Linear(time_emb_dim, time_emb_dim), # (B, 32) => (B, 32)
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim), # (B, 32) => (B, 32)
        ) 

        # Initial projection (modified to use a smaller kernel)
        self.init_proj = nn.Conv2d(channels, dim * dim_mults[0], kernel_size=3, stride=1, padding=1) # (B, 3, 64, 64) => (B, 64, 64, 64)

        down_features = [dim*mult for mult in dim_mults] # [64, 128, 256, 512]
        up_features =  down_features[::-1]  # reverse the downs_features list => [512, 256, 128, 64]

        # Downsampling
        self.downs = nn.ModuleList([])
        for index in range(len(down_features)-1):
            self.downs.append(SimpleBlock(down_features[index], down_features[index+1], time_emb_dim, up=False))

        # Upsampling
        self.ups = nn.ModuleList([])
        for index in range(len(up_features)-1):
            self.ups.append(SimpleBlock(up_features[index], up_features[index+1], time_emb_dim, up=True))

        self.last = nn.Conv2d(up_features[-1], channels, 1) 

    def forward(self, x, timestep):
        # Embedd time (B,) => (B, 32)
        t = self.time_mlp(timestep)
        # Initial conv (B, 3, 64, 64) => (B, 64, 64, 64)
        x = self.init_proj(x) 
        # Unet
        residual_inputs = [] # List for storing residuals
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop() # pop means pull out the last item from the list
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.last(x)

if __name__ == "__main__":

    def test_SimpleUNet():
        batch_size, img_channel, img_width, img_height = 1, 3, 64, 64
        timesteps = 250
        x = torch.randn((batch_size, img_channel, img_width, img_height)).to("cuda")
        t = torch.randint(0, timesteps, (batch_size,)).to("cuda") # Generate a timestep according to the batch size
        print("timestep: ", t)
        model = SimpleUNet().to("cuda")
        preds = model(x, timestep=t)
        print("shape of prediction: ", preds.shape)
        assert preds.shape == x.shape
        try:
            from utils import print_network
            print_network(model)
        except:
            pass
        print("=> test_SimpleUNet passed")

    def test_SinusoidalPositionEmbeddings():
        pos_emb = SinusoidalPositionEmbeddings(32)
        batch_size = 1
        timesteps = 250
        t = torch.randint(0, timesteps, (batch_size,)).to("cuda") # Generate a timestep according to the batch size
        emb = pos_emb(t)
        print("emb.shape: ", emb.shape)
        print(emb)
        print("=> test_SinusoidalPositionEmbeddings passed")
    
    test_SinusoidalPositionEmbeddings()
    test_SimpleUNet()
        

