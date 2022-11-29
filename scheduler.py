import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_ddpm_schedules(timesteps, start=0.0001, end=0.02):
    betas = linear_beta_schedule(timesteps, start, end)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # Return a dictionary storing the predefined values. 
    return {
        "betas": betas, #\beta
        "alphas": alphas,  # \alpha
        "alphas_cumprod": alphas_cumprod,  # \bar{\alpha} 
        "alphas_cumprod_prev": alphas_cumprod_prev,  
        "sqrt_recip_alphas": sqrt_recip_alphas, # 1/\sqrt{\alpha}
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,  # \sqrt{\bar{\alpha}}
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,  # \sqrt{1-\bar{\alpha}}
        "posterior_variance": posterior_variance,  # 
    }

#==========================================================================
if __name__ == "__main__":

    # Demo usage of the code
    import matplotlib.pyplot as plt
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    def plot_betas():
        fig = plt.gcf()
        fig.set_size_inches(9, 5)
        plt.subplot(1, 2, 1) # row 1, col 2 index 1
        plt.plot(betas)
        plt.title(r"$\beta$")
        plt.xlabel('Timesteps $t$')
        plt.ylabel('Value')

        plt.subplot(1, 2, 2) # index 2
        plt.plot(torch.sqrt(1 - betas))
        plt.title(r"$\sqrt{1 - \beta}$")
        plt.xlabel('Timesteps $t$')

        plt.show()

    def plot_alphas():
        fig = plt.gcf()
        fig.set_size_inches(9, 5)
        plt.subplot(1, 2, 1) # row 1, col 2 index 1
        plt.plot(torch.sqrt(alphas_cumprod), label=r'$\sqrt{\bar\alpha}$')
        plt.title(r'$\sqrt{\bar\alpha}$')
        plt.xlabel('Timesteps $t$')
        plt.ylabel('Value')

        plt.subplot(1, 2, 2) # index 2
        plt.plot(torch.sqrt(1 - alphas_cumprod), label=r'$\sqrt{1-\bar\alpha}$')
        plt.title(r'$\sqrt{1-\bar\alpha}$')
        plt.xlabel('Timesteps $t$')

        plt.show()

    # Show beta schedule
    plot_betas()

    # Show alphas schedule
    plot_alphas()
