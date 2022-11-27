import torch

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    timesteps = 250
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
