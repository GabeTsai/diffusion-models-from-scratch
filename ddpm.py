import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # DDPM from scratch (MNIST)

    `Unet` imp (canonical Ho et al. 2020, adapted from PixelCNN++) taken from [LucidRains' DDPM repo](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/d93647ad3cf97d786a064ef80429ee6c3e5ebd55/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L276).
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    from denoising_diffusion_pytorch import Unet
    from tqdm.auto import tqdm

    return (
        DataLoader,
        F,
        Unet,
        datasets,
        make_grid,
        nn,
        to_pil_image,
        torch,
        tqdm,
        transforms,
    )


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000            # diffusion timesteps
    image_size = 28
    channels = 1
    batch_size = 128
    lr = 2e-4
    epochs = 10
    return T, batch_size, channels, device, epochs, image_size, lr


@app.cell
def _(DataLoader, batch_size, datasets, transforms):
    # Reverse process assumes [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    return (train_loader,)


@app.cell
def _(Unet, channels, device):
    # Canonical Ho et al. 2020 UNet, tiny version for MNIST
    model = Unet(dim=64, dim_mults=(1, 2, 4), channels=channels, flash_attn=False).to(device)
    return (model,)


@app.cell
def _(F, T, device, model, nn, torch):
    class Diffusion(nn.Module):
        """DDPM noise schedule plus the p_sample (q) and reverse (p) process.

        Images in [-1, 1]; the model predicts the noise epsilon.
        Every schedule tensor has shape (T,) and is indexed by an integer timestep.
        """

        def __init__(self, model, timesteps=1000, device="cpu"):
            super().__init__()
            self.network = model
            self.T = timesteps
            self.device = device
            self.network.to(self.device)
            betas = torch.linspace(1e-4, 0.02, steps = self.T)
            alphas = 1 - betas
            alpha_bars = torch.cumprod(alphas, dim = 0)
            self.register_buffer('betas', betas)
            self.register_buffer('alphas', alphas)
            self.register_buffer('alpha_bars', alpha_bars)
            self.to(device)


        @staticmethod
        def _gather(coeffs, t, x_shape):
            """Index `coeffs` (T,) at timesteps `t` (B,), reshaped to broadcast over x.

            Returns shape (B, 1, 1, 1) so it multiplies a (B, C, H, W) tensor.
            """
            out = coeffs.gather(0, t) # (B, )
            return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

        def q_sample(self, x_0, t, noise):
            """Forward process: sample x_t ~ q(x_t | x_0) in closed form (DDPM eq. 4).

            Args:
                x_0:   (B, C, H, W) clean images in [-1, 1]
                t:     (B,) int64 timesteps
                noise: (B, C, H, W) ~ N(0, I)
            Returns:
                x_t:   (B, C, H, W)
            """
            alpha_bars_i = self._gather(self.alpha_bars, t, x_0.shape)
            return torch.sqrt(alpha_bars_i) * x_0 + torch.sqrt(1 - alpha_bars_i) * noise


        def p_losses(self, x_0, t):
            """L_simple: MSE between the true noise and the predicted noise (DDPM eq. 14).
            """
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise) # true noise
            eps_theta = self.network.forward(x_t, t) # predict noise
            return F.mse_loss(noise, eps_theta)


        @torch.no_grad()
        def p_sample(self, x_t, t):
            """One reverse step: sample x_{t-1} ~ p_theta(x_{t-1} | x_t) (DDPM eq. 11, Alg. 2).

            Args:
                x_t: (B, C, H, W)
                t:   (B,) int64 timesteps, all equal to the current step
            Returns:
                x_{t-1}: (B, C, H, W)
            """
            z = torch.randn_like((x_t))
            z[t == 0] = 0.0
            eps_theta = self.network(x_t, t)
            alphas_i = self._gather(self.alphas, t, x_t.shape) # (B, 1, 1, 1)
            alpha_bars_i = self._gather(self.alpha_bars, t, x_t.shape) # (B, 1, 1, 1)
            betas_i = self._gather(self.betas, t, x_t.shape) # (B, 1, 1, 1)
            return 1 / torch.sqrt(alphas_i) * (x_t - (1 - alphas_i) / torch.sqrt(1 - alpha_bars_i) * eps_theta) + torch.sqrt(betas_i) * z

        @torch.no_grad()
        def sample(self, n, image_size=28, channels=1):
            """Full reverse sampling process (Algorithm 2).

            Start from x_T ~ N(0, I) of shape (n, channels, image_size, image_size),
            then loop i = T-1 .. 0 calling forward / sampling from p_theta.

            Returns the final x_0 in [-1, 1].
            """
            x_t = torch.randn((n, channels, image_size, image_size), device=self.device)
            for t in range(self.T - 1, -1, -1):
                time_batch = torch.full((n, ), t, device=self.device, dtype=torch.long)
                x_t = self.p_sample(x_t, time_batch)
            return x_t

    diffusion = Diffusion(model, T, device)
    return (diffusion,)


@app.cell
def _(torch, tqdm):
    def train(model, diffusion, train_loader, epochs, lr):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"epoch {epoch}")
            for x_0, _ in pbar:
                x_0 = x_0.to(diffusion.device)
                t = torch.randint(0, diffusion.T, (x_0.shape[0],), device=diffusion.device)
                loss = diffusion.p_losses(x_0, t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (train,)


@app.cell
def _(make_grid, to_pil_image):
    def show_samples(diffusion, n=16, image_size=28, channels=1):
        diffusion.network.eval()
        imgs = diffusion.sample(n, image_size=image_size, channels=channels)  # [-1, 1]
        diffusion.network.eval()
        imgs = (imgs.clamp(-1, 1) + 1) / 2               # [0, 1] for display
        grid = make_grid(imgs.cpu(), nrow=int(n ** 0.5))
        return to_pil_image(grid)

    return (show_samples,)


@app.cell
def _():
    TRAIN = True
    return (TRAIN,)


@app.cell
def _(TRAIN, diffusion, epochs, lr, model, train, train_loader):
    if TRAIN:
        train(model, diffusion, train_loader, epochs=epochs, lr=lr)
    return


@app.cell
def _(TRAIN, channels, diffusion, image_size, show_samples):
    show_samples(diffusion, n=16, image_size=image_size, channels=channels) if not TRAIN else None
    return


if __name__ == "__main__":
    app.run()
