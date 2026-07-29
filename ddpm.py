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

    Also implements classifier-free guidance for conditioned sample generation.
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
    from networks import ConditionedUnet
    from tqdm.auto import tqdm

    return (
        ConditionedUnet,
        DataLoader,
        F,
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
    T = 1000
    image_size = 28
    channels = 1
    batch_size = 128
    lr = 2e-4
    epochs = 10
    num_classes = 10
    emb_dim = 64
    return T, batch_size, channels, device, emb_dim, epochs, lr, num_classes


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
def _(ConditionedUnet, channels, device, emb_dim, num_classes):
    # Canonical Ho et al. 2020 UNet, tiny version for MNIST
    model = ConditionedUnet(num_classes = num_classes, dim=emb_dim, dim_mults=(1, 2, 4), channels=channels, flash_attn=False).to(device)
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


        def p_losses(self, x_0, t, y, cond_drop_prob=0.1):
            """L_simple: MSE between the true noise and the predicted noise (DDPM eq. 14).
            Implements CFG - replace some labels with null class with prob 0.1 
            so network can still learn to generate samples unconditionally.
            """
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise) # true noise
            drop = torch.rand(y.shape, device=y.device) < cond_drop_prob 
            y = y.masked_fill(drop, self.network.num_classes)
            eps_theta = self.network(x_t, t, y) # predict noise
            return F.mse_loss(noise, eps_theta)


        @torch.no_grad()
        def p_sample(self, x_t, t, y, guidance_scale=0.0):
            """One reverse step: sample x_{t-1} ~ p_theta(x_{t-1} | x_t) (DDPM eq. 11, Alg. 2).
            If guidance_scale/w > 0.0, sample noise also conditioned on y according to CFG:
            pred_noise = (guidance_scale + 1)pred_noise_cond - pred_noise_uncond
            Args:
                x_t: (B, C, H, W)
                t:   (B,) int64 timesteps, all equal to the current step
                y:   (B,) int labels. Can be null class. 
            Returns:
                x_{t-1}: (B, C, H, W)
            """
            z = torch.randn_like((x_t))
            z[t == 0] = 0.0
            eps_theta = self.network(x_t, t, y)
            if guidance_scale > 0: 
                y_null = torch.full_like(y, self.network.num_classes)
                eps_theta_uncond = self.network(x_t, t, y_null)
                eps_theta = eps_theta + guidance_scale * (eps_theta - eps_theta_uncond)
            alphas_i = self._gather(self.alphas, t, x_t.shape) # (B, 1, 1, 1)
            alpha_bars_i = self._gather(self.alpha_bars, t, x_t.shape) # (B, 1, 1, 1)
            betas_i = self._gather(self.betas, t, x_t.shape) # (B, 1, 1, 1)
            return 1 / torch.sqrt(alphas_i) * (x_t - (1 - alphas_i) / torch.sqrt(1 - alpha_bars_i) * eps_theta) + torch.sqrt(betas_i) * z

        @torch.no_grad()
        def sample(self, y, image_size=28, channels=1, guidance_scale=0.0, step=0):
            """Reverse sampling process (Algorithm 2) until `step`. 

            Start from x_T ~ N(0, I) of shape (B, channels, image_size, image_size),
            then loop i = T-1 .. 0 calling forward / sampling from p_theta.

            Returns the final x_0 in [-1, 1].
            """
            B = y.shape[0]
            x_t = torch.randn((B, channels, image_size, image_size), device=self.device)
            t = self.T - 1
            while t >= step:
                time_batch = torch.full((B, ), t, device=self.device, dtype=torch.long)
                x_t = self.p_sample(x_t, time_batch, y, guidance_scale)
                t -= 1
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
            for x_0, y in pbar:
                x_0, y = x_0.to(diffusion.device), y.to(diffusion.device)
                t = torch.randint(0, diffusion.T, (x_0.shape[0],), device=diffusion.device)
                loss = diffusion.p_losses(x_0, t, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (train,)


@app.cell
def _(make_grid, to_pil_image, torch):
    def show_samples(diffusion, images_per_class=5, n=16, image_size=28, channels=1, guidance_scale=1.5, step=0):
        diffusion.network.eval()
        y = torch.arange(10, device=diffusion.device).repeat_interleave(images_per_class)
        imgs = diffusion.sample(y, image_size=image_size, channels=channels, guidance_scale=guidance_scale, step=step)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        return to_pil_image(make_grid(imgs.cpu(), nrow=images_per_class))

    return


@app.cell
def _(torch):
    def sample_trajectory(diffusion, y, steps, image_size=28, channels=1, guidance_scale=1.5):
        """Run one reverse trajectory, keeping x_t at each timestep in `steps`.

        Args:
            y:     (B,) int labels
            steps: iterable of timesteps to keep, e.g. [1000, 900, 500, 100, 0]
        Returns:
            {t: (B, C, H, W)} in [-1, 1]
        """
        diffusion.network.eval()
        wanted = set(steps)
        B = y.shape[0]
        x_t = torch.randn((B, channels, image_size, image_size), device=diffusion.device)
        snapshots = {diffusion.T: x_t.clone()} if diffusion.T in wanted else {}
        for t in range(diffusion.T - 1, min(wanted) - 1, -1):
            time_batch = torch.full((B,), t, device=diffusion.device, dtype=torch.long)
            x_t = diffusion.p_sample(x_t, time_batch, y, guidance_scale)
            if t in wanted:
                snapshots[t] = x_t.clone()
        return snapshots

    return (sample_trajectory,)


@app.cell
def _(make_grid, to_pil_image, torch):
    def show_trajectory(snapshots):
        """Grid of a trajectory: one row per sample, columns noisiest to cleanest.
        Column order is `sorted(snapshots, reverse=True)`.
        """
        order = sorted(snapshots, reverse=True)
        imgs = torch.stack([snapshots[t] for t in order], dim=1).flatten(0, 1)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        return to_pil_image(make_grid(imgs.cpu(), nrow=len(order)))

    return (show_trajectory,)


@app.cell
def _():
    TRAIN = False
    return (TRAIN,)


@app.cell
def _(TRAIN, diffusion, epochs, lr, model, train, train_loader):
    if TRAIN:
        train(model, diffusion, train_loader, epochs=epochs, lr=lr)
    return


@app.cell
def _(diffusion, sample_trajectory, torch):
    y = torch.arange(10, device=diffusion.device)
    x_steps = sample_trajectory(diffusion, y, [999, 750, 500, 250, 100, 0])
    return (x_steps,)


@app.cell
def _(show_trajectory, x_steps):
    show_trajectory(x_steps)
    return


if __name__ == "__main__":
    app.run()
