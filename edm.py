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
    # EDM from scratch (MNIST)

    Karras et al. 2022, reusing the `ConditionedUnet` trunk from `networks.py`.
    """)
    return


@app.cell
def _():
    from math import sqrt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.transforms.functional import to_pil_image
    from networks import ConditionedUnet
    from ema_pytorch import EMA

    from tqdm import tqdm
    from einops import rearrange, repeat, reduce

    return (
        ConditionedUnet,
        DataLoader,
        EMA,
        F,
        datasets,
        make_grid,
        nn,
        rearrange,
        reduce,
        sqrt,
        to_pil_image,
        torch,
        tqdm,
        transforms,
    )


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 28
    channels = 1
    batch_size = 128
    lr = 2e-4
    epochs = 5
    num_classes = 10
    emb_dim = 64
    num_steps = 18

    MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
    return (
        MNIST_MEAN,
        MNIST_STD,
        batch_size,
        channels,
        device,
        emb_dim,
        epochs,
        image_size,
        lr,
        num_classes,
    )


@app.cell
def _(DataLoader, MNIST_MEAN, MNIST_STD, batch_size, datasets, transforms):
    # Preconditioning assumes zero-mean data with RMS sigma_data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    return (train_loader,)


@app.cell
def _(ConditionedUnet, channels, device, emb_dim, num_classes):
    model = ConditionedUnet(num_classes = num_classes, dim=emb_dim, dim_mults=(1, 2, 4), channels=channels, flash_attn=False, random_fourier_features=True).to(device)
    return (model,)


@app.cell
def _():
    def norm_to_neg_one_one(x):
        return x * 2 - 1

    def unnorm_to_zero_one(x):
        return (x + 1) * 0.5

    return


@app.cell
def _(
    F,
    channels,
    device,
    image_size,
    model,
    nn,
    num_classes,
    rearrange,
    reduce,
    sqrt,
    torch,
    tqdm,
):
    class ElucidatedDiffusion(nn.Module):
        """EDM preconditioning, loss, and Heun sampler (Karras et al. 2022).

        D_theta = c_skip * x + c_out * F_theta predicts the clean image.
        Default values taken from Karras et al. 2022, Table 1
        """

        def __init__(
            self, 
            model, 
            image_size, 
            channels,
            sigma_data=1.0, 
            sigma_min=0.002, 
            sigma_max=80.0,
            rho=7.0, 
            P_mean=-1.2, 
            P_std=1.2, 
            device="cpu"
        ):
            super().__init__()
            self.network = model
            self.image_size = image_size
            self.channels = channels
            self.num_classes = num_classes
            self.sigma_data = sigma_data
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max
            self.rho = rho
            self.P_mean = P_mean
            self.P_std = P_std
            self.to(device)

        @property
        def device(self):
            return next(self.network.parameters()).device

        def denoiser_forward(self, x, sigma, y, clamp = False, guidance_scale = 1.0):
            """Preconditioned denoiser D_theta(x; sigma) (EDM eq. 7).

            Args:
                x:     (B, C, H, W) noisy images at noise level sigma
                sigma: (B, 1, 1, 1), (B, ) or (,) tensor or float noise levels
                y:     (B,) int labels. Can be null class.
            Returns:
                (B, C, H, W) predicted clean images
            """
            B = x.shape[0]        
            if not torch.is_tensor(sigma):
                sigma = torch.full((B,), sigma, device=self.device)
            elif sigma.ndim == 0: # handle 0-D tensor case
                sigma = sigma.expand(B)
            sigma_full = rearrange(sigma, 'b -> b 1 1 1')

            c_skip = self.sigma_data**2 / (sigma_full**2 + self.sigma_data**2)
            c_out = (sigma_full * self.sigma_data) / (self.sigma_data**2 + sigma_full**2)**0.5
            c_in = 1 / (sigma_full**2 + self.sigma_data**2)**0.5
            c_noise = 0.25 * torch.log(sigma)

            d_theta = c_skip * x + c_out * self.network.forward(c_in * x, c_noise, y)
            if guidance_scale != 1.0:
                y_null = torch.full_like(y, self.network.num_classes)
                d_theta_uncond = c_skip * x + c_out * self.network.forward(c_in * x, c_noise, y_null)
                d_theta = d_theta_uncond + guidance_scale * (d_theta - d_theta_uncond)
            if clamp:
                d_theta = d_theta.clamp(-1., 1.)
            return d_theta


        def forward(self, x, y, cond_drop_prob = 0.):
            """Weighted denoising loss (EDM eq. 8), with ln(sigma) ~ N(P_mean, P_std^2).
            """
            B = x.shape[0]
            # ln(sigma) drawn from N(P_mean, P_std^2) - Table 1, Training
            sigma = (self.P_mean + self.P_std * torch.randn((B,), device = self.device)).exp()
            sigma_full = rearrange(sigma, 'b -> b 1 1 1')

            noised_x = x + sigma_full * torch.randn_like(x) 
            # CFG so network learns null class
            if cond_drop_prob > 0.:
                drop = torch.rand(B, device=y.device) < cond_drop_prob
                y = torch.where(drop, torch.full_like(y, self.network.num_classes), y)

            denoised = self.denoiser_forward(noised_x, sigma, y)

            losses = reduce(F.mse_loss(denoised, x, reduction = 'none'), 'b ... -> b', 'mean')
            loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
            losses = loss_weight * losses

            return losses.mean()


        def sample_schedule(self, num_steps):
            """Sampling noise levels (EDM eq. 5), decreasing and terminating at 0.

            Returns (num_steps + 1,), rho-spaced to concentrate steps at low sigma.
            """
            if num_steps <= 1:
                raise ValueError(f"num_steps must be greater than 1, got {num_steps}")
            inv_rho = 1 / self.rho
            idx = torch.arange(num_steps, device=self.device, dtype = torch.float32)
            sigmas = (self.sigma_max**inv_rho + idx / (num_steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)).pow(self.rho)
            return F.pad(sigmas, (0, 1), value = 0.)

        def churn_schedule(self, num_steps, S_churn, S_tmin, S_tmax):
            """Noise levels (EDM eq. 5) paired with the per-step churn factor gamma_i.

            Returns (num_steps + 1,) sigmas and gammas; gamma is 0 outside [S_tmin, S_tmax].
            """
            sigmas = self.sample_schedule(num_steps)
            gammas = torch.where(
                (sigmas >= S_tmin) & (sigmas <= S_tmax),
                min(S_churn / num_steps, sqrt(2) - 1),
                0.
            )
            return sigmas, gammas

        def heun_step(self, x, sigma, sigma_next, gamma, y,
                      guidance_scale=1.0, S_noise=1.0, clamp=False):
            """One churn-then-Heun step (EDM Alg. 2, lines 4-8).

            Args:
                sigma, sigma_next, gamma: 0-D tensors for the current step
            Returns:
                (x_next, x_hat, denoised) where x_hat is the churned state at
                sigma_hat and denoised is D_theta evaluated there
            """
            eps = S_noise * torch.randn_like(x)
            sigma_hat = sigma + gamma * sigma
            x_hat = x + sqrt(sigma_hat**2 - sigma**2) * eps

            # calculate ODE tangent direction
            denoised = self.denoiser_forward(x_hat, sigma_hat, y, clamp=clamp, guidance_scale=guidance_scale)
            ode_dir = (x_hat - denoised) / sigma_hat
            # euler step from increased noise level to next noise level
            x_next = x_hat + (sigma_next - sigma_hat) * ode_dir

            if sigma_next != 0:
                # Heun second order correction
                ode_dir_next = (x_next - self.denoiser_forward(x_next, sigma_next, y, clamp=clamp, guidance_scale=guidance_scale)) / sigma_next
                x_next = x_hat + (sigma_next - sigma_hat) * 0.5  * (ode_dir + ode_dir_next)
            return x_next, x_hat, denoised

        @torch.no_grad()
        def sample(self, y, num_steps=18, guidance_scale=1.0,
                   S_churn=0.0, S_tmin=0.0, S_tmax=float("inf"), S_noise=1.0, clamp=False):
            """Heun sampler (EDM Alg. 2), starting from sigma_max * N(0, I).
            num_steps defaults to 18 to reproduce NFE=35 according to paper, since
            Heun's runs 2N-1 network evals

            S_churn = 0 reduces to the deterministic ODE sampler (Alg. 1).
            """
            input_shape = (y.shape[0], self.channels, self.image_size, self.image_size)
            sigmas, gammas = self.churn_schedule(num_steps, S_churn, S_tmin, S_tmax)

            x = sigmas[0] * torch.randn(input_shape, device=self.device)

            sigmas_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
            for sigma, sigma_next, gamma in tqdm(sigmas_gammas):
                x, _, _ = self.heun_step(x, sigma, sigma_next, gamma, y,
                                         guidance_scale=guidance_scale, S_noise=S_noise, clamp=clamp)
            return x

    edm = ElucidatedDiffusion(model, image_size=image_size, channels=channels, device=device)
    return (edm,)


@app.cell
def _(EMA, torch, tqdm):
    def train(edm, train_loader, epochs, lr):
        """Trains in place and returns the EMA copy to sample from."""
        opt = torch.optim.Adam(edm.network.parameters(), lr=lr)
        ema = EMA(edm, beta=0.999, update_after_step=100, update_every=1)
        edm.network.train()
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"epoch {epoch}")
            for x_0, y in pbar:
                x_0, y = x_0.to(edm.device), y.to(edm.device)
                loss = edm.forward(x_0, y, cond_drop_prob=0.1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ema.update()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        return ema.ema_model

    return (train,)


@app.cell
def _(edm, epochs, lr, train, train_loader):
    edm_ema = train(edm, train_loader, epochs=epochs, lr=lr)
    return (edm_ema,)


@app.cell
def _(MNIST_MEAN, MNIST_STD, make_grid, to_pil_image, torch):
    def show_samples(edm, images_per_class=5, n=10, image_size=28, num_steps=18, guidance_scale=1.5):
        edm.network.eval()
        y = torch.arange(n, device=edm.device).repeat_interleave(images_per_class)
        imgs = edm.sample(y, num_steps=num_steps, guidance_scale=guidance_scale)
        imgs = (imgs * MNIST_STD + MNIST_MEAN).clamp(0, 1)
        return to_pil_image(make_grid(imgs.cpu(), nrow=images_per_class))


    return (show_samples,)


@app.cell
def _(edm_ema, show_samples):
    show_samples(edm_ema, num_steps = 18)
    return


@app.cell
def _(torch):
    @torch.no_grad()
    def sample_trajectory(edm, y, num_steps=18, guidance_scale=1.5,
                          S_churn=0.0, S_tmin=0.0, S_tmax=float("inf"), S_noise=1.0, clamp=False):
        """Run one Heun trajectory, keeping every step.

        Returns a list of (sigma_hat, x_hat, denoised), noisiest first, with a
        final entry at sigma = 0 holding the finished sample.
        """
        edm.network.eval()
        input_shape = (y.shape[0], edm.channels, edm.image_size, edm.image_size)
        sigmas, gammas = edm.churn_schedule(num_steps, S_churn, S_tmin, S_tmax)
        x = sigmas[0] * torch.randn(input_shape, device=edm.device)

        steps = []
        for sigma, sigma_next, gamma in zip(sigmas[:-1], sigmas[1:], gammas[:-1]):
            x_next, x_hat, denoised = edm.heun_step(
                x, sigma, sigma_next, gamma, y,
                guidance_scale=guidance_scale, S_noise=S_noise, clamp=clamp
            )
            steps.append(((sigma + gamma * sigma).item(), x_hat, denoised))
            x = x_next
        steps.append((0.0, x, x))
        return steps

    return (sample_trajectory,)


@app.cell
def _(MNIST_MEAN, MNIST_STD, make_grid, to_pil_image, torch):
    def show_trajectory(steps, view="denoised"):
        """Grid of one trajectory: one row per sample, columns noisiest to cleanest.

        Args:
            view: "denoised" for D_theta's clean estimate, "x" for the churned state
        """
        frames = [d for _, _, d in steps] if view == "denoised" else [x for _, x, _ in steps]
        imgs = torch.stack(frames, dim=1).flatten(0, 1).cpu()
        if view == "denoised":
            imgs = (imgs * MNIST_STD + MNIST_MEAN).clamp(0, 1)
            return to_pil_image(make_grid(imgs, nrow=len(frames)))
        # x carries sigma_max-scale noise, so scale each tile to its own range to stay legible
        return to_pil_image(make_grid(imgs, nrow=len(frames), normalize=True, scale_each=True))

    return (show_trajectory,)


@app.cell
def _(mo):
    churn_ui = mo.ui.slider(0, 100, 1, value=0, label="S_churn", show_value=True, debounce=True)
    tmin_ui = mo.ui.slider(0.0, 10.0, 0.05, value=0.0, label="S_tmin", show_value=True, debounce=True)
    tmax_ui = mo.ui.slider(0.0, 80.0, 1.0, value=80.0, label="S_tmax", show_value=True, debounce=True)
    noise_ui = mo.ui.slider(0.0, 1.5, 0.01, value=1.0, label="S_noise", show_value=True, debounce=True)
    steps_ui = mo.ui.slider(2, 40, 1, value=18, label="num_steps", show_value=True, debounce=True)
    guidance_ui = mo.ui.slider(1.0, 5.0, 0.1, value=1.5, label="guidance_scale", show_value=True, debounce=True)
    seed_ui = mo.ui.number(0, 9999, 1, value=0, label="seed")

    mo.vstack([churn_ui, tmin_ui, tmax_ui, noise_ui, steps_ui, guidance_ui, seed_ui])
    return churn_ui, guidance_ui, noise_ui, seed_ui, steps_ui, tmax_ui, tmin_ui


@app.cell
def _(
    churn_ui,
    edm_ema,
    guidance_ui,
    mo,
    noise_ui,
    sample_trajectory,
    seed_ui,
    show_trajectory,
    steps_ui,
    tmax_ui,
    tmin_ui,
    torch,
):
    # Fixed seed so a slider move shows that parameter's effect, not a new draw
    torch.manual_seed(seed_ui.value)
    traj = sample_trajectory(
        edm_ema,
        torch.arange(4, device=edm_ema.device),
        num_steps=steps_ui.value,
        guidance_scale=guidance_ui.value,
        S_churn=churn_ui.value,
        S_tmin=tmin_ui.value,
        S_tmax=tmax_ui.value,
        S_noise=noise_ui.value,
    )

    # Show sigma/noise at each step and corresponding denoised estimate 
    # At each step, fresh noise from langevin churn raises the sample's noise level slightly. 
    # The ODE step, driven by the denoiser's clean-image estimate, then carries the noise down 
    # to the next scheduled level. The denoiser treats injected noise identically to noise already present. 
    # Adding the fresh churn exchanges some structured, off-distribution deviation for exact, 
    # on-distribution Gaussian noise (eps in Algorithm 2 line 6), so the trajectory self-corrects 
    # toward the true marginals as σ decreases. 
    # Should also see that churned state is a much more noisier version of the input.

    mo.vstack([
        mo.md(f"**D_theta estimate** &nbsp; sigma: {' -> '.join(f'{s:.3g}' for s, _, _ in traj)}"),
        show_trajectory(traj, "denoised"),
        mo.md("**churned state x_hat** &nbsp. Each tile scaled to its own range"),
        show_trajectory(traj, "x"),
    ])
    return


if __name__ == "__main__":
    app.run()
