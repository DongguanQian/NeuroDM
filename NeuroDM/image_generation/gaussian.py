import math
from inspect import isfunction
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    out = repeat_noise() if repeat else noise()
    return out


def cosine_beta_schedule(timesteps, c_l=0, s=0.008):
    if c_l == 0:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    else:
        steps = timesteps + 1
        betas = torch.linspace(0.0001, 0.999, steps)
    res = torch.clip(betas, 0, 0.999)
    return res


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            *,
            image_size=64,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            c_l=0
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps=timesteps, c_l=c_l)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, label=None, flag=0):
        noise = self.denoise_fn(x, t, label)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if flag == 1:
            model_output, model_var_values = torch.split(noise, 3, dim=1)
            posterior_log_variance = model_var_values
            posterior_variance = torch.exp(posterior_log_variance)

        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
        }

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        gradient = cond_fn(x, self.betas[t], **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    @torch.no_grad()
    def p_sample(self, x, t, label=None, clip_denoised=True, repeat_noise=False, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, x.device
        out = self.p_mean_variance(x=x, t=t, label=label, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)

        return out["mean"] + nonzero_mask * (0.5 * out["log_variance"]).exp() * noise

    @torch.no_grad()
    def sample(self, batch_size=64, label=None, img=None, cond_fn=None, model_kwargs=None):
        if label is not None:
            batch_size = len(label)
        if img is None:
            img = torch.randn((batch_size, 3, self.image_size, self.image_size)).to(device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long), label=label,
                                cond_fn=cond_fn, model_kwargs=model_kwargs)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)).to(device)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t=None, label=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, label)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self,
                x,
                label=None,
                *args,
                **kwargs
                ):
        b, c, h, w, _, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, label, *args, **kwargs)
