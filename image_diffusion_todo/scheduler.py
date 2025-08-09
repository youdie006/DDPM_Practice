from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            x_t (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int`): current timestep in a reverse process.
            eps_theta (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM reverse step.
        
        # t를 배치 크기에 맞춰 텐서로 변환
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        
        # 필요한 파라미터들 추출
        alpha_t = self._get_teeth(self.alphas, t_tensor)  # αt
        alpha_cumprod_t = self._get_teeth(self.alphas_cumprod, t_tensor)  # ᾱt
        beta_t = self._get_teeth(self.betas, t_tensor)  # βt
        
        # eps_factor = (1-αt)/√(1-ᾱt)
        eps_factor = (1 - alpha_t) / (1 - alpha_cumprod_t).sqrt()
        
        # 평균 계산: μθ = 1/√αt * (xt - eps_factor * εθ)
        mean = (x_t - eps_factor * eps_theta) / alpha_t.sqrt()
        
        # 분산 추가 (t > 0일 때만)
        if t > 0:
            sigma_t = self._get_teeth(self.sigmas, t_tensor)  # σt
            noise = torch.randn_like(x_t)
            sample_prev = mean + sigma_t * noise  # stochastic sampling
        else:
            sample_prev = mean  # t=0에서는 deterministic
        
        # Task 1의 p_sample과 동일한 로직
        # 이미지의 경우 차원만 다름 [B,C,H,W] vs [B,2]
        
        #######################
        
        return sample_prev
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps       = torch.randn(x_0.shape, device='cuda')

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        
        # alphas_cumprod의 t번째 값 추출 (ᾱt)
        alpha_cumprod_t = self._get_teeth(self.alphas_cumprod, t)  # [B] -> [B,1,1,1]
        
        # Forward diffusion 수식: q(xt|x0) = N(xt; √(ᾱt)x0, (1-ᾱt)I)
        # xt = √(ᾱt) * x0 + √(1-ᾱt) * ε
        x_t = alpha_cumprod_t.sqrt() * x_0 + (1 - alpha_cumprod_t).sqrt() * eps
        
        # Task 1의 q_sample과 동일한 로직
        # _get_teeth로 4D 텐서에 맞게 shape 조정
        # 이미지 [B,C,H,W]와 브로드캐스팅 가능
        #######################

        return x_t, eps
