import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.fc(x)
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        return alpha * x


class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        (TODO) Build a noise estimating network.

        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        
        layers = []
        dims = [dim_in] + dim_hids + [dim_out]  # [입력, 히든들..., 출력] 차원 리스트
        
        # TimeLinear 레이어들을 연결하여 네트워크 구성
        for i in range(len(dims) - 1):
            layers.append(TimeLinear(dims[i], dims[i+1], num_timesteps))  # 시간 조건부 선형 레이어
            if i < len(dims) - 2:  # 마지막 레이어 제외
                layers.append(nn.ReLU())  # 활성화 함수
        
        self.layers = nn.ModuleList(layers)  # ModuleList로 저장 (forward에서 순차 처리)
        
        # TimeLinear는 시간 t에 따라 다른 변환을 적용하는 레이어
        # DDPM에서 각 시간 단계마다 다른 노이즈 레벨 처리 필요
        # 네트워크가 현재 timestep 인식하고 적절한 노이즈 예측
        # 마지막 레이어는 노이즈 직접 출력하므로 ReLU 미적용
        
        ######################
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """ㅁ
        ######## TODO ########
        # DO NOT change the code outside this part.
        
        # ModuleList의 레이어들을 순차적으로 적용
        for layer in self.layers:
            if isinstance(layer, TimeLinear):  # TimeLinear 레이어인 경우
                x = layer(x, t)  # (x, t) 두 개 입력 전달
            else:  # ReLU 같은 activation 레이어인 경우
                x = layer(x)  # x만 전달
        
        # TimeLinear는 (x, t) 두 입력을 받아 시간 조건부 변환 수행
        # ReLU는 일반적인 activation이므로 x만 받음
        # 최종 출력은 입력과 같은 차원의 노이즈 예측값
        
        ######################
        return x
