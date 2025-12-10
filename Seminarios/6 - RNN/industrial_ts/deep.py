import torch
import torch.nn as nn

from .tsdiffusion import TSDiffusion


class DeepEncoder(nn.Module):
    """
    Simple per-timestep MLP encoder that concatenates timestamps to the latent
    representation before applying feedforward blocks.
    """
    def __init__(
        self,
        hidden_dim: int,
        mlp_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_layers = max(1, int(mlp_layers))
        self.dropout = float(max(0.0, dropout))
        self.use_layernorm = use_layernorm
        self.act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        layers: list[nn.Module] = []
        in_dim = hidden_dim + 1  # hidden state + timestamp
        for layer_idx in range(self.mlp_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.act)
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            if self.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor, only_gru: bool = False) -> torch.Tensor:  # noqa: ARG002
        ts = timestamps.unsqueeze(-1).to(dtype=x.dtype)
        inp = torch.cat([x, ts], dim=-1)
        return self.net(inp)


class TSDF_DEEP(TSDiffusion):
    """
    Diffusion model variant that replaces the GRU encoders with per-timestep MLPs.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        status_dim: int = 0,
        lam: list[float, float, float, float, float, float] = [0.9, 0.0, 0.0, 0.1, 0.0, 0.0],
        num_steps: int = 1000,
        cost_columns: list | None = None,
        log_likelihood: bool = False,
        mlp_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
        sigma_temp: float = 0.7,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            static_dim=static_dim,
            status_dim=status_dim,
            lam=lam,
            num_steps=num_steps,
            cost_columns=cost_columns,
            log_likelihood=log_likelihood,
            sigma_temp=sigma_temp,
        )
        self.encoder = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(),
        )
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )
        encoder_args = dict(
            hidden_dim=hidden_dim,
            mlp_layers=mlp_layers,
            activation=activation,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        if status_dim > 0:
            self.tmax_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            self.encoder_ode_tmax = DeepEncoder(**encoder_args)
            if self.log_likelihood:
                self.lambda_tmax_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
        if self.lam[3] > 0.0:
            self.miss_head = nn.Linear(hidden_dim, 1)
        if self.lam[0] > 0.0 or self.lam[4] > 0.0:
            self.encoder_ode_x = DeepEncoder(**encoder_args)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, in_channels)
            )
            if log_likelihood:
                self.lambda_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)
                )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool = False,
        return_x_hat: bool = False,
        mask: torch.Tensor = None,
        mask_ts: torch.Tensor = None,
        test: bool = True,
        only_gru: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        only_gru = True
        noise = None
        noise_hat = None
        vae_x = None
        vae_mu = None
        vae_logvar = None
        vae_logvar_obs = None
        vae_tmax = None
        vae_tmax_mu = None
        vae_tmax_logvar = None
        vae_tmax_logvar_obs = None
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        if mask_ts is None and mask is not None:
            mask_ts = mask.any(dim=2, keepdim=True).float()
        if not already_latent:
            h = self.encoder(torch.cat([x, mask], dim=-1))
            if not test and self.lam[1] > 0:
                noise = torch.randn_like(h) * mask_ts
                ab = self.alpha_bar[t].view(-1, 1, 1)
                h = torch.sqrt(ab) * h + torch.sqrt(1 - ab) * noise
            else:
                t = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
                noise = None
        else:
            h = x
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)
            h = h + se
        if self.lam[0] > 0.0 or self.lam[4] > 0.0:
            h = self.encoder_ode_x(h, timestamps, only_gru)
        if self.lam[2] > 0.0 or self.lam[5] > 0.0:
            ht = self.encoder_ode_tmax(h, timestamps, only_gru)
            tmax_hat = self.tmax_head(ht)
        else:
            ht = None
            tmax_hat = None
        if self.lam[4] > 0:
            mu_logvar = self.vae_latent(h)
            vae_mu, vae_logvar = torch.chunk(mu_logvar, 2, dim=-1)
            std = torch.exp(0.5 * vae_logvar)
            eps = torch.randn_like(std)
            z_vae = vae_mu + eps * std
            vae_x = self.vae_decoder(z_vae)
            vae_logvar_obs = self.vae_sigma_head(z_vae).clamp(min=-5.0, max=5.0)
        if self.lam[5] > 0 and ht is not None:
            mu_logvar_t = self.vae_tmax_latent(ht)
            vae_tmax_mu, vae_tmax_logvar = torch.chunk(mu_logvar_t, 2, dim=-1)
            std_t = torch.exp(0.5 * vae_tmax_logvar)
            eps_t = torch.randn_like(std_t)
            z_tmax = vae_tmax_mu + eps_t * std_t
            vae_tmax = self.vae_tmax_decoder(z_tmax)
            vae_tmax_logvar_obs = self.vae_tmax_sigma_head(z_tmax).clamp(min=-5.0, max=5.0)

        x_hat = self.decoder(h) if return_x_hat and self.lam[0] > 0 else None

        return (
            h,
            h,
            ht,
            x_hat,
            tmax_hat,
            noise,
            noise_hat,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_tmax,
            vae_tmax_mu,
            vae_tmax_logvar,
            vae_logvar_obs,
            vae_tmax_logvar_obs,
        )
