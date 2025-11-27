from .ode_jump import ODEJump
from .tsdiffusion import TSDiffusion
import torch.nn as nn
import torch


class GRU(nn.Module):
    def __init__(
        self,
        hidden_dim,
        bi_gru,
        bi_method,
        bi_coupled
    ):
        super().__init__()
        self.bi_gru = bi_gru
        self.bi_method = bi_method
        self.bi_coupled = bi_coupled
        self.hidden_dim=hidden_dim
        self.gru = nn.GRUCell(hidden_dim,hidden_dim)
        if bi_gru:
            self.gru_bw = nn.GRUCell(hidden_dim,hidden_dim)
            if bi_method == 'gate':
                self.bi_gate = nn.Sequential(
                    nn.Linear(hidden_dim*2,hidden_dim),
                    nn.Sigmoid()
                )
            elif bi_method == 'gru':
                self.gru_fuser = nn.GRUCell(hidden_dim*2,hidden_dim)
    def forward(self, x):
        """
        x  : (B, T, C)
        ts : (B, T)   segundos unix (normalizados ou não)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        states = []
        if self.bi_gru:

            states_bw = []
            for i in reversed(range(T)):
                h = self.gru_bw(x[:, i], h)                                 # jump
                states_bw.append(h)
            states_bw.reverse()
            if not self.bi_coupled:
                h = torch.zeros(B, self.hidden_dim, device=x.device)

        for i in range(T):
            h = self.gru(x[:, i], h)                                 # jump
            states.append(h)
        if self.bi_gru:
            states_concat = [torch.cat([f,b],dim=-1) for f,b in zip(states,states_bw)]
            if self.bi_method == 'concat':
                states = states_concat
            elif self.bi_method == 'gate':
                states_concat = torch.stack(states_concat, dim=1)
                states = torch.stack(states, dim=1)
                states_bw = torch.stack(states_bw, dim=1)
                sigma = self.bi_gate(states_concat)
                states = sigma * states + (1-sigma) * states_bw
            elif self.bi_method == 'gru':
                states_concat = torch.stack(states_concat, dim=1)
                h = torch.zeros(B, self.hidden_dim, device=x.device)
                states = []
                for i in range(T):
                    h = self.gru_fuser(states_concat[:,i],h)
                    states.append(h)
        if self.bi_method != 'gate':
            H = torch.stack(states, dim=1)   # (B, T, hidden_dim*2)
        else:
            H = states  # (B, T, hidden_dim)

        return H

class TS_GRU(ODEJump):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        denoised: bool = False,
        lam: list[float,float] = [0.9, 0.1],
        cost_columns: list = None,
        bi_gru: bool = False,
        bi_method: str = 'concat',
        bi_coupled: bool = False
        
    ):
        self.lam = lam
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam, cost_columns)
        self.val_loss = float('inf')
        self.model_dim = hidden_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim if not (bi_gru and bi_method=='concat') else hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, in_channels),
        )
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )
        self.gru = GRU(hidden_dim,bi_gru,bi_method,bi_coupled)
        # (d) m_b  — probabilidade de observação (Bernoulli) para L4
        self.miss_head = nn.Linear(hidden_dim if not (bi_gru and bi_method=='concat') else hidden_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None,
        x_denoised: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        # Embedding de entrada
        if not already_latent:
            if x_denoised is not None:
                # aplique gate (treino=True quando model.training)
                x_fused, _ = self.denoise_gate(x, x_denoised, mask, train_mode=self.training)
                h_in = torch.cat([x_fused, mask], dim=-1)
            else:
                h_in = torch.cat([x, mask], dim=-1)

            h = self.encoder(h_in)  # (B,T,hidden_dim)
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        #tm_e = self.time_encoding(timestamps.to(h.dtype)).to(h.dtype)  # tempo contínuo
        #h = h + tm_e
        h = self.gru(h)
        state = h
        return state,self.decoder(state) if return_x_hat else None    
        
class GRUEncoder(nn.Module):
    """
    Self‑Attentive Jump‑ODE simplificado:
    - GRUCell executa o *jump* g_ψ na chegada de cada evento (x_i, t_i)
    - ODEFunc integra h(t) entre eventos.
    - Self‑attention usa máscara para faltantes (opcional).
    """
    def __init__(self, in_dim, hidden_dim, bi_gru, bi_method, bi_coupled):
        super().__init__()
        self.bi_gru = bi_gru
        self.bi_method = bi_method
        self.bi_coupled = bi_coupled
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.gru = nn.GRUCell(in_dim, in_dim)
        self.norm_x = nn.LayerNorm(in_dim)
        self.norm_H = nn.LayerNorm(in_dim if not (bi_gru and bi_method=='concat') else in_dim*2)
        if bi_gru:
            self.gru_bw = nn.GRUCell(in_dim, in_dim)
            if bi_method == 'gate':
                self.bi_gate = nn.Sequential(
                    nn.Linear(hidden_dim*2,hidden_dim),
                    nn.Sigmoid()
                )
            elif bi_method == 'gru':
                self.gru_fuser = nn.GRUCell(hidden_dim*2,hidden_dim)

        if in_dim != hidden_dim:
            self.encoder=nn.Sequential(
                nn.Linear(in_dim,hidden_dim*4),
                nn.GELU(),
                nn.Linear(hidden_dim*4,hidden_dim)
            )
 

    def forward(self, x, ts, only_gru=False):
        B, T, C = x.shape
        #eps = 1e-6
        h = torch.zeros(B, self.in_dim, device=x.device)
        states = []
        x = self.norm_x(x)
        if self.bi_gru:

            states_bw = []
            for i in reversed(range(T)):
                h = self.gru_bw(x[:, i], h)                                 # jump
                states_bw.append(h)
            states_bw.reverse()
            if not self.bi_coupled:
                h = torch.zeros(B, self.hidden_dim, device=x.device)
        for i in range(T):

            # JUMP no evento i (como no esquema original)
            h = self.gru(x[:, i], h)
            states.append(h)
        if self.bi_gru:
            states_concat = [torch.cat([f,b],dim=-1) for f,b in zip(states,states_bw)]
            if self.bi_method == 'concat':
                states = states_concat
            elif self.bi_method == 'gate':
                states_concat = torch.stack(states_concat, dim=1)
                states = torch.stack(states, dim=1)
                states_bw = torch.stack(states_bw, dim=1)
                sigma = self.bi_gate(states_concat)
                states = sigma * states + (1-sigma) * states_bw
            elif self.bi_method == 'gru':
                states_concat = torch.stack(states_concat, dim=1)
                h = torch.zeros(B, self.hidden_dim, device=x.device)
                states = []
                for i in range(T):
                    h = self.gru_fuser(states_concat[:,i],h)
                    states.append(h)

        if self.bi_method != 'gate':
            H = torch.stack(states, dim=1)   # (B, T, hidden_dim*2)
        else:
            H = states  # (B, T, hidden_dim)
        H = self.norm_H(H)
        return H if self.in_dim == self.hidden_dim else self.encoder(H)

class TSDF_GRU(TSDiffusion):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        status_dim: int = 0,
        lam: list[float,float,float,float] = [0.9, 0.0, 0.0, 0.1],
        num_steps: int = 1000,
        cost_columns: list = None,
        bi_gru: bool = False,
        bi_method: str = 'concat',
        bi_coupled: bool = False,
        log_likelihood: bool = False
        ):
        super().__init__(        
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            static_dim=static_dim,
            lam=lam,
            num_steps=num_steps,
            cost_columns=cost_columns,
            status_dim=status_dim,
            log_likelihood=log_likelihood
        )
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_dim),
            nn.ReLU(),
        )

        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )

        if status_dim > 0:
            self.tmax_head = nn.Sequential(
                nn.Linear(hidden_dim  if not (bi_gru and bi_method=='concat') else hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            self.encoder_ode_tmax = GRUEncoder(hidden_dim, hidden_dim, bi_gru, bi_method,bi_coupled)
            if self.log_likelihood:
                self.lambda_tmax_head = nn.Sequential(
                    nn.Linear(hidden_dim  if not (bi_gru and bi_method=='concat') else hidden_dim * 2, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)       # escalar
                )   
        if self.lam[3] > 0.0:  
            self.miss_head = nn.Linear(hidden_dim if not (bi_gru and bi_method=='concat') else hidden_dim * 2, 1)
        if self.lam[0] > 0.0:
            self.encoder_ode_x = GRUEncoder(hidden_dim, hidden_dim, bi_gru, bi_method,bi_coupled)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim if not (bi_gru and bi_method=='concat') else hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, in_channels),
            )
            if log_likelihood:
                self.lambda_head = nn.Sequential(
                    nn.Linear(hidden_dim  if not (bi_gru and bi_method=='concat') else hidden_dim * 2, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)       # escalar
                )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask: torch.Tensor = None,
        mask_ts: torch.Tensor = None,
        test: bool=True,
        only_gru: bool = False
    ) -> torch.Tensor:
        only_gru = True
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        noise = None
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        # Embedding de entrada
        if not already_latent:
            h = self.encoder(torch.cat([x, mask], dim=-1))
            if not test and self.lam[1]>0:
                noise = torch.randn_like(h) * mask_ts
                ab = self.alpha_bar[t].view(-1, 1, 1)
                h = torch.sqrt(ab) * h + torch.sqrt(1 - ab) * noise
            else:
                t = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
                noise = torch.zeros_like(h)
        else:
            h = x
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if self.lam[0]>0:
            h = self.encoder_ode_x(h, timestamps, only_gru)
        if self.lam[2]>0:
            ht = self.encoder_ode_tmax(h, timestamps, only_gru)
            tmax_hat = self.tmax_head(ht)
        else:
            ht = None
            tmax_hat = None

        if test:
            return h,h,ht,self.decoder(h) if return_x_hat and self.lam[0]>0 else None,tmax_hat
        else:
            return h,h,ht,self.decoder(h) if return_x_hat and self.lam[0]>0 else None,tmax_hat, noise, noise