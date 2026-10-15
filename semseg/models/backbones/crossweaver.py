import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
import torch.nn.init as init


class MLP(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


class DWConv(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class CustomDWConv(nn.Module):

    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class CustomPWConv(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):

    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            xx = x.permute(0, 2, 1).reshape(B, C, H, W)
            xx = self.sr(xx).reshape(B, C, -1).permute(0, 2, 1)
            xx = self.norm(xx)
        else:
            xx = x
        k, v = self.kv(xx).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ChannelAttentionBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2)


class MixFFN(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.dwconv5 = CustomDWConv(c2, 5)
        self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))


class SAF(nn.Module):
    def __init__(self, channels, num_modals):
        super().__init__()
        self.channels = channels
        self.num_modals = num_modals
        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])
        self.mix_ffn = nn.ModuleList([
            MixFFN(self.channels[0], self.channels[0]),
            MixFFN(self.channels[1], self.channels[1]),
            MixFFN(self.channels[2], self.channels[2]),
            MixFFN(self.channels[3], self.channels[3]),
        ])
        self.channel_attns = nn.ModuleList([
            ChannelAttentionBlock(self.channels[0]),
            ChannelAttentionBlock(self.channels[1]),
            ChannelAttentionBlock(self.channels[2]),
            ChannelAttentionBlock(self.channels[3]),
        ])
        self.stage_modality_gates = nn.ModuleList([
            ModalityScalarGate(channel) for channel in self.channels
        ])

    def forward(self, x, layer_idx):
        if len(x) != self.num_modals or not x:
            raise ValueError(
                f"SAF expects {self.num_modals} feature streams, received {len(x)}"
            )
        B, C, H, W = x[0].shape
        modality_tokens = [feature.flatten(2).transpose(1, 2) for feature in x]
        weights = self.stage_modality_gates[layer_idx](modality_tokens)
        concatenated = torch.cat(x, dim=1).flatten(2).transpose(1, 2)
        x_sum = self.liner_fusion_layers[layer_idx](concatenated)
        x_sum = self.mix_ffn[layer_idx](x_sum, H, W) + self.channel_attns[layer_idx](x_sum, H, W)
        residual = torch.zeros_like(x[0])
        for index, feature in enumerate(x):
            residual = residual + weights[:, index].reshape(B, 1, 1, 1) * feature
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2) + residual

mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}


class CoverageHead(nn.Module):

    def __init__(self, dim, hidden=None):
        super().__init__()
        h = hidden or max(32, dim // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, h), nn.GELU(),
            nn.Linear(h, 1), nn.Sigmoid()
        )
        last = self.net[-2]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, g):  # [B,C]
        return self.net(g).squeeze(-1)  # [B]


def coverage_range_by_stage(sr_ratio: int):
    if   sr_ratio >= 8: return (0.70, 0.95)
    elif sr_ratio >= 4: return (0.75, 0.97)
    elif sr_ratio == 2: return (0.80, 0.98)
    else:               return (0.85, 0.99)


class ModalityScalarGate(nn.Module):

    def __init__(self, dim, init_tau=1.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(dim, 1, bias=True)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_tau)))
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.cached_weights = None

    def forward(self, x_list):
        g_list = [self.pool(x.transpose(1,2)).squeeze(-1) for x in x_list]  # [B,C]×M
        G = torch.stack(g_list, dim=1)                                      # [B,M,C]
        S = self.proj(G).squeeze(-1)                                        # [B,M]
        tau = torch.exp(self.logit_scale).clamp(0.1, 10.0)
        W_hat = torch.softmax(S / tau, dim=1)                               # [B,M]
        W = W_hat
        self.cached_weights = W.detach()
        return W


class TokenGateSoft(nn.Module):

    def __init__(self, dim, init_tau=1.5):
        super().__init__()
        h = max(8, dim // 4)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, h), nn.GELU(),
            nn.Linear(h, 1)
        )
        nn.init.zeros_(self.scorer[-1].weight); nn.init.zeros_(self.scorer[-1].bias)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_tau))))

    def score(self, x):  # x:[B,N,C]
        return torch.sigmoid(self.scorer(x)).squeeze(-1)  # [B,N]∈(0,1)

    def soft_mask(self, scores, coverage, min_keep_frac=None):
        B, N = scores.shape
        if isinstance(coverage, (float, int)):
            coverage = torch.full((B,), float(coverage), device=scores.device, dtype=scores.dtype)
        coverage = coverage.clamp(0.5, 0.999)
        with torch.no_grad():
            s_sorted, _ = scores.sort(dim=1, descending=True)            # [B,N]
            k = (coverage * N).ceil().to(dtype=torch.long, device=scores.device)
            k_min = 1 if (min_keep_frac is None) else max(1, int(N * float(min_keep_frac)))
            k = torch.maximum(k, torch.full_like(k, k_min))
            idx = torch.clamp(k - 1, 0, N - 1)
            thr = torch.gather(s_sorted, 1, idx.unsqueeze(-1)).squeeze(-1)  # [B]
        tau = torch.exp(self.log_tau).clamp(0.2, 5.0)
        w = torch.sigmoid((scores - thr.unsqueeze(-1)) / tau)               # [B,N]
        return w
def multigrids_by_stage(sr_ratio):
    if   sr_ratio >= 8: return [(11,11), (7,7), (3,3)]
    elif sr_ratio >= 4: return [(7,7), (3,3)]
    elif sr_ratio == 2: return [(3,3)]
    else:               return [(2,2)]


class MultiScaleXAttnRPB(nn.Module):

    def __init__(self, dim, heads=8, grids=None, drop=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.h = heads
        self.d = dim // heads
        self.grids = grids or [(7,7),(3,3)]
        self.scale = self.d ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k_list = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in self.grids])
        self.v_list = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in self.grids])
        self.log_sigma = nn.ParameterList([nn.Parameter(torch.zeros(self.h, 2)) for _ in self.grids])
        hidden = max(32, dim // 4)
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, len(self.grids))
        )
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)
        self.gamma = nn.Parameter(torch.tensor(2e-2))
        nn.init.zeros_(self.proj.weight)


    def _pool_grid(self, src, hw, Gh, Gw):
        B, N, C = src.shape
        H, W = hw
        fmap = src.transpose(1,2).reshape(B, C, H, W)
        pooled = F.adaptive_avg_pool2d(fmap, (Gh, Gw)).flatten(2).transpose(1,2)
        return pooled


    def _gauss_bias(self, H, W, Gh, Gw, log_sigma):
        device = log_sigma.device
        ys_q = torch.linspace(-1, 1, steps=H, device=device)
        xs_q = torch.linspace(-1, 1, steps=W, device=device)
        yy_q, xx_q = torch.meshgrid(ys_q, xs_q, indexing='ij')
        qxy = torch.stack([yy_q, xx_q], dim=-1).reshape(H * W, 2)        # [N,2]
        ys_k = torch.linspace(-1, 1, steps=Gh, device=device)
        xs_k = torch.linspace(-1, 1, steps=Gw, device=device)
        yy_k, xx_k = torch.meshgrid(ys_k, xs_k, indexing='ij')
        kxy = torch.stack([yy_k, xx_k], dim=-1).reshape(Gh * Gw, 2)      # [K,2]
        diff = qxy.unsqueeze(1) - kxy.unsqueeze(0)                        # [N,K,2]
        sigma = torch.exp(log_sigma).clamp(0.25, 3.0)                     # [h,2]
        sx2 = (sigma[:, 0] ** 2).view(1, self.h, 1, 1)                    # [1,h,1,1]
        sy2 = (sigma[:, 1] ** 2).view(1, self.h, 1, 1)
        dx2 = (diff[..., 0] ** 2).unsqueeze(0).unsqueeze(0)               # [1,1,N,K]
        dy2 = (diff[..., 1] ** 2).unsqueeze(0).unsqueeze(0)
        bias = -0.5 * (dx2 / sx2 + dy2 / sy2)                             # [1,h,N,K]
        return bias


    def forward(self, tgt, src, hw):
        B, Nt, C = tgt.shape
        H, W = hw
        q = self.q(tgt).view(B, Nt, self.h, self.d).transpose(1, 2)   # [B,h,Nt,d]
        g_src = src.transpose(1, 2).mean(-1)                           # [B,C]
        alpha = torch.softmax(self.scale_gate(g_src), dim=-1)          # [B,S]

        outs_per_scale = []
        for s, (Gh, Gw) in enumerate(self.grids):
            src_kv = self._pool_grid(src, hw, Gh, Gw)                  # [B,K,C]
            k = self.k_list[s](src_kv).view(B, -1, self.h, self.d).transpose(1, 2)
            v = self.v_list[s](src_kv).view(B, -1, self.h, self.d).transpose(1, 2)

            # Compute attention softmax in float32 with numerical safeguards.
            logits = (q @ k.transpose(-2, -1)) * self.scale            # [B,h,Nt,K]
            bias = self._gauss_bias(H, W, Gh, Gw, self.log_sigma[s]).to(logits.dtype)
            logits = (logits + bias).float()                           # float32 softmax
            logits = logits - logits.max(dim=-1, keepdim=True)[0]      # stable logits
            attn = torch.softmax(logits, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

            y = (attn @ v.float())                                     # keep fp32
            y = y.transpose(1, 2).reshape(B, Nt, self.h * self.d).to(tgt.dtype)
            outs_per_scale.append(y)

        Y = torch.stack(outs_per_scale, dim=2)                         # [B,Nt,S,C]
        alpha_h = alpha.view(B, 1, len(self.grids), 1)
        Y = (Y * alpha_h).sum(dim=2)
        Y = self.proj(self.drop(Y))
        return self.gamma * Y


class ChannelGate(nn.Module):

    def __init__(self, dim, reduction=16):
        super().__init__()
        h = max(4, dim // reduction)
        self.fc1 = nn.Linear(dim, h, bias=False)
        self.fc2 = nn.Linear(h, dim, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x, hw):
        g = x.transpose(1,2).mean(-1)                  # [B,C]
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(g))))
        return x * s.unsqueeze(1)


class FilmAdapterLitePlus(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.chan = ChannelGate(dim, reduction=16)
        self.to_scale = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=True))
        self.to_shift = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=True))
        self.gamma = nn.Parameter(torch.tensor(2e-2))
        nn.init.zeros_(self.to_scale[-1].weight); nn.init.zeros_(self.to_scale[-1].bias)
        nn.init.zeros_(self.to_shift[-1].weight); nn.init.zeros_(self.to_shift[-1].bias)

    def forward(self, tgt, src, hw):
        src = self.chan(src, hw)
        g = src.transpose(1,2).mean(-1)          # [B,C]
        s = torch.tanh(self.to_scale(g))         # [-1,1]
        b = self.to_shift(g)
        gain = 1.0 + 0.5 * s
        return self.gamma * (tgt * gain.unsqueeze(1) + b.unsqueeze(1))


class MIB(nn.Module):

    def __init__(self, dim, head, sr_ratio=1, dpr=0., num_modalities=2,
                 enable_cross_stage1=False, mqa_heads=None, min_keep_frac=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim*4))


        self.mod_gate  = ModalityScalarGate(dim)
        self.chan_gate = ChannelGate(dim, reduction=16)
        self.tok_gate  = TokenGateSoft(dim, init_tau=1.5)
        self.cov_head  = CoverageHead(dim)

        self.cov_min, self.cov_max = coverage_range_by_stage(sr_ratio)
        self.alpha_min, self.alpha_max = 0.10, 0.50

        self.min_keep_frac = float(min_keep_frac)

        self.enable_cross = enable_cross_stage1 or (sr_ratio < 8)

        grids = multigrids_by_stage(sr_ratio)
        h_cross = max(4, min(8, head if mqa_heads is None else mqa_heads))
        self.xattn_att = nn.ModuleDict()
        self.film_mlp  = nn.ModuleDict()
        for i in range(num_modalities):
            for j in range(i+1, num_modalities):
                k = f"{i}-{j}"
                self.xattn_att[k] = MultiScaleXAttnRPB(dim, heads=h_cross, grids=grids, drop=0.0)
                self.film_mlp[k]  = FilmAdapterLitePlus(dim)

        self.router_dim = max(32, dim // 2)
        self.router_q   = nn.Linear(dim, self.router_dim, bias=False)  # for target tokens
        self.router_k   = nn.Linear(dim, self.router_dim, bias=False)  # for source global
        self.router_tau = nn.Parameter(torch.tensor(1.0))              # attention routing temperature
        self.router_topk = 2

        self.router_tau_film = nn.Parameter(torch.tensor(1.2))

    @staticmethod

    def _k(i, j):
        return f"{i}-{j}" if i < j else f"{j}-{i}"


    def _predict_coverage(self, src_tokens, modal_weight, W_all=None):
        g = src_tokens.transpose(1,2).mean(-1)
        cov = self.cov_head(g)
        cov = self.cov_min + (self.cov_max - self.cov_min) * cov
        if W_all is not None:
            eps = 1e-8
            H = -(W_all.clamp_min(eps).log() * W_all).sum(dim=1) / math.log(W_all.shape[1])  # [B]∈[0,1]
            alpha = self.alpha_min + (1 - H) * (self.alpha_max - self.alpha_min)
            cov_modal = self.cov_min + (self.cov_max - self.cov_min) * modal_weight
            cov = (1 - alpha) * cov + alpha * cov_modal
        return cov.clamp(self.cov_min, self.cov_max)


    def forward(self, inputs, H, W):
        """
        inputs: list of [B, N, C]
        return: list of [B, N, C]
        """
        M = len(inputs)

        outs = [x + self.drop_path(self.attn(self.norm1(x), H, W)) for x in inputs]

        if self.enable_cross:
            W_modal = self.mod_gate(inputs)     # [B,M]
            base    = [o for o in outs]         # fixed target query bank

            src_bank = []   # [M] of [B,N,C]
            K_src    = []   # [M] of [B,Dr]（router key）
            for i in range(M):
                src_i = self.norm1(inputs[i])
                src_i = self.chan_gate(src_i, (H, W))
                scores = self.tok_gate.score(src_i)                                   # [B,N]
                cov_i  = self._predict_coverage(src_i, W_modal[:, i], W_all=W_modal)  # [B]
                w_tok  = self.tok_gate.soft_mask(scores, cov_i, self.min_keep_frac)   # [B,N]
                src_i  = src_i * w_tok.unsqueeze(-1) * W_modal[:, i].view(-1,1,1)
                src_bank.append(src_i)
                g_i = src_i.transpose(1,2).mean(-1)                                   # [B,C]
                K_src.append(self.router_k(g_i))                                      # [B,Dr]

            msgs = [[None]*M for _ in range(M)]  # message from source i to target j
            for i in range(M):
                for j in range(M):
                    if i == j: continue
                    msg_ij = self.drop_path(
                        self.xattn_att[self._k(i, j)](base[j], src_bank[i], (H, W))
                    )  # [B,N,C]

                    bj = F.normalize(base[j], dim=-1)
                    mj = F.normalize(msg_ij, dim=-1)
                    cos = (bj * mj).sum(dim=-1, keepdim=True)         # [B,N,1]
                    agree = torch.sigmoid(2.0 * cos)                  # kappa=2.0
                    msgs[i][j] = msg_ij * agree

            delta = [torch.zeros_like(b) for b in base]
            K_src_stacked = torch.stack(K_src, dim=1)                 # [B,M,Dr]
            for j in range(M):
                Qj = self.router_q(base[j])                           # [B,N,Dr]
                logits = (Qj.unsqueeze(2) * K_src_stacked.unsqueeze(1)).sum(-1)  # [B,N,M]

                mask = torch.zeros_like(logits)
                mask[:, :, j] = float('-inf')
                logits = logits + mask
                
                logits = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
                invalid_row = ~torch.isfinite(logits).any(dim=-1, keepdim=True)
                logits = torch.where(invalid_row, torch.zeros_like(logits), logits)
                logits = logits - logits.max(dim=-1, keepdim=True)[0]


                tau = self.router_tau.clamp(0.3, 2.0)
                probs = torch.softmax(logits / tau, dim=-1)           # [B,N,M]

                if (self.router_topk is not None) and (self.router_topk < M):
                    topv, topi = probs.topk(self.router_topk, dim=-1)
                    probs_sparse = torch.zeros_like(probs).scatter_(-1, topi, topv)
                    probs = probs_sparse

                acc = 0.0
                for i in range(M):
                    if i == j: continue
                    pij = probs[:, :, i].unsqueeze(-1)                # [B,N,1]
                    acc = acc + pij * msgs[i][j]
                delta[j] = acc

            outs = [base[j] + delta[j] for j in range(M)]

        outs = [x + self.drop_path(self.mlp(self.norm2(x), H, W)) for x in outs]

        if self.enable_cross:
            base2   = [o for o in outs]
            W_modal2 = self.mod_gate(outs)

            src2_bank = []
            K2_src    = []
            for i in range(M):
                src2 = self.norm2(outs[i])
                src2 = self.chan_gate(src2, (H, W))
                src2 = src2 * W_modal2[:, i].view(-1,1,1)
                src2_bank.append(src2)
                g2_i = src2.transpose(1,2).mean(-1)
                K2_src.append(self.router_k(g2_i))
            K2_src_stacked = torch.stack(K2_src, dim=1)               # [B,M,Dr]

            msgs2 = [[None]*M for _ in range(M)]
            for i in range(M):
                for j in range(M):
                    if i == j: continue
                    msg_ij = self.drop_path(
                        self.film_mlp[self._k(i, j)](base2[j], src2_bank[i], (H, W))
                    )  # [B,N,C]

                    bj = F.normalize(base2[j], dim=-1)
                    mj = F.normalize(msg_ij,   dim=-1)
                    cos = (bj * mj).sum(dim=-1, keepdim=True)
                    agree = torch.sigmoid(2.0 * cos)
                    msgs2[i][j] = msg_ij * agree

            delta2 = [torch.zeros_like(b) for b in base2]
            for j in range(M):
                Qj2 = self.router_q(base2[j])                         # [B,N,Dr]
                logits2 = (Qj2.unsqueeze(2) * K2_src_stacked.unsqueeze(1)).sum(-1)  # [B,N,M]
                mask = torch.zeros_like(logits2)
                mask[:, :, j] = float('-inf')
                logits2 = logits2 + mask
                
                logits2 = torch.nan_to_num(logits2.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
                invalid_row2 = ~torch.isfinite(logits2).any(dim=-1, keepdim=True)
                logits2 = torch.where(invalid_row2, torch.zeros_like(logits2), logits2)
                logits2 = logits2 - logits2.max(dim=-1, keepdim=True)[0]
                
                tau2 = self.router_tau_film.clamp(0.6, 3.0)
                probs2 = torch.softmax(logits2 / tau2, dim=-1)

                acc2 = 0.0
                for i in range(M):
                    if i == j: continue
                    pij2 = probs2[:, :, i].unsqueeze(-1)
                    acc2 = acc2 + pij2 * msgs2[i][j]
                delta2[j] = acc2

            outs = [base2[j] + delta2[j] for j in range(M)]

        return outs



class CrossWeaver(nn.Module):

    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        self.modals = modals[1:] if len(modals)>1 else []
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.num_stages = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        pano_1 = [1,2,5,8]
        pano_2 = [8,4,2,1]
        for i in range(self.num_stages):
            cur = 0 if i == 0 else sum(depths[:i])
            patch_embed = PatchEmbed(3 if i==0 else embed_dims[i-1],
                                     embed_dims[i],
                                     7 if i==0 else 3,
                                     4 if i==0 else 2,
                                     7//2 if i==0 else 3//2)
            block = nn.ModuleList([
                MIB(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1)
                for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        feature_cross = SAF(self.channels, num_modals=self.num_modals + 1)
        setattr(self, f"feature_cross", feature_cross)


    def forward(self, x: list) -> list:
        x_in = [t.clone() for t in x]
        B = x[0].shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            for kk in range(len(x_in)):
                x_in[kk], H, W = patch_embed(x_in[kk])
            block = getattr(self, f"block{i + 1}")
            for blk_num in range(len(block)):
                x_in = block[blk_num](x_in, H, W)
            norm = getattr(self, f"norm{i + 1}")
            for kk in range(len(x_in)):
                x_in[kk] = norm(x_in[kk]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            feature_cross = getattr(self, f"feature_cross")
            x_fusion = feature_cross(x_in, layer_idx=i)
            outs.append(x_fusion)
        return outs

