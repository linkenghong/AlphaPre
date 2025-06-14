import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class AmpTimeCell(nn.Module):
    def __init__(self, t_in, t_out, size_factor=1):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out*size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out*size_factor), t_out),
        )
        self.scale = 0.02

        self.w1 = nn.Parameter((self.scale * torch.randn(2, t_in, t_out*size_factor)))
        self.b1 = nn.Parameter((self.scale * torch.randn(2, 1, 1, 1, t_out*size_factor)))
        self.w2 = nn.Parameter((self.scale * torch.randn(2, t_out*size_factor, t_out)))
        self.b2 = nn.Parameter((self.scale * torch.randn(2, 1, 1, 1, t_out)))
    
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        bias = self.tmlp(x)
        xf = torch.fft.rfft2(x, dim=[2,3], norm="ortho")
        x1_real = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[0]) - \
                  torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[1]) + \
                  self.b1[0]
        x1_imag = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[1]) + \
                  torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[0]) + \
                  self.b1[1]
        x1_real, x1_imag = F.relu(x1_real), F.relu(x1_imag)
        
        x2_real = torch.einsum('bchwt,to->bchwo', x1_real, self.w2[0]) - \
                  torch.einsum('bchwt,to->bchwo', x1_imag, self.w2[1]) + \
                  self.b2[0]
        x2_imag = torch.einsum('bchwt,to->bchwo', x1_real, self.w2[1]) + \
                  torch.einsum('bchwt,to->bchwo', x1_imag, self.w2[0]) + \
                  self.b2[1]

        x2 = torch.view_as_complex(torch.stack([x2_real, x2_imag], dim=-1))
        x = torch.fft.irfft2(x2, dim=[2,3], norm="ortho")
        x = x + bias
        return x.permute(0,4,1,2,3)


class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, size_factor=1.0,
        ):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out*size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out*size_factor), t_out),
        )
        self.amptime =  AmpTimeCell(t_in, t_out)
        self.conv = nn.Sequential(nn.Conv2d(dim*t_out, dim*t_out, kernel_size=3,padding=1),
                                  nn.GroupNorm(4, dim*t_out),
                                  nn.SiLU(),
                                  nn.Conv2d(dim*t_out, dim*t_out, kernel_size=3,padding=1),)

    def forward(self, x):
        residual = self.tmlp(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        x = self.amptime(x)
        x = x + residual

        residual = x
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        x = x + residual
        return x


class AmpliNet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=3, mlp_ratio=2):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
        self.tmlp = nn.Sequential(
            nn.Linear(pre_seq_length, int(aft_seq_length*mlp_ratio)),
            nn.SELU(True),
            nn.Linear(int(aft_seq_length*mlp_ratio), aft_seq_length),
        )
        self.convin = nn.Sequential(ResnetBlock(dim, hidden_dim),
                                    ResnetBlock(hidden_dim, hidden_dim),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.amplist = nn.ModuleList([
            AmpCell(pre_seq_length if i==0 else aft_seq_length, aft_seq_length, hidden_dim) for i in range(n_layers)
        ])
        self.convout = nn.Sequential(ResnetBlock(hidden_dim, hidden_dim),
                                     ResnetBlock(hidden_dim, hidden_dim),
                                     nn.Conv2d(hidden_dim, dim, kernel_size=1))

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)
        x_ = x.permute(0,2,3,4,1)
        xr = self.tmlp(x_)
        xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for ampcell in self.amplist:
            x = ampcell(x)
        x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convout(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return x


class PhaseNet(nn.Module):
    def __init__(self, input_shape, pre_seq_length, aft_seq_length, input_dim, hidden_dim, 
                 n_layers, kernel_size, bias=1):
        super().__init__()
        h, w = input_shape
        input_shape = (h, w//2+1)
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.pha_conv0 = nn.Conv2d(2+input_dim*pre_seq_length, input_dim*aft_seq_length, 1)
        self.phase_0 = nn.Sequential(ResnetBlock(2+input_dim*pre_seq_length, hidden_dim, kernel_size=1),
                                     ResnetBlock(hidden_dim, hidden_dim, kernel_size=1),
                                     nn.Conv2d(hidden_dim, input_dim*aft_seq_length, kernel_size=1))
        self.phase_1 = nn.Sequential(ResnetBlock(2+input_dim*pre_seq_length, hidden_dim, kernel_size=1),
                                     ResnetBlock(hidden_dim, hidden_dim, kernel_size=1),
                                     nn.Conv2d(hidden_dim, input_dim*aft_seq_length, kernel_size=1))
        self.phase_2 = nn.Sequential(ResnetBlock(2+input_dim*pre_seq_length, hidden_dim, kernel_size=3,padding_mode='circular'),
                                     ResnetBlock(hidden_dim, hidden_dim, kernel_size=3,padding_mode='circular'),
                                     nn.Conv2d(hidden_dim, input_dim*aft_seq_length, kernel_size=1))
        
        self.pha_conv1 = nn.Conv2d(4*input_dim*aft_seq_length, input_dim*aft_seq_length, 1)
        u = torch.fft.fftfreq(h)
        v = torch.fft.rfftfreq(w)
        u, v = torch.meshgrid(u, v)
        uv = torch.stack((u,v),dim=0)
        self.register_buffer('uv', uv)

    def forward(self, x): # x:[b,t,c,h,w]
        B,T,C,H,W = x.shape
        x_fft = torch.fft.rfft2(x)
        x_amps, x_phas = torch.abs(x_fft), torch.angle(x_fft) 
        x_phas = self.pha_norm(x_phas)
        x_phas_ = rearrange(x_phas, 'b t c h w -> b (t c) h w')
        x_puv = torch.cat((x_phas_, self.uv.repeat(B,1,1,1)), dim=1)
        x_phast = self.pha_conv0(x_puv)
        x_phas0 = x_phast + self.phase_0(x_puv)
        x_phas1 = x_phast * self.phase_1(x_puv)
        x_phas2 = x_phast * self.phase_2(x_puv)
        x_phas_t = torch.cat((x_phast, x_phas0, x_phas1, x_phas2), dim=1)
        x_phas_t = self.pha_conv1(x_phas_t)
        x_phas_t = rearrange(x_phas_t, 'b (t c) h w -> b t c h w', t=self.aft_seq_length)
        x_phas_t = x_phas[:,-1:] + x_phas_t
        x_phas_t = self.pha_unnorm(x_phas_t)
        xt_fft = x_amps[:,-1:] * torch.exp(torch.tensor(1j) * x_phas_t)
        xt = torch.fft.irfft2(xt_fft)
        return xt, x_phas_t, x_amps

    def pha_norm(self, x):
        return x / torch.pi

    def pha_unnorm(self, x):
        return x * torch.pi
    
class AlphaMixer(nn.Module):
    def __init__(self, input_shape, spec_num, input_dim, hidden_dim, aft_seq_length) -> None:
        super().__init__()
        h, w = input_shape
        self.aft_seq_length = aft_seq_length
        self.spec_num = spec_num
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[...,:spec_num,:spec_num] = 1.
        spec_mask[...,-spec_num:,:spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)
        self.out_mixer = nn.Sequential(ResnetBlock(3*input_dim, hidden_dim),
                                       ResnetBlock(hidden_dim, hidden_dim),
                                       nn.Conv2d(hidden_dim, input_dim, kernel_size=1))

    def forward(self, xas, xps, phas):
        xas_fft = torch.fft.rfft2(xas)
        amps = torch.abs(xas_fft)
        alpha_fft = amps * self.spec_mask * torch.exp(torch.tensor(1j) * phas)
        alpha = torch.fft.irfft2(alpha_fft)
        xap = torch.cat([xas, xps, alpha],dim=2)
        xap = rearrange(xap, 'b t c h w -> (b t) c h w')
        xt = self.out_mixer(xap)
        xt = rearrange(xt, '(b t) c h w -> b t c h w', t=self.aft_seq_length)
        return xt


class AlphaPre(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, input_shape, input_dim, 
                 hidden_dim, n_layers, spec_num=20, kernel_size=1, bias=1, 
                 pha_weight=0.01, anet_weight=0.1, amp_weight=0.01, aweight_stop_steps=10000):
        super(AlphaPre, self).__init__()

        self.amplinet = AmpliNet(pre_seq_length, aft_seq_length, input_dim, hidden_dim)
        self.phasenet = PhaseNet(input_shape, pre_seq_length, aft_seq_length, input_dim, hidden_dim, n_layers, kernel_size, bias)
        self.alphamixer = AlphaMixer(input_shape, spec_num, input_dim, hidden_dim, aft_seq_length)
        self.input_shape, self.input_dim = input_shape, input_dim
        self.hidden_dim = hidden_dim
        self.spec_num = spec_num
        self.pha_weight = pha_weight
        self.anet_weight = anet_weight
        self.amp_weight = amp_weight
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.criterion = nn.MSELoss()
        self.itr = 0
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate =  self.amp_weight/self.aweight_stop_steps

        h, w = input_shape
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[...,:spec_num,:spec_num] = 1.
        spec_mask[...,-spec_num:,:spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)

    def forward(self, x, y, cmp_fft_loss=False): # x:[b,t,c,h,w]
        self.itr += 1
        xas = self.amplinet(x)
        xas = torch.sigmoid(xas)
        xps, x_phas_t, x_amps = self.phasenet(x)
        xt = self.alphamixer(xas, xps, x_phas_t)

        return xt, xps, xas, x_phas_t, x_amps

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        B = frames_in.shape[0]
        xt, xps, xas, x_phas_t, x_amps = self(frames_in, frames_gt, compute_loss)
        pred = xt
        if compute_loss:
            if self.itr < self.aweight_stop_steps:
                self.amp_weight -= self.sampling_changing_rate
            else:
                self.amp_weight  = 0.
            loss = 0.
            loss += self.criterion(pred, frames_gt)
            frames_fft = torch.fft.rfft2(frames_gt)
            frames_pha = torch.angle(frames_fft)
            frames_abs = torch.abs(frames_fft)
            pha_loss = (1 - torch.cos(frames_pha * self.spec_mask - x_phas_t * self.spec_mask)).sum() / (self.spec_mask.sum()*B*self.aft_seq_length*self.input_dim)
            loss += self.pha_weight*pha_loss
            xas_fft = torch.fft.rfft2(xas)
            xas_abs = torch.abs(xas_fft)
            amp_loss = self.criterion(xas_abs, frames_abs)
            loss += self.amp_weight*amp_loss
            anet_loss = self.criterion(xas, frames_gt)
            loss += self.anet_weight*anet_loss
            loss = {'total_loss': loss, 'phase_loss': self.pha_weight*pha_loss,
                    'ampli_loss': self.amp_weight*amp_loss, 'anet_loss': self.anet_weight*anet_loss}
            return pred, loss
        else:
            return pred, None


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', groupnorm=True):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding = kernel_size//2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros'): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )


def get_model(
    img_channels=1,
    dim = 64,
    T_in = 5, 
    T_out = 20,
    input_shape = (128,128),
    n_layers = 3,
    spec_num = 20,
    pha_weight=0.01, 
    anet_weight=0.1,
    amp_weight=0.01,
    aweight_stop_steps=10000,
    **kwargs
):
    model = AlphaPre(pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape, input_dim=img_channels, 
                     hidden_dim=dim, n_layers=n_layers, spec_num=spec_num,
                     pha_weight=pha_weight, anet_weight=anet_weight, amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
                     )
    
    return model