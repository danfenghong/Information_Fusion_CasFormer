import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def shift_back(inputs, step=2): 
    [bs, nC, row, col] = inputs.shape
    if row == col:
        return inputs
    else:
        down_sample = 256 // row
        step = float(step) / float(down_sample * down_sample)
        out_col = row
        for i in range(nC):
            inputs[:, i, :, :out_col] = \
                inputs[:, i, :, int(step * i):int(step * i) + out_col]
        return inputs[:, :, :, :out_col]


# ----------------------------------------
#       Mask-Guided Attention
# ----------------------------------------
class MaskGuidedMechanism(nn.Module):
    def __init__(
            self, n_feat):
        super(MaskGuidedMechanism, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_shift):
        # x: b,c,h,w
        [bs, nC, row, col] = mask_shift.shape
        mask_shift = self.conv1(mask_shift)
        attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
        res = mask_shift * attn_map
        mask_shift = res + mask_shift
        mask_emb = shift_back(mask_shift)
        mask_emb = mask_emb.permute(0, 2, 3, 1)
        return mask_emb


# ----------------------------------------
#       Mspe Transformer Block
# ----------------------------------------
class Mspe(nn.Module): 
    def __init__(
            self,
            dim, heads,
            dim_head
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q1 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k1 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v1 = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1)).cuda()
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.band_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.MaskGuidedMechanism = MaskGuidedMechanism(dim)

    def forward(self, x, mask):
        """
        x_in: [b,h,w,c]
        mask: [1,c,h,w]
        return out: [b,h,w,c]
        """

        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        q1_inp = self.to_q1(x)
        k1_inp = self.to_k1(x)
        v1_inp = self.to_v1(x)
        mask_attn = self.MaskGuidedMechanism(mask)
      
        if b != 0:
            mask_attn = (mask_attn[0, :, :, :]).expand([b, h, w, c])
        q1, k1, v1, mask_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                    (q1_inp, k1_inp, v1_inp, mask_attn.flatten(1, 2)))
        v1 = v1 * mask_attn
      
        q1 = q1.transpose(-2, -1)
        k1 = k1.transpose(-2, -1)
        v1 = v1.transpose(-2, -1)
        q1 = F.normalize(q1, dim=-1, p=2)
        k1 = F.normalize(k1, dim=-1, p=2)
        attn = (k1 @ q1.transpose(-2, -1))  
        attn = attn * self.rescale.cuda()
        attn = attn.softmax(dim=-1)
        x = attn @ v1  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.band_emb(v1_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        Xspe = out_c + out_p

        return Xspe

# --------------------------------------------------------------------------------
#           Spatial Feature Extractor (SpaFE)——————RGB+RGB(DW,UP)
# --------------------------------------------------------------------------------
class SpaFE(nn.Module):
    def __init__(self, n_fts=31):
        super(SpaFE, self).__init__()
        # Define number of input channels
        self.n_fts = n_fts

        lv1_c = int(n_fts)
        lv2_c = int(n_fts * 2)
        lv4_c = int(n_fts * 4)
        # 3 256 256 ->28 256 256
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=lv1_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(lv1_c),
                                     nn.LeakyReLU(negative_slope=0.0),
                                     )
        # 3 256 256 -> 56 128 128
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=lv2_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(lv2_c),
                                     nn.LeakyReLU(negative_slope=0.0),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     )
        # 3 256 256 -> 112 64 64
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=lv2_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(lv2_c),
                                     nn.LeakyReLU(negative_slope=0.0),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(in_channels=lv2_c, out_channels=lv4_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(lv4_c),
                                     nn.LeakyReLU(negative_slope=0.0),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )

    def forward(self, x_rgb):
        x1 = self.layer_1(x_rgb)
        x2 = self.layer_2(x_rgb)
        x3 = self.layer_3(x_rgb)

        return [x1, x2, x3]

class SpeFE(nn.Module):
    def __init__(self, dim):
        super(SpeFE, self).__init__()
        self.dim = dim
        self.conv_11 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        # self.ln_11 = nn.LayerNorm()
        self.LeakyReLU = nn.LeakyReLU(dim)
        self.conv_12 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        # self.ln_12 = nn.LayerNorm()

    def forward(self, LR_HSI_Up):
        ln_11 = nn.LayerNorm(LR_HSI_Up.shape).cuda()
        ln_12 = nn.LayerNorm(LR_HSI_Up.shape).cuda()
        out1_1 = self.LeakyReLU(ln_11(self.conv_11(LR_HSI_Up)))
        # out1_1 = self.LeakyReLU(self.conv_11(LR_HSI_Up))
        out1_2 = self.LeakyReLU(ln_12(self.conv_12(out1_1)))
        # out1_2 = self.LeakyReLU(self.conv_12(out1_1))
        LR_HSI = out1_2 + LR_HSI_Up

        return LR_HSI_Up


# --------------------------------------------------------------------------------
#          Spatial-Spetral Cross-Attention
# -------------------------------------------------------------------------------
class MulCorssAttention(nn.Module):
    def __init__(self, dim, heads, dim_head=64, token_height=16, token_width=16, q_bias=False, k_bias=False,
                 v_bias=False, proj_drop=0.):
        super().__init__()
        self.heads = 8
        self.scale = dim_head ** -0.5
        self.token_height = token_height
        self.token_width = token_width
        self.dim = dim
        self.to_q2 = nn.Linear(token_height * token_width * 2, token_height * token_width * 2, q_bias)
        self.to_k2 = nn.Linear(token_height * token_width * 2, token_height * token_width * 2, k_bias)
        self.to_v2 = nn.Linear(token_height * token_width * 2, token_height * token_width * 2, v_bias)
        self.to_out = nn.Sequential(
            nn.Linear(token_height * token_width * 2, token_height * token_width * dim),
            nn.Dropout(proj_drop),
        )
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, V_2_in, K_2_in, Q_2_in):
        b, c, h, w = Q_2_in.shape
        (image_height, image_width) = (h, w)
        assert image_height % self.token_height == 0 and image_width % self.token_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_num_x = image_height // self.token_height
        patch_num_y = image_width // self.token_width
        num_patches = (image_height // self.token_height) * (image_width // self.token_width)
        token_dim = self.token_height * self.token_width * c
        pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.token_height * self.token_width * 2)).cuda()
        ########################################################################################
        to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.token_height, p2=self.token_width),
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, self.token_height * self.token_width * 2),
            nn.LayerNorm(self.token_height * self.token_width * 2)).cuda()
        ########################################################################################

        Q_2_in = to_patch_embedding(Q_2_in)
        K_2_in = to_patch_embedding(K_2_in)
        V_2_in = to_patch_embedding(V_2_in)

        b, n, _ = Q_2_in.shape
        Q = self.proj_drop(Q_2_in)
        Q += pos_embedding[:, :n]
        K = self.proj_drop(K_2_in)
        K += pos_embedding[:, :n]
        V = self.proj_drop(V_2_in)
        V += pos_embedding[:, :n]
        Q = self.to_q2(Q)
        K = self.to_k2(K)
        V = self.to_v2(V)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (Q, K, V))
        dots = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.proj_drop(attn)
        out = torch.matmul(attn, v2)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) (p1 p2 c)->b c (h p1) (w p2)', p1=self.token_height, p2=self.token_width,
                        h=patch_num_x, w=patch_num_y)
        return out


# -------------------------------------------------------
#           LR-HSI and RGB Fusion
# ------------------------------------------------------
class HRFusion(nn.Module):
    def __init__(self, *, token_size, dim, heads, pool='cls', dim_head=64, emb_dropout=0.):
        super(HRFusion, self).__init__()

        (self.token_height, self.token_width) = token_size
        self.token_size = token_size
        self.dim = dim

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.dropout = nn.Dropout(emb_dropout)
        self.to_latent = nn.Identity()

        self.MulCorssAttention = MulCorssAttention(dim, heads, dim_head, proj_drop=0.)

        self.Mspe = Mspe(dim, dim_head, heads)
        self.SpaFE = SpaFE()
        self.SpaFE = SpaFE(dim)
        self.conv_v = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=3, padding=1)
        self.BN=nn.BatchNorm2d(dim)

    def forward(self, x, mask, rgb):
        if mask is not None:
            LR_HSI = self.Mspe(x, mask)
        else:
            LR_HSI = x

        LR_HSI = LR_HSI.permute(0, 3, 1, 2)
        b, c, h, w = LR_HSI.shape
        V = LR_HSI
        K = torch.concat((LR_HSI, rgb), dim=1)
        K = self.BN(self.conv_v(K))
        Q = rgb

        atten = self.MulCorssAttention(V, K, Q)
        x = atten + K  

        return x.permute(0, 2, 3, 1)


# -------------------------------------------------------
#           FeedForward
# ------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# -------------------------------------------------------
#           Cascade  Transformer
# ------------------------------------------------------
class CascadeTransformer(nn.Module):
    def __init__(
            self, dim, dim_head, heads, num_blocks=1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                Mspe(dim=dim, dim_head=dim_head, heads=heads),
                HRFusion(dim=dim, dim_head=dim_head, heads=heads, token_size=(16, 16)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, mask, x_rgb):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)

        for (attn1, attn2, ff) in self.blocks:
            x1 = attn1(x, mask) + x  # .permute(0, 2, 3, 1)
            x2 = attn2(x1, mask=None, rgb=x_rgb) + x1
            x3 = ff(x2) + x2
        out = x3.permute(0, 3, 1, 2)
        return out


class HRFT(nn.Module):
    def __init__(self, dim=31, stage=3, num_blocks=None):
        super(HRFT, self).__init__()

        if num_blocks is None:
            num_blocks = [1, 1, 1]

        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(31, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                CascadeTransformer(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim,
                                   heads=dim_stage // dim),  # dim_stage // dim
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2
        # dim_stage

        # Bottleneck
        self.bottleneck = CascadeTransformer(dim=dim_stage, dim_head=dim, heads=dim_stage // dim,
                                             num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0,
                                   output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                CascadeTransformer(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                                   heads=(dim_stage // 2) // dim)]))  # (dim_stage // 2) // dim
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 31, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)  #LeakyReLU

        self.SpaFE = SpaFE(n_fts=31)

    def forward(self, x, x_rgb, mask=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if mask == None:
            mask = torch.zeros((1, 31, 256, 316)).cuda()

        rgb_list = self.SpaFE(x_rgb)

        # Embedding
        fea = self.lrelu(self.embedding(x))
        # Encoder
        fea_encoder = []
        masks = []
        for i, (CascadeTransformer, FeaDownSample, MaskDownSample) in enumerate(self.encoder_layers):
            fea = CascadeTransformer(fea, mask, rgb_list[i])
            masks.append(mask)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            mask = MaskDownSample(mask)

        # Bottleneck
        fea = self.bottleneck(fea, mask, rgb_list[2])

        # Decoder  
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            mask = masks[self.stage - 1 - i]
            fea = LeWinBlcok(fea, mask, rgb_list[1 - i])

        # Mapping
        out = self.mapping(fea) + x

        return out
