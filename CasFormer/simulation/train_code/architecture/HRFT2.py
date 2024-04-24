import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
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


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
      # type: (Tensor, float, float, float, float) -> Tensor
      return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
      fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
      if mode == 'fan_in':
            denom = fan_in
      elif mode == 'fan_out':
            denom = fan_out
      elif mode == 'fan_avg':
            denom = (fan_in + fan_out) / 2
      variance = scale / denom
      if distribution == "truncated_normal":
            trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
      elif distribution == "normal":
            tensor.normal_(std=math.sqrt(variance))
      elif distribution == "uniform":
            bound = math.sqrt(3 * variance)
            tensor.uniform_(-bound, bound)
      else:
            raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
      variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


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


def shift_back(inputs, step=2):  # input [bs,28,256,310]  output [bs, 28, 256, 256]
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
class Mspe(nn.Module):  # [bs, 28, 256, 256]
      def __init__(
              self,
              dim,
              dim_head=64,
              heads=8,
      ):
            super().__init__()
            self.num_heads = heads
            self.dim_head = dim_head
            self.to_q1 = nn.Linear(dim, dim_head * heads, bias=False)
            self.to_k1 = nn.Linear(dim, dim_head * heads, bias=False)
            self.to_v1 = nn.Linear(dim, dim_head * heads, bias=False)
            self.rescale = torch.ones([heads, 1, 1],requires_grad=True)
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
            # x = x.permute(0, 2, 3, 1)
            b, h, w, c = x.shape
            x = x.reshape(b, h * w, c)
            q1_inp = self.to_q1(x)
            k1_inp = self.to_k1(x)
            v1_inp = self.to_v1(x)
            mask_attn = self.MaskGuidedMechanism(mask)
            # mask_attn = mask_attn.permute(0, 2, 3, 1)
            if b != 0:
                  mask_attn = (mask_attn[0, :, :, :]).expand([b, h, w, c])
            q1, k1, v1, mask_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                        (q1_inp, k1_inp, v1_inp, mask_attn.flatten(1, 2)))
            v1 = v1 * mask_attn
            # q: b,heads,hw,c
            q1 = q1.transpose(-2, -1)
            k1 = k1.transpose(-2, -1)
            v1 = v1.transpose(-2, -1)
            q1 = F.normalize(q1, dim=-1, p=2)
            k1 = F.normalize(k1, dim=-1, p=2)
            attn = (k1 @ q1.transpose(-2, -1))  # A = K^T*Q
            attn = attn * self.rescale.cuda()
            attn = attn.softmax(dim=-1)
            x = attn @ v1  # b,heads,d,hw
            x = x.permute(0, 3, 1, 2)  # Transpose
            x = x.reshape(b, h * w, self.num_heads * self.dim_head)
            out_c = self.proj(x).view(b, h, w, c)
            out_p = self.band_emb(v1_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            Xspe = out_c + out_p

            return Xspe


# ----------------------------------------
#           光谱特征提取
# ----------------------------------------
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

            return LR_HSI


# --------------------------------------------------------------------------------
#           Spatial Feature Extractor (SpaFE)——————RGB+RGB(DW,UP)
# --------------------------------------------------------------------------------
class SpaFE(nn.Module):
      def __init__(self, n_fts=28):
            super(SpaFE, self).__init__()
            # Define number of input channels
            self.n_fts = n_fts

            lv1_c = int(n_fts)
            lv2_c = int(n_fts * 2)
            lv4_c = int(n_fts * 4)

            # input:C
            # First level convolutions
            self.conv_256_1 = nn.Conv2d(in_channels=3, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_1 = nn.BatchNorm2d(lv1_c)

            # Hfusion28
            self.conv_28 = nn.Conv2d(in_channels=28, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_28 = nn.BatchNorm2d(lv1_c)

            self.conv_256_2 = nn.Conv2d(in_channels=lv1_c, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_2 = nn.BatchNorm2d(lv1_c)

            # input:2C
            # Second level convolutions
            self.conv_128_1 = nn.Conv2d(in_channels=lv1_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_1 = nn.BatchNorm2d(lv2_c)
            self.conv_128_2 = nn.Conv2d(in_channels=lv2_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_2 = nn.BatchNorm2d(lv2_c)

            # input:4C
            # Third level convolutions
            self.conv_64_1 = nn.Conv2d(in_channels=lv2_c, out_channels=lv4_c, kernel_size=3, padding=1)
            self.bn_64_1 = nn.BatchNorm2d(lv4_c)
            self.conv_64_2 = nn.Conv2d(in_channels=lv4_c, out_channels=lv4_c, kernel_size=3, padding=1)
            self.bn_64_2 = nn.BatchNorm2d(lv4_c)

            # Max pooling
            self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # LeakyReLU
            self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)

      def forward(self, x_rgb):
            # RGB = RGB.permute(1, 3, 0, 2)
            # x_rgb = torch.from_numpy(np.array(x_rgb))
            b, c, h, w = x_rgb.shape
            # with torch.no_grad():
            RGB_D = F.interpolate(x_rgb, size=h // 4, mode='bilinear')
            RGB_DU = F.interpolate(RGB_D, size=h, mode='bilinear')

            # 分辨率无变化的RGB
            # First level outputs            # input:C
            if c == 3:
                  RGB_out = self.LeakyReLU(self.bn_256_1(self.conv_256_1(x_rgb)))
            else:
                  RGB_out = self.LeakyReLU(self.bn_28(self.conv_28(x_rgb)))

            RGB_out = self.LeakyReLU(self.bn_256_2(self.conv_256_2(RGB_out)))
            RGB_Out1 = RGB_out
            # Second level outputs            # input:2C
            RGB_out = self.MaxPool2x2(self.LeakyReLU(RGB_Out1))
            RGB_out = self.LeakyReLU(self.bn_128_1(self.conv_128_1(RGB_out)))
            RGB_out = self.LeakyReLU(self.bn_128_2(self.conv_128_2(RGB_out)))
            RGB_Out2 = RGB_out
            # Third level outputs            # input:3C
            RGB_out = self.MaxPool2x2(self.LeakyReLU(RGB_Out2))
            RGB_out = self.LeakyReLU(self.bn_64_1(self.conv_64_1(RGB_out)))
            RGB_out = self.LeakyReLU(self.bn_64_2(self.conv_64_2(RGB_out)))
            RGB_Out3 = RGB_out

            # 4倍下采样后的RGB
            # First level outputs            # input:C
            if c == 3:
                  RGB_DU_out = self.LeakyReLU(self.bn_256_1(self.conv_256_1(RGB_DU)))
            else:
                  RGB_DU_out = self.LeakyReLU(self.bn_28(self.conv_28(RGB_DU)))

            RGB_DU_out = self.LeakyReLU(self.bn_256_2(self.conv_256_2(RGB_DU_out)))
            RGB_DU_Out1 = RGB_DU_out
            # Second level outputs            # input:2C
            RGB_DU_out = self.MaxPool2x2(self.LeakyReLU(RGB_DU_Out1))
            RGB_DU_out = self.LeakyReLU(self.bn_128_1(self.conv_128_1(RGB_DU_out)))
            RGB_DU_out = self.LeakyReLU(self.bn_128_2(self.conv_128_2(RGB_DU_out)))
            RGB_DU_Out2 = RGB_DU_out
            # Third level outputs            # input:4C
            RGB_DU_out = self.MaxPool2x2(self.LeakyReLU(RGB_DU_Out2))
            RGB_DU_out = self.LeakyReLU(self.bn_64_1(self.conv_64_1(RGB_DU_out)))
            RGB_DU_out = self.LeakyReLU(self.bn_64_2(self.conv_64_2(RGB_DU_out)))
            RGB_DU_Out3 = RGB_DU_out

            return [[RGB_Out1, RGB_DU_Out1], [RGB_Out2, RGB_DU_Out2], [RGB_Out3, RGB_DU_Out3]]


# --------------------------------------------------------------------------------
#           Spatial-Spetral Cross-Attention
# --------------------------------------------------------------------------------
class MulCorssAttention(nn.Module):
      def __init__(
              self,
              dim,
              dim_head=64,
              heads=8
      ):
            super().__init__()
            self.num_heads = heads
            self.dim_head = dim_head
            self.to_q = nn.Linear(dim, dim_head * heads, bias=False)  # nn.Linear(in_feature,out_feature,bias)
            self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
            self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
            self.rescale = nn.Parameter(torch.ones([heads, 1, 1],requires_grad=True)).cuda()  # torch.nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。即在定义网络时这个tensor就是一个可以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
            self.proj = nn.Linear(dim_head * heads, dim, bias=True)
            self.band_emb = nn.Sequential(
                  nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
                  GELU(),
                  nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            )
            #       nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            #       nn.BatchNorm3d(dim)
            # )
            self.dim = dim

      def forward(self, V_in, K_in, Q_in):  # x_in[ 256 256  28]; mask[# 256 256  28] V_in,K_in,Q_in:# 256 256  28
            Q_in = Q_in.permute(0, 2, 3, 1)
            K_in = K_in.permute(0, 2, 3, 1)
            V_in = V_in.permute(0, 2, 3, 1)
            """
            x_in: [b,h,w,c]
            mask: [1,h,w,c]
            return out: [b,h,w,c]
            """
            b, h, w, c = V_in.shape
            V_in = V_in.reshape(b, h * w, c)
            Q_in = Q_in.reshape(b, h * w, c)
            K_in = K_in.reshape(b, h * w, c)
            # V = V_in.reshape(b, h * w, c)
            q_inp_2 = self.to_q(Q_in)
            k_inp_2 = self.to_k(K_in)
            v_inp_2 = self.to_v(V_in)

            q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                             (q_inp_2, k_inp_2, v_inp_2))
            # v1 = v1 * mask_attn
            # q: b,heads,hw,c
            # q2 = q2.transpose(-2, -1)
            # k2 = k2.transpose(-2, -1)
            # v2 = v2.transpose(-2, -1)
            q2 = F.normalize(q2, dim=-1, p=2)
            k2 = F.normalize(k2, dim=-1, p=2)
            attn_2 = (k2 @ q2.transpose(-2, -1))  # A = K^T*Q
            # attn_2 = attn_2 * self.rescale
            attn_2 = attn_2.softmax(dim=-1)
            x_2 = attn_2 @ v2  # b,heads,d,hw
            x_2 = x_2.permute(0, 3, 1, 2)  # Transpose
            x_2 = x_2.reshape(b, h * w, self.num_heads * self.dim_head)
            out_c_2 = self.proj(x_2).view(b, h, w, c)

            out_p_2 = self.band_emb(v_inp_2.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)#
            HR_HSI = out_c_2 + out_p_2

            return HR_HSI


# -------------------------------------------------------
#           LR-HSI and RGB Fusion
# ------------------------------------------------------
class HRFusion(nn.Module):
      def __init__(
              self, dim, dim_head, heads=8):
            super(HRFusion, self).__init__()
            # self.SpaFE = None
            self.dim = dim
            self.num_heads = heads
            self.dim_head = dim_head

            self.Mspe = Mspe(dim, dim_head, heads)
            self.SpaFE = SpaFE()

            self.SpeFE = SpeFE(dim)
            ###########################################################################################
            ### Multi-Head Attention ###
            self.MA2 = MulCorssAttention(dim=self.dim, dim_head=self.dim_head, heads=heads)

      # rgblist[i] = rgb_listi->[a,b] rgb_listi[i]
      def forward(self, x, mask, rgb):
            """
                  x_rgb: [b,h,w,c]
                  LR_HSI: [b,h,w,c]
                  return out: [b,h,w,c]
                  """
            ####################################### Mspe 结果进行上采样 ##################################
            if mask is not None:
                  LR_HSI = self.Mspe(x, mask)
            else:
                  LR_HSI = x
            LR_HSI = LR_HSI.permute(0, 3, 1, 2)
            b, c, h, w = LR_HSI.shape
            LR_HSI_Up = F.interpolate(LR_HSI, size=(h, w), mode='bilinear')
            # LR_HSI_Up = LR_HSI_Up.permute(0, 3, 2, 1)
            #############################################################################################
            V_2 = rgb[0]
            K_2 = rgb[1]
            Q_2 = self.SpeFE(LR_HSI_Up)

            T = self.MA2(V_2, K_2, Q_2)  # MultiHeadAttention## 256 256  28
            # 256 256  28

            return T


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
              self, dim, dim_head, heads=8, num_blocks=1
      ):
            super().__init__()
            self.blocks = nn.ModuleList([])
            for _ in range(num_blocks):
                  self.blocks.append(nn.ModuleList([
                        Mspe(dim=dim, dim_head=dim_head, heads=heads),
                        HRFusion(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim))
                  ]))

      def forward(self, x, mask, x_rgb):
            """
            x: [b,c,h,w]
            return out: [b,c,h,w]
            """
            x = x.permute(0, 2, 3, 1)
            # mask = mask.permute(0, 2, 3, 1)
            for (attn1, attn2, ff) in self.blocks:
                  x1 = attn1(x, mask) + x  # .permute(0, 2, 3, 1)
                  x2 = attn2(x1, mask=None, rgb=x_rgb) + x1
                  x3 = ff(x2) + x2
            out = x3.permute(0, 3, 1, 2)
            return out


class HRFT(nn.Module):
      def __init__(self, dim=28, stage=3, num_blocks=None):
            super(HRFT, self).__init__()

            if num_blocks is None:
                  num_blocks = [2, 2, 2]

            self.dim = dim
            self.stage = stage

            # Input projection
            self.embedding = nn.Conv2d(28, self.dim, 3, 1, 1, bias=False)

            # Encoder
            self.encoder_layers = nn.ModuleList([])
            dim_stage = dim
            for i in range(stage):
                  self.encoder_layers.append(nn.ModuleList([
                        CascadeTransformer(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim,
                                           heads=dim_stage // dim),
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
                                           heads=(dim_stage // 2) // dim)]))
                  dim_stage //= 2

            # Output projection
            self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

            #### activation function
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

            self.SpaFE = SpaFE()

      def forward(self, x, x_rgb, mask=None):
            """
            x: [b,c,h,w]
            return out:[b,c,h,w]
            """
            if mask == None:
                  mask = torch.zeros((1, 28, 128, 182)).cuda()

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

            # Decoder  RGB_list[1-i]
            for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
                  fea = FeaUpSample(fea)
                  fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
                  mask = masks[self.stage - 1 - i]
                  fea = LeWinBlcok(fea, mask, rgb_list[1 - i])

            # Mapping
            out = self.mapping(fea) + x

            return out
