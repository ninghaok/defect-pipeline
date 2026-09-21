from typing import Sequence
import math
import torch
import torch.nn as nn
from timm.layers.create_act import create_act_layer
from timm.layers.helpers import make_divisible


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=None, gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Linear(channels, rd_channels, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Linear(rd_channels, channels, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean(dim=1, keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax(dim=1, keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
    
    
# Attention Layer with learnable reference features
class MLP(nn.Module):
    """ 
    Orthogonal projection layer, we use the mlp in transformer as the projection layer.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True,
                 with_attn=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        
        self.with_attn = with_attn
        if with_attn:
            self.se = SEModule(hidden_features)

    def forward(self, x):
        out = self.fc1(x)
        if self.with_attn:
            out = self.se(out)
        out = self.act(out)
        out = self.fc2(out)
        
        return out


class ProjectEmbedding(nn.Module):
    def __init__(self, in_channels, d_model):
        super(ProjectEmbedding, self).__init__()
        
        self.project = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.project(x.permute(0, 2, 1)).transpose(1, 2)
        
        return x
    
    
class AttentionMLPLayer(nn.Module):
    def __init__(self, attention_config, with_attn=True):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(attention_config['hidden_size'], attention_config['num_attention_heads'], batch_first=True)
        self.mlp = MLP(attention_config['hidden_size'], 4 * attention_config['hidden_size'], attention_config['hidden_size'], with_attn=with_attn)

    def forward(self, x, x_ref):
        # x: (B, L, C)
        x_ori = x
        
        x1, _ = self.cross_attn(x, x_ref, x_ref)
        x1 = x_ori - x1
        
        x = x1
       
        out = self.mlp(x)
        out = out + x
        
        return out


class MultiLayerAttention(nn.Module):
    def __init__(self, attention_config, num_layers):
        super().__init__()
        
        self.device = attention_config['device']
        self.num_layers = num_layers
        layers = []
        for _ in range(num_layers):
            attention_layer = AttentionMLPLayer(attention_config)
            layers.append(attention_layer)
        self.layers = nn.ModuleList(layers)
        
        self.x_ref = nn.Parameter(torch.randn(1, 2048, attention_config['hidden_size']))
        self.ref_proj = ProjectEmbedding(attention_config['hidden_size'], attention_config['hidden_size'])
    
    def forward(self, x, keep_shape=True):
        x_ref = self.ref_proj(self.x_ref.repeat([x.shape[0], 1, 1]))
        
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        
        for layer in self.layers:
            x = layer(x, x_ref)
        if keep_shape:
            out = x.permute(0, 2, 1).reshape(b, c, h, w)
        
        return out
    
        
class MultiScaleAttentionProjector(nn.Module):
    def __init__(self,
                 channels: Sequence[int] = (1024, 1024, 1024, 1024),
                 device='cuda:0'):
        super().__init__()
        attention_config = {'hidden_size': channels[-1], 
                            'num_attention_heads': 16,
                            'max_image_size': (16, 16),
                            'device': device,
                            }
        self.l1_proj = MultiLayerAttention(attention_config, 1)
        self.l2_proj = MultiLayerAttention(attention_config, 1)
        self.l3_proj = MultiLayerAttention(attention_config, 1)
        self.l4_proj = MultiLayerAttention(attention_config, 1)
        
    def forward(self, x1, x2, x3, x4, keep_shape=True):
        out1 = self.l1_proj(x1, keep_shape=keep_shape)
        out2 = self.l2_proj(x2, keep_shape=keep_shape)
        out3 = self.l3_proj(x3, keep_shape=keep_shape)
        out4 = self.l4_proj(x4, keep_shape=keep_shape)
        
        return out1, out2, out3, out4


if __name__ == '__main__':
    model = MultiScaleAttentionProjector([1280, 1280, 1280, 1280], device='cuda:2').to('cuda:2')
    x = torch.randn(2, 1280, 16, 16).to('cuda:2')
    xs = [x, x, x, x]
    model(*xs)
    print("....")


