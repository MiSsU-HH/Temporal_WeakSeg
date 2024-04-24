import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.TSViT.module import Attention, PreNorm, FeedForward
import numpy as np


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., return_att=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.return_att = return_att
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, return_att=return_att)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        if self.return_att:
            attn_weights = []
            for attn, ff in self.layers:
                x, weights_i = attn(x)
                attn_weights.append(weights_i)
                x = ff(x) + x
            return self.norm(x), attn_weights
        else:
            for attn, ff in self.layers:
                x = attn(x)
                x = ff(x) + x
            return self.norm(x)


class TSViTcls(nn.Module):
    """
    Temporal-Spatial ViT for object classification (used in main results, section 4.3)
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        self.return_att = model_config["return_att"]
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        # self.depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(365, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.temporal_depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout, return_att=False)
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dim))
        print('space pos embedding: ', self.space_pos_embedding.shape)
        print('space token: ', self.space_token.shape)
        self.space_transformer = Transformer(self.dim, self.spatial_depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout, self.return_att)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 1))
        self.patch_head = nn.Conv2d(self.dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)  # 4 60 24 24 11
        B, T, C, H, W = x.shape
        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]   # 4 60 10 24 24
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=365).to(torch.float32)
        xt = xt.reshape(-1, 365)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x = self.to_patch_embedding(x)   # 256(24/3 * 24/3 * 4) 60 128
        x = x.reshape(B, -1, T, self.dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)   # 256 60 128
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)
        x = torch.cat((cls_temporal_tokens, x), dim=1)   # 256(24/3 * 24/3 * 4) 78 128
        x = self.temporal_transformer(x)
        x = x[:, :self.num_classes]  #256 18 128
        x = x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim).permute(0, 2, 1, 3).reshape(B*self.num_classes, self.num_patches_1d**2, self.dim)
        x += self.space_pos_embedding#[:, :, :(n + 1)]
        x = self.dropout(x)   # 72  64  128
        cls_space_tokens = repeat(self.space_token, '() N d -> b N d', b=B * self.num_classes)
        x = torch.cat((cls_space_tokens, x), dim=1) # 72  65  128
        if self.return_att:
            x, attn_weights = self.space_transformer(x)  # 4  72  4  65  65  //  layers  b*C  heads  patchs+1  patchs+1
        else:
            x = self.space_transformer(x)
        x_cls_logits = x[:, 0]  # 72  1  128
        x_cls_logits = self.mlp_head(x_cls_logits.reshape(-1, self.dim)) # 72  1 
        x_cls_logits = x_cls_logits.reshape(B, self.num_classes)  # 4  18
        x_patch = x[:, 1:] # 72  64  128
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c]) # 72  8  8  128
        x_patch = x_patch.permute([0, 3, 1, 2]) # 72  128  8  8  
        x_patch = x_patch.contiguous()
        x_patch = self.patch_head(x_patch) # 72  1  8  8 
        x_patch= x_patch.reshape(B, self.num_classes, self.num_patches_1d, self.num_patches_1d) # 4  18  8  8 
        x_patch_logits = torch.mean(x_patch, dim=(2, 3)) # 4  18
        if self.return_att:
            attn_weights = torch.stack(attn_weights) # layers  b*C  heads  patchs+1  patchs+1
            attn_weights = torch.mean(attn_weights, dim=2)  # layers  b*C  patchs+1  patchs+1
            attn_weights = attn_weights.reshape(attn_weights.shape[0], B, self.num_classes, attn_weights.shape[2], attn_weights.shape[3])
            # layers  b  C  patchs+1  patchs+1
            patch_attn = attn_weights[:, :, :, 1:, 1:].mean(2).sum(0) # B  patchs  patchs
            cls_attn = attn_weights[-2:,:,:,:1,1:].mean(0).squeeze().reshape(B, self.num_classes, self.num_patches_1d, self.num_patches_1d) # B  C  h  w
            feature_map = x_patch.detach().clone()  # B * C * h * w
            feature_map = F.relu(feature_map)
            cams = cls_attn * feature_map  # B * C * h * w
            cams = torch.sqrt(cams)
            cams_refine = torch.matmul(patch_attn.unsqueeze(1), cams.view(B, self.num_classes, -1, 1)).reshape(B, self.num_classes, self.num_patches_1d, self.num_patches_1d)
            feature_map_refine = torch.matmul(patch_attn.unsqueeze(1), feature_map.view(B, self.num_classes, -1, 1)).reshape(B, self.num_classes, self.num_patches_1d, self.num_patches_1d)
            # B 1 N N   B C N 1    ->   B C H W
            return x_cls_logits, x_patch_logits, cls_attn, feature_map_refine, cams, cams_refine
        else:
            return x_cls_logits, x_patch_logits
        # x = self.space_transformer(x)
        # x_cls_logits = x[:, 0:self.num_classes].mean(-1)
        # x_patch_logits = x[:, self.num_classes:]
        # n, p, c = x_patch_logits.shape
        # x_patch_logits = torch.reshape(x_patch_logits, [n, int(p ** 0.5), int(p ** 0.5), c])
        # x_patch_logits = x_patch_logits.permute([0, 3, 1, 2])
        # x_patch_logits = x_patch_logits.contiguous()
        # x_patch_logits = self.patch_head(x_patch_logits)

        # x = self.mlp_head(x.reshape(-1, self.dim))
        # x = x.reshape(B, self.num_classes)
        # return x



if __name__ == "__main__":
    res = 24
    model_config = {'img_res': res, 'patch_size': 3, 'patch_size_time': 1, 'patch_time': 4, 'num_classes': 20,
                    'max_seq_len': 16, 'dim': 128, 'temporal_depth': 10, 'spatial_depth': 4, 'depth': 4,
                    'heads': 3, 'pool': 'cls', 'num_channels': 14, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4}
    train_config = {'dataset': "psetae_repl_2018_100_3", 'label_map': "labels_20k2k", 'max_seq_len': 16, 'batch_size': 5,
                    'extra_data': [], 'num_workers': 4}

    model = TSViTcls(model_config).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    img = torch.rand((2, 16, 14, res, res)).cuda()
    out = model(img)
    print("Shape of out :", out.shape)  # [B, num_classes]
