from typing import Optional
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor
from .mln import MLN, nerf_positional_encoding
from .diffattention.multihead_diffattn import MultiheadDiffAttn
from .diffattention.multihead_diffcrossattn import MultiheadDiffCrossAttn

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
    ):
        super().__init__()
        self.post_norm = post_norm
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,  # ✅ [B, N]
    ):
        src2 = self.norm1(src)
        
        effective_mask = mask
        if attention_bias is not None:
            B, N = src.size(0), src.size(1)
            
            # [B, N] -> [B*num_heads, N, N]
            bias_mask = attention_bias.unsqueeze(1).expand(B, N, N)  # [B, N, N]
            bias_mask = bias_mask.repeat_interleave(self.num_heads, dim=0)  # [B*H, N, N]
            
            if effective_mask is None:
                effective_mask = bias_mask
            else:
                if effective_mask.dim() == 2:
                    # [N, N] -> [B*num_heads, N, N]
                    effective_mask = effective_mask.unsqueeze(0).expand(B * self.num_heads, N, N)
                elif effective_mask.dim() == 3 and effective_mask.size(0) == B:
                    # [B, N, N] -> [B*num_heads, N, N]
                    effective_mask = effective_mask.repeat_interleave(self.num_heads, dim=0)
                effective_mask = effective_mask + bias_mask
        
        src2 = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=effective_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward_post(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,  # ✅
    ):
        effective_mask = mask
        if attention_bias is not None:
            B, N = src.size(0), src.size(1)
            bias_mask = attention_bias.unsqueeze(1).expand(B, N, N)
            bias_mask = bias_mask.repeat_interleave(self.num_heads, dim=0)
            
            if effective_mask is None:
                effective_mask = bias_mask
            else:
                if effective_mask.dim() == 2:
                    effective_mask = effective_mask.unsqueeze(0).expand(B * self.num_heads, N, N)
                elif effective_mask.dim() == 3 and effective_mask.size(0) == B:
                    effective_mask = effective_mask.repeat_interleave(self.num_heads, dim=0)
                effective_mask = effective_mask + bias_mask
        
        src2 = self.attn(
            query=src,
            key=src,
            value=src,
            attn_mask=effective_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,  # ✅
    ):
        if self.post_norm:
            return self.forward_post(
                src=src, 
                mask=mask, 
                key_padding_mask=key_padding_mask,
                attention_bias=attention_bias
            )

        return self.forward_pre(
            src=src, 
            mask=mask, 
            key_padding_mask=key_padding_mask,
            attention_bias=attention_bias
        )


class InteractionBlock(nn.Module):
    def __init__(
        self,
        dim=128,
        pose_dim=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_diff_interaction=False,
    ):
        super().__init__()

        self.mln = MLN(pose_dim * 12)
        self.use_diff_interaction = use_diff_interaction
        if self.use_diff_interaction:
            self.cross_blocks = nn.ModuleList(
                Cross_Block_differential(
                    embed_dim=dim,
                    num_heads=num_heads,
                    depth=i,
                    mlp_ratio=mlp_ratio,
                    drop_path=attn_drop,
                )
                for i in range(2)
            )
        else:
            self.cross_blocks = nn.ModuleList(
                Cross_Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=attn_drop,
                )
                for i in range(2)
            )

    def forward(
        self,
        cur_embed,
        memory_embed,
        cur_pose,
        memory_pose,
        cur_pos_embed=None,
        memory_pos_embed=None,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights=False,
    ):
        cur_pose = nerf_positional_encoding(cur_pose).unsqueeze(1).repeat(1, cur_embed.size(1), 1)
        memory_pose = nerf_positional_encoding(memory_pose).unsqueeze(1).repeat(1, memory_embed.size(1), 1)
        cur_embed = self.mln(cur_embed, cur_pose)
        memory_embed = self.mln(memory_embed, memory_pose)
        if cur_pos_embed is not None:
            cur_embed += cur_pos_embed
        if memory_pos_embed is not None:
            memory_embed += memory_pos_embed

        attn_w = attn_noise = None
        for cross in self.cross_blocks:
            if return_attn_weights and isinstance(cross, Cross_Block_differential):
                cur_embed, attn_w, attn_noise = cross(src=cur_embed, src_kv=memory_embed,mask=mask, key_padding_mask=key_padding_mask,return_attn_weights=True)
            else:
                cur_embed = cross(src=cur_embed, src_kv=memory_embed, mask=mask, key_padding_mask=key_padding_mask)
        if return_attn_weights:
            return cur_embed, attn_w, attn_noise
        return cur_embed    

class Cross_Block(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2_kv = self.norm1(src_kv)
        src2 = self.attn(
            query=src2,
            key=src2_kv,
            value=src2_kv,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):

        return self.forward_pre(src=src, src_kv=src_kv, mask=mask, key_padding_mask=key_padding_mask)

class Block_differential(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiheadDiffAttn(embed_dim, depth, num_heads)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        # rel_pos,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
    ):  
        src2 = self.norm1(src)
        if return_attn_weights:
            src2,attn_output_weights,attention_noise = self.attn(
                x=src2,
                attn_mask=attn_mask,
                key_padding_mask = key_padding_mask,
            )
            
        else:
            src2  = self.attn(
                x=src2,
                attn_mask=attn_mask,
                key_padding_mask = key_padding_mask,
            )[0]
            
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        if return_attn_weights:
            return src, attn_output_weights,attention_noise
        return src

    def forward_post(
        self,
        src,
        # rel_pos,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
    ):
        if return_attn_weights:
            src2,attn_output_weights,attention_noise = self.attn(
                x=src,
                attn_mask=attn_mask,
                key_padding_mask = key_padding_mask,
            )
        else:
            src2 = self.attn(
                x=src,
                attn_mask=attn_mask,
                key_padding_mask = key_padding_mask,
            )
        
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        if return_attn_weights:
            return src, attn_output_weights,attention_noise
        return src

    def forward(
        self,
        src,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
    ):
        if self.post_norm:
            return self.forward_post(
                src=src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, return_attn_weights=return_attn_weights
            )

        return self.forward_pre(
            src=src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, return_attn_weights=return_attn_weights
        )

class Cross_Block_differential(nn.Module):
    def __init__(
        self,
        depth,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiheadDiffCrossAttn(embed_dim, depth, num_heads)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
    ):  
        src2 = self.norm1(src)
        src2_kv = self.norm1(src_kv)
        if return_attn_weights:
            src2,attn_output_weights,attention_noise = self.attn(
                x=src2,
                context_x = src2_kv,
                mask=mask,
                key_padding_mask = key_padding_mask,
            )
            
        else:
            src2  = self.attn(
                x=src2,
                context_x = src2_kv,
                mask=mask,
                key_padding_mask = key_padding_mask,
            )[0]
            
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        if return_attn_weights:
            return src, attn_output_weights,attention_noise
        return src

    def forward_post(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
    ):
        if return_attn_weights:
            src2,attn_output_weights,attention_noise = self.attn(
                x=src,
                context_x = src_kv,
                mask=mask,
                key_padding_mask = key_padding_mask,
            )
        else:
            src2 = self.attn(
                x=src,
                context_x = src_kv,
                mask=mask,
                key_padding_mask = key_padding_mask,
            )
        
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        if return_attn_weights:
            return src, attn_output_weights,attention_noise
        return src

    def forward(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
    ):
        if self.post_norm:
            return self.forward_post(
                src=src, src_kv = src_kv, mask=mask, key_padding_mask=key_padding_mask, return_attn_weights=return_attn_weights
            )

        return self.forward_pre(
            src=src, src_kv = src_kv, mask=mask, key_padding_mask=key_padding_mask, return_attn_weights=return_attn_weights
        )