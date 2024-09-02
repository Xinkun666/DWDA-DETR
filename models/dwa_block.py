import torch
torch.set_printoptions(threshold=10_000)
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_

class DeformableWindowAttention(nn.Module):
    def __init__(self, size,dim, num_heads,attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.size = size
        self.attn_drop = nn.Dropout(attn_drop,inplace=True)
        self.proj_drop = nn.Dropout(proj_drop,inplace=True)
        self.proj_out = nn.Linear(dim, dim)
        self.heads = num_heads
        self.window_size = int(math.ceil(math.sqrt(self.dim)))+1 if int(math.ceil(math.sqrt(self.dim)))%2 == 0 else int(math.ceil(math.sqrt(self.dim)))
        if self.window_size > 2 * size - 1:
            self.window_size = 2 * size -1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.window_size,self.window_size,num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)
        self.padding_size = self.window_size//2
        # self.proj_mask = nn.Sequential(
        #     nn.Linear(dim,self.window_size*self.window_size,bias=False),
        #     nn.GELU()
        # )
        self.qkv = nn.Linear(dim,dim*3)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.size and C == self.dim, 'dimension or size not fitness'
        attn_mask = torch.ones(1,1,H,W)
        attn_mask = F.pad(attn_mask,(self.padding_size,) * 4)
        attn_mask_w = F.unfold(attn_mask,kernel_size=(self.window_size, self.window_size), padding=0, stride=1)
        attn_mask_w = rearrange(attn_mask_w,'b c n -> b n c')
        x_total = rearrange(x,'b c h w -> b (h w) c')
        # kv_mask = self.proj_mask(x_total)
        # kv_mask = torch.where(kv_mask > 0.5, torch.ones_like(kv_mask), torch.zeros_like(kv_mask))
        qkv = self.qkv(x_total)
        q,k,v = torch.chunk(qkv,3,dim=2)
        q = q * self.scale
        q = rearrange(q,'b n (h c1) -> b h n c1',h=self.heads)
        q = q.unsqueeze(-2)
        k = rearrange(k,'b (h w) c -> b c h w',h=H)
        v = rearrange(v,'b (h w) c -> b c h w',h=H)
        k_pad = F.pad(k,(self.padding_size,)*4)
        v_pad = F.pad(v,(self.padding_size,)*4)
        k_w = F.unfold(k_pad, kernel_size=(self.window_size, self.window_size), padding=0, stride=1)
        k_w = rearrange(k_w,'b (n1 c) n2 -> b n2 n1 c',n1=self.window_size*self.window_size)
        v_w = F.unfold(v_pad, kernel_size=(self.window_size, self.window_size), padding=0, stride=1)
        v_w = rearrange(v_w, 'b (n1 c) n2 -> b n2 n1 c', n1=self.window_size * self.window_size)
        k_w,v_w = [rearrange(t,'b n1 n2 (h c1) -> b h n1 n2 c1',h=self.heads) for t in [k_w,v_w]]
        attn = torch.einsum('b h n m c, b h n l c -> b h n m l',q,k_w)
        attn = rearrange(attn,'b h n m l -> b h n (m l)')
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # kv_mask = kv_mask.unsqueeze(1).unsqueeze(-1).repeat(1,self.heads,1,1,1)
        attn_mask_w = attn_mask_w.unsqueeze(1).unsqueeze(-2).repeat(1,self.heads,1,1,1)
        # v_w = v_w * kv_mask
        attn = rearrange(attn, 'b h n (m1 m2) -> b h n m1 m2', m1=1)
        attn = attn * attn_mask_w
        x = torch.einsum('b h n m l, b h n l c -> b h n m c',attn,v_w)
        x = rearrange(x,'b h n1 n2 c -> b n1 n2 (h c)')
        x = x.squeeze(2)
        x = self.proj_drop(self.proj_out(x))
        x = rearrange(x,'b (w h) c -> b c w h',w=W)
        return x,None,None

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

# # Example usage
# dim = 3  # Example channel dimension
# model = DeformableWindowAttention(dim)
# input_tensor = torch.randn(1, 3, 7, 7)  # Example input
# output = model(input_tensor)
# print(output.shape)  # Should match input shape
if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    # 假设输入张量x的形状为[4, 5, 5, 3]，其中4是批次大小
    x = torch.randn(1, 3, 4, 4)

    model = DeformableWindowAttention(size=4,dim=3,num_heads=3)
    output = model(x)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
