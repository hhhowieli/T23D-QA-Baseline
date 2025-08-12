import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts):
        # tokenized_prompts = clip.tokenize(prompts).cuda()
        tokenized_prompts = prompts
        prompts = x = self.token_embedding(tokenized_prompts).type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x_all = x @ self.text_projection
        x = x[torch.arange(x.shape[0]).to(x.device), tokenized_prompts.argmax(dim=-1).to(x.device)] @ self.text_projection.to(x.device)
        
        return x, x_all
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, query_dim, kv_dim, num_heads, output_dim=None):
        super(MultiHeadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim if output_dim else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, query, key, value, mask=None, return_attn=False):
        batch_size = query.size(0)
        L = key.size(1)

        # Linear projections
        q = self.q_proj(query) # NLC
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape and transpose for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # print(scores.shape)
        if mask is not None:
            if mask.dim() <= 3:
                mask = mask[:,None,:,None]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Combine heads
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        if return_attn:
            return output, attn
        return output


class TextVisualCA(nn.Module):
    def __init__(
        self,
        embed_dim,
        text_dim,
        visual_dim,
        output_dim,
        num_heads,
    ):
        super().__init__()
        
        output_dim = visual_dim if output_dim is None else output_dim
        
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        
        self.ca = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        
    def forward(self, text_embedding, visual_embedding, return_attn=False):
        
        if text_embedding.dim() <= 2:
            text_embedding = text_embedding.unsqueeze(1)
        
        text_embedding = self.text_proj(text_embedding)
        visual_embedding = self.visual_proj(visual_embedding)
            
        res, attn = self.ca(text_embedding, visual_embedding, visual_embedding, return_attn=True)
        
        # if res.dim() >= 3:
        #     res = res.squeeze(1)
        
        if return_attn:
            return res, attn
        
        return res
            

class ViewSpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim):
        super().__init__()
        
        output_dim = embed_dim if output_dim is None else output_dim
        
        self.ca = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        
    def forward(self, feats):
        """
            feats: B V N D
        """
        B, D = feats.size(0), feats.size(3)
        cls_token = feats[:, :, 0, :]
        # feats = feats.reshape(B, -1, D)
        # print(cls_token.shape)
        res = self.ca(cls_token, feats, feats)
        
        return res



class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet,self).__init__()
        self.fc1w_conv = nn.Conv2d(in_channels=112,out_channels=512,kernel_size=3,padding=1)
        self.fc1b_fc = nn.Linear(112, 112)
        self.fc2w_conv = nn.Conv2d(in_channels=112,out_channels=128,kernel_size=3,padding=1)
        self.fc2b_fc = nn.Linear(112, 56)
        self.fc3w_conv = nn.Conv2d(in_channels=112,out_channels=32,kernel_size=3,padding=1)
        self.fc3b_fc = nn.Linear(112, 28)
        self.fc4w_fc= nn.Linear(112,28)
        self.fc4b_fc = nn.Linear(112, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        
        fc1w = self.fc1w_conv(x).view(-1, 112,224, 1, 1)
        fc2w = self.fc2w_conv(x).view(-1, 56,112, 1, 1)
        fc3w = self.fc3w_conv(x).view(-1, 28,56, 1, 1)
        fc4w = self.fc4w_fc(self.pool(x).squeeze()).view(-1, 1,28, 1, 1)

        fc1b = self.fc1b_fc(self.pool(x).squeeze()).view(-1, 112)
        fc2b = self.fc2b_fc(self.pool(x).squeeze()).view(-1, 56)
        fc3b = self.fc3b_fc(self.pool(x).squeeze()).view(-1, 28)
        fc4b = self.fc4b_fc(self.pool(x).squeeze()).view(-1, 1)
        out = {}
        out['fc1w'] = fc1w
        out['fc2w'] = fc2w
        out['fc3w'] = fc3w
        out['fc4w'] = fc4w
        out['fc1b'] = fc1b
        out['fc2b'] = fc2b
        out['fc3b'] = fc3b
        out['fc4b'] = fc4b
        return out
    
class TargetNet(nn.Module):
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.fc1 = nn.Sequential(
            TargetFC(paras['fc1w'], paras['fc1b']),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            TargetFC(paras['fc2w'], paras['fc2b']),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            TargetFC(paras['fc3w'], paras['fc3b']),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            TargetFC(paras['fc4w'], paras['fc4b']),
        )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
class TargetFC(nn.Module):
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias
    def forward(self, input_):
        input_re = input_
        weight_re = self.weight.squeeze(0)
        bias_re = self.bias.squeeze(0)
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re)
        return out