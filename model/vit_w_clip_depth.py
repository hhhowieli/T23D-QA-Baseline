import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from transformers import AutoModel

from copy import deepcopy
from .utils import TextEncoder, TextVisualCA, ViewSpatialAttention

from einops import rearrange

class ViTWClip(nn.Module):
    def __init__(
        self,
        num_views=6,
        output_dim=1,
        quality_type=["geometry quality", "texture quality", "alignment quality", "overall quality"]
    ):
        super().__init__()
        
        clip_model = clip.load(
            "./ckpt/clip/openai/ViT-B-16.pt", device="cpu"
        )[0]

        self.visual_encoder = clip_model.visual
        self.depth_encoder = deepcopy(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        
        # self.view_embedding = nn.Embedding(num_views, self.visual_encoder.proj.shape[1])
        
        # self.zh_encoder =  AutoModel.from_pretrained("bert-base-chinese")
        # self.zh_head = nn.Linear(self.zh_encoder.config.hidden_size, self.text_encoder.text_projection.shape[1])
        
        # self.hid_feature_dim = self.encoder.proj.shape[0]
        self.feature_dim = self.visual_encoder.proj.shape[1]
        self.pred_cls_dim = self.text_encoder.text_projection.shape[1]
        # self.vit.proj = nn.Identity()
        
        self.text_visual = TextVisualCA(
            embed_dim=self.feature_dim*2,
            text_dim=self.pred_cls_dim,
            visual_dim=self.visual_encoder.proj.shape[1],
            output_dim=self.pred_cls_dim,
            num_heads=8
        )
        
        self.pred_text_enhance = TextVisualCA(
            embed_dim=self.feature_dim*2,
            text_dim=self.pred_cls_dim,
            visual_dim=self.visual_encoder.proj.shape[1],
            output_dim=self.pred_cls_dim,
            num_heads=8
        )
        
        self.visual_sa = TextVisualCA(
            embed_dim=self.feature_dim*2,
            text_dim=self.text_encoder.text_projection.shape[1],
            visual_dim=self.visual_encoder.proj.shape[1],
            output_dim=self.feature_dim,
            num_heads=8
        )
        
        # self.visual_fusion = TextVisualCA(
        #     embed_dim=self.feature_dim*2,
        #     text_dim=self.feature_dim,
        #     visual_dim=self.feature_dim,
        #     output_dim=self.feature_dim,
        #     num_heads=8
        # )

        self.visual_visual = TextVisualCA(
            embed_dim=self.feature_dim*2,
            text_dim=self.pred_cls_dim,
            visual_dim=self.visual_encoder.proj.shape[1],
            output_dim=self.pred_cls_dim,
            num_heads=8
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(self.pred_cls_dim+self.feature_dim, self.feature_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1)
        )
        
        self.regressor_all = nn.Sequential(
            nn.Linear(len(quality_type)*self.feature_dim, self.feature_dim),
            nn.Tanh(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 4)
        )

        self.depth_head = nn.Sequential(
            nn.Linear(self.visual_encoder.proj.shape[1], self.feature_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        self.num_views = num_views
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # for param in self.zh_encoder.parameters():
        #     param.requires_grad = False
            
        prefix_prompt = clip_model.encode_text(
            clip.tokenize(quality_type)
        )
        
        # self.register_buffer("prefix", prefix_prompt)
        # self.pred_cls = nn.Parameter(
        #     torch.cat([prefix_prompt, deepcopy(prefix_prompt)], dim=-1)
        # )
        # prefix_prompt = prefix_prompt.unsqueeze(1).expand(-1, 2, -1).flatten(start_dim=1)
        self.pred_cls = nn.Parameter(prefix_prompt)
        self.pred_cls_f = nn.Parameter(deepcopy(prefix_prompt), requires_grad=False)
        

    def forward(self, view_xs, d_images, order, prompt, prompt_zh=None):
        B, V = view_xs.size(0), view_xs.size(1)
        
        loss_cos = self.calculate_cos(self.pred_cls)
        pred_cls = (self.pred_cls+self.pred_cls_f).unsqueeze(0).expand(B, -1, -1)
        
        prompt_eot, prompt_all = self.text_encoder(prompt)
        # prompt_cls_zh, prompt_all_zh = self.encode_text_zh(prompt_zh, pooling_strategy="cls")
        # prompt_all_zh = self.zh_head(prompt_all_zh)
        
        view_xs = rearrange(view_xs, "b v c h w -> (b v) c h w")
        global_view_visual_feats, view_visual_feats = self.visual_forward(self.visual_encoder, view_xs)

        d_images = rearrange(d_images, "b v c h w -> (b v) c h w")
        global_view_visual_feats_d, view_visual_feats_d = self.visual_forward(self.depth_encoder, d_images)

        view_visual_feats_d = view_visual_feats_d + self.depth_head(view_visual_feats_d)

        view_visual_feats = rearrange(view_visual_feats, "(b v) l d -> b (v l) d", b=B, v=V)
        view_visual_feats_d = rearrange(view_visual_feats_d, "(b v) l d -> b (v l) d", b=B, v=V)

        # pred_cls = pred_cls + self.pred_text_enhance(pred_cls, torch.cat([prompt_all, prompt_all_zh], dim=1))
        pred_cls = pred_cls + self.pred_text_enhance(pred_cls, prompt_all)
        
        # view_visual_feats = view_visual_feats + self.visual_sa(view_visual_feats, view_visual_feats)
        # view_visual_feats_d = view_visual_feats_d + self.visual_sa(view_visual_feats_d, view_visual_feats_d)
        
        # view_visual_feats = view_visual_feats + self.visual_fusion(view_visual_feats, view_visual_feats_d)
        
        pred_cls = pred_cls + self.visual_visual(pred_cls, torch.cat([view_visual_feats, view_visual_feats_d], dim=1))

        geo_text_pred_cls = pred_cls[:,0:2,:]
        geo_text_pred_cls = geo_text_pred_cls + self.text_visual(geo_text_pred_cls, view_visual_feats)
        # geo_text_pred_cls = geo_text_pred_cls + self.visual_visual(geo_text_pred_cls, torch.cat([view_visual_feats, view_visual_feats_d], dim=1))
        
        align_pred_cls = pred_cls[:,2,:].unsqueeze(dim=1)
        align_pred_cls = align_pred_cls + self.text_visual(align_pred_cls, torch.cat([view_visual_feats, prompt_all], dim=1))

        pred_cls = torch.cat([geo_text_pred_cls, align_pred_cls, pred_cls[:,3,:].unsqueeze(1)], dim=1)
        pred_cls[:,3,:] = pred_cls.mean(dim=1)

        # cos_loss = self.cosine_similarity_loss(pred_cls)

        global_visual_feat = rearrange(global_view_visual_feats, "(b v) d -> b v d", b=B).mean(dim=1, keepdim=True).expand(-1, 4, -1)
        pred_cls = torch.cat([pred_cls, global_visual_feat], dim=-1)

        # score = self.regressor_all(pred_cls.flatten(start_dim=1))
        
        score = self.regressor(pred_cls).squeeze()
        return score, loss_cos
        # scores = torch.stack(scores).mean(dim=0)
        
        # return score, cos_loss
        
        # cos_sim = self.cosine_similarity(pred_cls)
        # return score, cos_sim
        
    def visual_forward(self, visual_encoder, x):
        x = visual_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual_encoder.positional_embedding.to(x.dtype)
        x = visual_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # all_x = self.visual_encoder.ln_post(x)
        # all_x = x

        x = self.visual_encoder.ln_post(x)

        if self.visual_encoder.proj is not None:
            x = x @ self.visual_encoder.proj
        all_x = x[:,1:,:]
        x = x[:,0,:]

        return x, all_x
    
    def cosine_similarity_loss(self, pred_cls, epsilon=0):
        pred_cls = F.normalize(pred_cls, dim=-1)
        pred_cls = pred_cls @ pred_cls.transpose(-1, -2)
        pred_cls = pred_cls - torch.eye(pred_cls.size(1), device=pred_cls.device).unsqueeze(0)
        
        cos_loss = F.relu(pred_cls - epsilon)
        cos_loss = cos_loss.mean(dim=-1).mean(dim=-1).mean()

        return cos_loss
    
    def encode_text_zh(self, inputs, pooling_strategy="mean"):
        """
        编码中文文本为特征向量
        :param text_batch: 中文文本列表（支持批量处理）
        :param pooling_strategy: 池化策略（可选 "mean", "cls", "max"）
        :return: 文本特征张量 [batch_size, hidden_size]
        """
        # 文本预处理和分词
        inputs = inputs.to(self.zh_encoder.device)
        # 前向传播获取隐藏状态
        with torch.no_grad():
            outputs = self.zh_encoder(**inputs)
        
        # 获取最后一层隐藏状态 [batch_size, seq_len, hidden_size]
        last_hidden_states = outputs.last_hidden_state
        # 应用池化策略
        if pooling_strategy == "cls":
            # 使用[CLS]标记的特征
            features = last_hidden_states[:, 0, :]
        elif pooling_strategy == "mean":
            # 计算均值池化
            input_mask = inputs["attention_mask"].unsqueeze(-1)
            features = (last_hidden_states * input_mask).sum(1) / input_mask.sum(1)
        elif pooling_strategy == "max":
            # 最大池化
            input_mask = inputs["attention_mask"].unsqueeze(-1)
            features = (last_hidden_states * input_mask).max(1).values
        else:
            raise ValueError(f"不支持的池化策略: {pooling_strategy}")
        return features, last_hidden_states
    
    def calculate_cos(self, features):

        margin = torch.tensor([0.0], device=features.device)
        loss = torch.tensor([], device=features.device)
        num_cond, _ = features.shape
        for i in range(0,num_cond):
            for j in range(i+1,num_cond):
                cos = torch.maximum(F.cosine_similarity(features[i].unsqueeze(0),features[j].unsqueeze(0)),margin)
                loss = torch.cat([loss, cos])
        loss = torch.mean(loss) 

        return loss
    
if __name__=="__main__":
    
    model = ViTWClip()
    x = torch.randn(2, 6, 3, 224, 224)
    out, w = model(x, x, ["1", "2"])
    print(out.shape, w.shape)
    