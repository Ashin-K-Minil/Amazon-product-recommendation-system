import torch
import torch.nn as nn

class HybridRecommender(nn.Module):
    def __init__(self, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type

        self.image_proj = nn.Linear(2048, 512)
        self.review_proj = nn.Linear(768, 512)

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )
        elif fusion_type == "attention":
            self.attn = nn.Linear(512, 1)
            self.fusion = nn.Sequential(
                nn.Linear(512, 128)
            )

    def forward(self, img_feat, review_feat):
        img_proj = self.image_proj(img_feat)
        if img_feat.shape[0] != review_feat.shape[0]:
            review_feat = review_feat.expand(img_feat.shape[0], -1)
        review_proj = self.review_proj(review_feat)

        if self.fusion_type == "concat":
            x = torch.cat((img_proj, review_proj), dim=1)
        elif self.fusion_type == "attention":
            alpha = torch.softmax(torch.cat((
                self.attn(img_proj),
                self.attn(review_proj)
            ), dim= 1), dim=1)
            x = alpha[:, 0:1] * img_proj + alpha[:, 1:2] * review_proj

        return self.fusion(x)