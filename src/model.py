import torch.nn as nn
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from util import *


class model_y_enhance(nn.Module):
    def __init__(self, num_style):
        super(model_y_enhance, self).__init__()

        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)  

        # load pretrained class token for initialize
        self.cls_tokens = nn.ParameterList([nn.Parameter(self.vit.class_token.clone().detach()) for _ in range(num_style)])

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.35)
        self.flatten = nn.Flatten()

        self.layer1 = nn.Sequential(self.fc1,
                                    self.elu,
                                    self.dropout,
                                    self.fc2,
                                    self.elu,
                                    self.dropout,
                                    self.fc3
                                    )

    def forward(self, img, style_list):
        
        # embedded image patch
        img_embedding = self.vit._process_input(img) 

        # embedded style token 
        token_embedding = torch.cat([self.cls_tokens[idx] for idx in style_list], dim=0)  

        # concat token & image
        total_embedding = torch.cat((token_embedding, img_embedding), dim=1)  

        # output of encoder
        token = self.vit.encoder(total_embedding)
        style_token1 = token[:, 0, :]

        # coef prediction
        coef_pred = self.layer1(style_token1)

        return style_token1.unsqueeze(1), coef_pred.T


class model_ccm(nn.Module):
    def __init__(self):
        super(model_ccm, self).__init__()

        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)  

        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.35)

        self.layer1 = nn.Sequential(self.fc1,
                                    self.elu,
                                    self.dropout,
                                    self.fc2,
                                    self.elu,
                                    self.dropout,
                                    self.fc3
                                    )

    def forward(self, rgb, y_enhanced, style_token1):
        
        img = rgb + y_enhanced
        # embedded image patch
        img_embedding = self.vit._process_input(img)  

        # concat token & image
        total_embedding = torch.cat((style_token1, img_embedding), dim=1) 

        # output of encoder
        token = self.vit.encoder(total_embedding)
        style_token2 = token[:, 0, :]

        # ccm prediction
        ccm = self.layer1(style_token2)

        return ccm


class model_total(nn.Module):
    def __init__(self, num_style):
        super(model_total, self).__init__()

        self.weight_predictor = model_y_enhance(num_style)
        self.ccm_predictior = model_ccm()

    def forward(self, rgb, style_list, u_mat):

        style_token1, coef_pred = self.weight_predictor(rgb, style_list)  
        y_enhan, tf = apply_tf(rgb, u_mat, coef_pred)

        ccm_pred = self.ccm_predictior(rgb, y_enhan, style_token1) 
        ccm_applied = apply_ccm(y_enhan, ccm_pred)

        return y_enhan, ccm_applied, coef_pred, ccm_pred, tf

