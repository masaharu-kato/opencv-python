import torch
import torch.nn as nn
from torchvision import models, transforms

# Perceptual Loss (VGG Loss) の定義
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[2, 7, 16, 25, 34]): # VGG19のrelu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        super(PerceptualLoss, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.feature_extractor = nn.ModuleList()
        current_layer_idx = 0 # VGG19の層のインデックス
        relu_count = 0 # ReLU層の後に特徴抽出層のインデックスをカウントするためのカウンター

        for i, layer in enumerate(vgg19): # type: ignore
            if isinstance(layer, nn.Conv2d):
                self.feature_extractor.append(layer)
                current_layer_idx += 1
            elif isinstance(layer, nn.ReLU):
                # inplace=False に変更する (勾配計算に影響を与えないため)
                self.feature_extractor.append(nn.ReLU(inplace=False))
                current_layer_idx += 1
                relu_count += 1 # ReLU層の後に特徴抽出層のインデックスをカウント

            # feature_layers に達したらループを抜ける
            if relu_count in feature_layers:
                break # 目的の層に達したらループを抜ける
            
        # 勾配計算を無効化し、評価モードにする
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
                
        self.feature_layers = feature_layers # feature_layers は、Relu層の後のカウントと一致させる
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_image, target_image):
        # VGGの入力要件に合わせて画像を正規化
        pred_image = self.normalize(pred_image)
        target_image = self.normalize(target_image)

        loss = torch.tensor(0.0, device=pred_image.device) 

        x_pred = pred_image
        x_target = target_image
        
        relu_counter = 0 # ReLU層の後のカウント
        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if isinstance(layer, nn.ReLU):
                relu_counter += 1
            
            # 指定された層で損失を計算 (ReLU層の後の特徴を使用)
            if relu_counter in self.feature_layers: 
                loss += self.mse_loss(x_pred, x_target)
        
        return loss