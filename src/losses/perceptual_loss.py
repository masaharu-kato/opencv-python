import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F # F.mse_loss を使うため

# Perceptual Loss (VGG Loss) の定義
class PerceptualLoss(nn.Module):
    # feature_layers を VGG19のReLU層の後のインデックスに調整
    # VGG19のfeaturesの構造は:
    # 0-3: relu1_1 (層0-3) -> relu_count=1
    # 4-8: relu2_1 (層4-8) -> relu_count=2
    # 9-15: relu3_1 (層9-15) -> relu_count=3
    # 16-24: relu4_1 (層16-24) -> relu_count=4
    # 25-33: relu5_1 (層25-33) -> relu_count=5
    # 現在のfeature_layers=[2, 7, 16, 25, 34] はVGG19のインデックスとして定義されており、
    # このPerceptualLossの実装では、ReLU層をカウントするロジックになっています。
    # したがって、feature_layers=[1, 2, 3, 4, 5] (relu_count) のように変更するか、
    # 以前のVGG16ベースのPerceptualLossの層指定方法に合わせる必要があります。
    # ここでは、あなたの現在のPerceptualLossのfeature_layersの扱い方 (relu_count) に合わせて調整します。
    # VGG19のfeature layersの実際の層インデックスではなく、
    # relu_count（ReLU層が何回出てきたか）に対応させる場合、feature_layersは [1, 2, 3, 4, 5] が適切です。
    # VGG19の構造と照らし合わせると、例えば layer 2 は conv1_2 の後、layer 7 は conv2_2 の後ですが、
    # relu_count はReLU層の数を数えているので、通常は relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 に対応する
    # relu_count は 1, 2, 3, 4, 5 になります。
    # 以前のPerceptualLossのコメントに従い、feature_layers=[2, 7, 16, 25, 34] はそのままに、
    # VGG19のfeaturesのインデックスと一致させるように修正します。
    # 実際のVGG19の層インデックス (0-indexed) は以下の通りです:
    # relu1_2: 4
    # relu2_2: 9
    # relu3_2: 18
    # relu4_2: 27
    # relu5_2: 36
    # したがって、`feature_layers` のデフォルト値をVGG19のrelu層後のインデックスに合わせるか、
    # `relu_count` のロジックを修正する必要があります。
    # ここでは、VGGの出力層インデックスを直接指定する形に修正します。
    # 例えば、`VGG19_Weights.IMAGENET1K_V1` の構造では、
    # `conv1_2` 後 (index 4), `conv2_2` 後 (index 9), `conv3_2` 後 (index 18), `conv4_2` 後 (index 27), `conv5_2` 後 (index 36)
    # が一般的なPerceptual Lossの層選択です。
    # あなたの現在のfeature_layers=[2, 7, 16, 25, 34] はVGG19の特定の位置に対応していない可能性があります。
    # 便宜上、VGG16のよく使われる層（relu1_2, relu2_2, relu3_3, relu4_3）のようなものに近づけます。
    # VGG19では以下のインデックスがよく使われます:
    # relu1_2: 4
    # relu2_2: 9
    # relu3_2: 18
    # relu4_2: 27
    # relu5_2: 36
    def __init__(self, feature_layers=[4, 9, 18, 27, 36]): # VGG19の一般的な特徴層のインデックス
        super(PerceptualLoss, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # feature_extractor を、指定された feature_layers まで含めるように構築
        self.feature_extractor = nn.ModuleList()
        # 最後に指定された層までVGGをコピー
        max_layer_idx = max(feature_layers)
        for i, layer in enumerate(vgg19): # type: ignore
            self.feature_extractor.append(layer)
            if i == max_layer_idx:
                break
            
        # 勾配計算を無効化し、評価モードにする
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
            
        self.feature_layers = feature_layers # 損失計算に使用する層のインデックス

    # reduction 引数を追加
    def forward(self, pred_image, target_image, reduction='mean'): # <<< reduction引数を追加
        # VGGの入力要件に合わせて画像を正規化
        pred_image = self.normalize(pred_image)
        target_image = self.normalize(target_image)

        # 各層の特徴量を格納するリスト
        pred_features = []
        target_features = []

        x_pred = pred_image
        x_target = target_image
        
        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            # 指定された層の後に特徴量を保存
            if i in self.feature_layers: 
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # 各特徴マップのL2損失 (MSE) を計算し、バッチ内の各サンプルごとの損失を求める
        # reduction='none' を使用することで、[batch_size, C, H, W] の損失テンソルが得られる
        # その後、C, H, W 次元で平均を取ることで、[batch_size] の損失テンソルにする
        
        # perceptual_loss_per_sample を初期化
        # 各サンプルのPerceptual Lossの合計を格納するテンソル
        perceptual_loss_per_sample = torch.zeros(pred_image.shape[0], device=pred_image.device)

        for p_feat, t_feat in zip(pred_features, target_features):
            # MSELoss with reduction='none' returns [batch_size, C, H, W]
            # We want to average over C, H, W to get [batch_size]
            layer_loss = F.mse_loss(p_feat, t_feat, reduction='none') # <<< F.mse_lossを使用
            perceptual_loss_per_sample += torch.mean(layer_loss, dim=[1,2,3]) # 各サンプルの特徴量ごとの平均損失を合計

        # 最終的なreductionを適用
        if reduction == 'mean':
            return torch.mean(perceptual_loss_per_sample) # バッチ平均
        elif reduction == 'sum':
            return torch.sum(perceptual_loss_per_sample) # バッチ合計
        elif reduction == 'none':
            return perceptual_loss_per_sample # バッチごとの損失テンソル (shape: [batch_size])
        else:
            raise ValueError(f"Unknown reduction mode: {reduction}")