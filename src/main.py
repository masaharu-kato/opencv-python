import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_optimizer

# ローカルモジュールのインポート
from dataset import RelativeImagePairDataset
from models.unet import UNet
from losses.perceptual_loss import PerceptualLoss
from losses.color_utils import rgb_to_hsv

# --- メインの学習関数 ---
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 特徴量リストを文字列から変換
    features_list = [int(f) for f in args.features.split(',')]

    verbose = bool(args.verbose)
    _tqdm = tqdm if verbose else lambda itr, *args, **kwargs: iter(itr)

    lp_weight = float(args.lp_weight)
    dynamic_lpw_rate = float(args.dynamic_lpw_rate) if args.dynamic_lpw_rate else None
    hsv_s_weight = float(args.hsv_s_weight)

    # モデルのインスタンス化
    # use_se_block と use_cbam は排他的に設定することを推奨
    model = UNet(in_channels=3, out_channels=3, features=features_list, 
                 use_se_block=args.use_se_block and not args.use_cbam, # CBAMと排他的にする
                 use_cbam=args.use_cbam).to(device) # << CBAMを有効にする

    # データセットの準備
    full_dataset = RelativeImagePairDataset(args.dataset_dir, (args.image_width, args.image_height), use_cj=args.use_cj)

    # データの分割
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 最適化手法と損失関数の定義
    # RAdamを使用する場合
    optimizer = pytorch_optimizer.RAdam(model.parameters(), lr=args.learning_rate)
    # AdamWを使用する場合
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)


    # スケジューラーの定義
    # T_0: 最初の周期の長さ（エポック数）
    # T_mult: 各周期の長さを次の周期でどれだけ伸ばすか（1にすると周期長は一定）
    # eta_min: 学習率の最小値
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6) # 例としてT_0=10 (10エポックで学習率をリセット)
        
    # L1 Loss と Perceptual Loss を組み合わせる
    l1_loss = nn.L1Loss(reduction='none') # reduction='none' でピクセルごとのL1誤差を取得
    perceptual_loss = PerceptualLoss().to(device) # Perceptual Loss をインスタンス化

    best_avg_val_l1_loss = float('inf') # L1 Lossは小さい方が良いので無限大で初期化
    best_val_ssim = -1.0 # SSIMは大きい方が良いので-1.0で初期化
    prev_avg_val_l1_loss = float('inf')
    prev_avg_val_ssim = -1.0 # SSIMは大きい方が良いので-1.0で初期化
    
    # 学習ループ
    for epoch in range(args.epochs):
        
        model.train()
        running_l1_loss = 0.0
        running_perceptual_loss = 0.0
        running_hsv_s_loss = 0.0
        running_total_loss = 0.0

        for batch_idx, (input_tensor, clean_tensor, mask_tensor) in enumerate(_tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")):
            input_tensor = input_tensor.to(device)
            clean_tensor = clean_tensor.to(device)
            mask_tensor = mask_tensor.to(device) # マスクもデバイスへ

            optimizer.zero_grad()
            output_tensor = model(input_tensor)

            # L1 Loss の計算 with Mask
            l1_loss_per_pixel = l1_loss(output_tensor, clean_tensor)
            
            # 各ピクセルの全チャンネルのL1誤差を合計 (または平均) して、マスクの形状に合わせる
            if l1_loss_per_pixel.dim() == 4: # Assuming N C H W
                l1_loss_per_pixel_mean_channels = torch.mean(l1_loss_per_pixel, dim=1, keepdim=True) 
            else: # Should not happen with current setup but as a safeguard
                l1_loss_per_pixel_mean_channels = l1_loss_per_pixel

            masked_l1_loss = l1_loss_per_pixel_mean_channels * mask_tensor 
            
            num_valid_pixels = torch.sum(mask_tensor) 
            if num_valid_pixels > 0:
                loss_l1 = torch.sum(masked_l1_loss) / num_valid_pixels
            else:
                loss_l1 = torch.tensor(0.0).to(device)

            total_loss = loss_l1
            running_l1_loss += loss_l1.item()

            # Perceptual Loss の計算 (マスクは適用しない、画像全体で計算)
            if lp_weight > 0: # lp_weightが0より大きい場合のみ計算
                loss_perceptual = perceptual_loss(output_tensor, clean_tensor)
                total_loss += lp_weight * loss_perceptual
                running_perceptual_loss += loss_perceptual.item()

            # --- HSV 彩度損失の追加 ---
            if hsv_s_weight > 0: # hsv_s_weightが0より大きい場合のみ計算
                clean_hsv = rgb_to_hsv(clean_tensor)
                output_hsv = rgb_to_hsv(output_tensor)
                
                hsv_s_loss_per_pixel = l1_loss(output_hsv[:, 1:2, :, :], clean_hsv[:, 1:2, :, :])
                masked_hsv_s_loss = hsv_s_loss_per_pixel * mask_tensor
                
                if num_valid_pixels > 0:
                    loss_hsv_s = torch.sum(masked_hsv_s_loss) / num_valid_pixels
                else:
                    loss_hsv_s = torch.tensor(0.0).to(device)
                
                total_loss += hsv_s_weight * loss_hsv_s
                running_hsv_s_loss += loss_hsv_s.item()
            # --- HSV 彩度損失の追加ここまで ---
            
            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item()

            # スケジューラーを更新 (Optimizer.step() の直後が一般的)
            # 各バッチの進捗を正確に渡す
            scheduler.step(epoch + int(batch_idx / len(train_loader))) 

        avg_l1_loss = running_l1_loss / len(train_loader)
        avg_perceptual_loss = running_perceptual_loss / len(train_loader)
        avg_hsv_s_loss = running_hsv_s_loss / len(train_loader) # val_loaderではなくtrain_loaderで割る
        avg_total_loss = running_total_loss / len(train_loader)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Train Loss: L1={avg_l1_loss:.4f}, Perceptual={avg_perceptual_loss:.4f}, HSV_S={avg_hsv_s_loss:.4f}, Total={avg_total_loss:.4f}")


        # バリデーション
        model.eval()

        val_running_l1_loss = 0.0
        val_running_perceptual_loss = 0.0
        val_running_hsv_s_loss = 0.0
        val_running_total_loss = 0.0
        val_ssim_scores = [] 

        with torch.no_grad():
            for batch_idx, (input_tensor, clean_tensor, mask_tensor) in enumerate(_tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)")):
                input_tensor = input_tensor.to(device)
                clean_tensor = clean_tensor.to(device)
                mask_tensor = mask_tensor.to(device)

                output_tensor = model(input_tensor)

                # L1 Loss の計算 with Mask (訓練時と同じロジック)
                l1_loss_per_pixel = l1_loss(output_tensor, clean_tensor)
                if l1_loss_per_pixel.dim() == 4:
                    l1_loss_per_pixel_mean_channels = torch.mean(l1_loss_per_pixel, dim=1, keepdim=True) 
                else:
                    l1_loss_per_pixel_mean_channels = l1_loss_per_pixel

                masked_l1_loss = l1_loss_per_pixel_mean_channels * mask_tensor 
                
                num_valid_pixels = torch.sum(mask_tensor)
                if num_valid_pixels > 0:
                    loss_l1 = torch.sum(masked_l1_loss) / num_valid_pixels
                else:
                    loss_l1 = torch.tensor(0.0).to(device)

                total_loss = loss_l1
                val_running_l1_loss += loss_l1.item()

                # Perceptual Loss の計算 (訓練時と同じロジック)
                if lp_weight > 0:
                    loss_perceptual = perceptual_loss(output_tensor, clean_tensor)
                    total_loss += lp_weight * loss_perceptual
                    val_running_perceptual_loss += loss_perceptual.item()

                # HSV 彩度損失
                if hsv_s_weight > 0:
                    clean_hsv = rgb_to_hsv(clean_tensor)
                    output_hsv = rgb_to_hsv(output_tensor)
                    
                    hsv_s_loss_per_pixel = l1_loss(output_hsv[:, 1:2, :, :], clean_hsv[:, 1:2, :, :])
                    masked_hsv_s_loss = hsv_s_loss_per_pixel * mask_tensor
                    
                    if num_valid_pixels > 0:
                        loss_hsv_s = torch.sum(masked_hsv_s_loss) / num_valid_pixels
                    else:
                        loss_hsv_s = torch.tensor(0.0).to(device)
                    
                    total_loss += hsv_s_weight * loss_hsv_s
                    val_running_hsv_s_loss += loss_hsv_s.item()

                val_running_total_loss += total_loss.item()
                
                # --- SSIM 計算 ---
                for i in range(input_tensor.shape[0]): 
                    output_img = (output_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    clean_img = (clean_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    current_ssim = ssim_metric(output_img, clean_img, data_range=255, channel_axis=2, win_size=11)
                    val_ssim_scores.append(current_ssim)


        avg_val_l1_loss = val_running_l1_loss / len(val_loader)
        avg_val_perceptual_loss = val_running_perceptual_loss / len(val_loader)
        avg_val_hsv_s_loss = val_running_hsv_s_loss / len(val_loader) 
        avg_val_total_loss = val_running_total_loss / len(val_loader)
        avg_val_ssim = np.mean(val_ssim_scores) if val_ssim_scores else 0.0

        # Save model if L1 Loss and/or SSIM are improved
        f_best_improved = False
        if avg_val_l1_loss < best_avg_val_l1_loss:
            best_avg_val_l1_loss = avg_val_l1_loss
            f_best_improved = True

        if avg_val_ssim > best_val_ssim:
            best_val_ssim = avg_val_ssim
            f_best_improved = True

        f_prev_improved = (avg_val_l1_loss < prev_avg_val_l1_loss) or (avg_val_ssim > prev_avg_val_ssim)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] {'*' if f_best_improved else '+' if f_prev_improved else '-'} Validation Loss: L1={avg_val_l1_loss:.4f}, Perceptual={avg_val_perceptual_loss:.4f} (w={lp_weight:.4f}), HSV_S={avg_val_hsv_s_loss:.4f}, Total={avg_val_total_loss:.4f}, SSIM={avg_val_ssim:.4f}")

        if f_best_improved:
            model_save_path = f"{args.model_save_dir}/best_unet_model_l1loss_{avg_val_l1_loss:.2f}_ssim_{avg_val_ssim:.2f}.pth"
            os.makedirs(args.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            # print(f"Best model saved to {model_save_path}")

        if dynamic_lpw_rate:
            if f_best_improved: # f_prev_improved:
                lp_weight *= dynamic_lpw_rate
            else:
                lp_weight /= dynamic_lpw_rate

        prev_avg_val_l1_loss = avg_val_l1_loss
        prev_avg_val_ssim = avg_val_ssim

        

# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の強化版学習スクリプト。")
    parser.add_argument("dataset_dir", type=str, 
                        help="データセットディレクトリ（動画ごとのファイル名のサブディレクトリを含む）")
    parser.add_argument("model_save_dir", type=str,
                        help="モデルの保存先ディレクトリ")
    parser.add_argument("--image_height", type=int, default=544,
                        help="モデルへの入力画像高さ。")
    parser.add_argument("--image_width", type=int, default=720,
                        help="モデルへの入力画像幅。")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("--epochs", type=int, default=100,
                        help="学習エポック数。")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="初期学習率。")
    parser.add_argument("-lp", "--lp_weight", type=float, default=1.0,
                        help="lp_weight")
    parser.add_argument("-lpr", "--dynamic_lpw_rate", type=float, default=None,
                        help="dynamic_lpw_rate")
    parser.add_argument("-hs", "--hsv_s_weight", type=float, default=1.0,
                        help="hsv_s_weight")
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="学習データセットの割合。残りは検証データセット。")
    parser.add_argument("--num_workers", type=int, default=(os.cpu_count() or 2) // 2,
                        help="データローダーが使用するワーカースレッド数。")
    parser.add_argument("--no_cuda", action="store_true",
                        help="CUDA (GPU) を使用しない場合、このフラグを設定。")
    parser.add_argument("--features", type=str, default="64,128,256",
                        help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("-seb", "--use_se_block", action="store_true",
                        help="Squeeze-and-Excitation (SE) Blockを使用する場合、このフラグを設定。")
    parser.add_argument("-cbam", "--use_cbam", action="store_true", # << CBAM用フラグを追加
                        help="Convolutional Block Attention Module (CBAM) を使用する場合、このフラグを設定。")
    parser.add_argument("-cj", "--use_cj", action="store_true",
                        help="use ColorJitter on the dataset")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show train and progress message")

    args = parser.parse_args()

    train_model(args)