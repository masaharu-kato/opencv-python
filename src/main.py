import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_optimizer
import random

# ローカルモジュールのインポート
from dataset import RelativeImagePairDataset
from models.unet import UNet
import lpips

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True # for reproducibility in CuDNN
    # torch.backends.cudnn.benchmark = False # for reproducibility in CuDNN

# --- メインの学習関数 ---
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 特徴量リストを文字列から変換
    features_list = [int(f) for f in args.features.split(',')]

    verbose = bool(args.verbose)
    _tqdm = tqdm if verbose else lambda itr, *args, **kwargs: iter(itr)

    lp_weight = float(args.lp_weight)
    # hsv_s_weight は削除

    # モデルのインスタンス化
    model = UNet(in_channels=3, out_channels=3, features=features_list, 
                 use_se_block=args.use_se_block and not args.use_cbam,
                 use_cbam=args.use_cbam).to(device)

    # データセットの準備
    full_dataset = RelativeImagePairDataset(args.dataset_dir, (args.image_width, args.image_height))

    # データの分割
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # --- ハードマイニングのための初期設定 ---
    num_train_samples = len(train_dataset) # train_dataset (Subset) のサイズ

    # train_dataset.indices は、full_dataset の中で train_dataset に含まれる要素の元のインデックスのリスト
    # このリストを逆引きできるマップを作成する
    # これは train_model 関数内で一度だけ構築すればよい
    full_idx_to_subset_idx_map = {original_idx: i for i, original_idx in enumerate(train_dataset.indices)}
    
    # 最初のepochで使用する均等な重みで sample_weights を初期化
    sample_weights = torch.ones(num_train_samples, dtype=torch.float32)

    sampler = WeightedRandomSampler(sample_weights, num_samples=num_train_samples, replacement=True) # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # カスタムサンプラーの作成
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_train_samples, replacement=True) # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # ----------------------------------------

    # 最適化手法と損失関数の定義
    optimizer = pytorch_optimizer.RAdam(model.parameters(), lr=args.learning_rate)

    # スケジューラーの定義
    # T_0: 最初の周期の長さ（エポック数）
    # T_mult: 各周期の長さを次の周期でどれだけ伸ばすか（1にすると周期長は一定）
    # eta_min: 学習率の最小値
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6) # 例としてT_0=10 (10エポックで学習率をリセット)
        
    # L1 Loss と Perceptual Loss を組み合わせる
    l1_loss_fn = nn.L1Loss(reduction='none') # reduction='none' でピクセルごとのL1誤差を取得
    
    # 'vgg' は通常、Perceptual Lossで使われるVGG-16/19ベース
    # 'alex' や 'squeeze' も選択可能
    # net_type は 'alex', 'vgg', 'squeeze' から選択。'alex'が推奨されることが多い。
    # cuda=True でGPUを使用 (デフォルトはFalse)

    _lpips_loss_fn = lpips.LPIPS(net='alex', spatial=False).to(device) # spatial=Falseで通常のLPIPS
    _lpips_loss_fn.eval()

    def lpips_loss_fn(output_tensor, clean_tensor):
        scaled_output = output_tensor * 2.0 - 1.0
        scaled_clean = clean_tensor * 2.0 - 1.0
        with warnings.catch_warnings(category=UserWarning):
            return _lpips_loss_fn(scaled_output, scaled_clean)


    best_avg_val_total_loss = float('inf')
    prev_avg_val_total_loss = float('inf')
    best_avg_val_ssim = 0
    prev_avg_val_ssim = 0
    
    # 学習ループ
    for epoch in range(args.epochs):
        model.train()
        running_l1_loss = 0.0
        running_perceptual_loss = 0.0
        running_total_loss = 0.0

        current_epoch_train_losses = torch.zeros(num_train_samples, dtype=torch.float32)

        # DataLoaderから original_indices_in_subset を受け取る
        # これは train_dataset (Subset) 内でのインデックス
        for batch_idx, (input_tensor, clean_tensor, mask_tensor, original_full_dataset_indices) in enumerate(_tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)")):
            input_tensor = input_tensor.to(device)
            clean_tensor = clean_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            optimizer.zero_grad()
            output_tensor = model(input_tensor)

            # L1 Loss の計算 with Mask
            l1_loss_per_pixel = l1_loss_fn(output_tensor, clean_tensor)
            if l1_loss_per_pixel.dim() == 4:
                l1_loss_per_sample_spatial = torch.mean(l1_loss_per_pixel, dim=1)
            else:
                l1_loss_per_sample_spatial = l1_loss_per_pixel

            masked_l1_loss_per_sample = l1_loss_per_sample_spatial * mask_tensor.squeeze(1)
            current_l1_losses_batch = torch.sum(masked_l1_loss_per_sample, dim=[1,2]) / (torch.sum(mask_tensor.squeeze(1), dim=[1,2]) + 1e-6)
            loss_l1 = torch.mean(current_l1_losses_batch)
            
            # Perceptual Loss の計算
            loss_perceptual = torch.tensor(0.0).to(device)
            perceptual_losses_batch = torch.zeros_like(current_l1_losses_batch, device=device) # 仮の初期化
            if lp_weight > 0:
                perceptual_losses_batch = lpips_loss_fn(output_tensor, clean_tensor)
                loss_perceptual = torch.mean(perceptual_losses_batch)

            total_loss = loss_l1 + lp_weight * loss_perceptual

            total_loss.backward()
            optimizer.step()

            running_l1_loss += loss_l1.item()
            running_perceptual_loss += loss_perceptual.item()
            running_total_loss += total_loss.item()
            
            # ハードマイニングのための、現在のバッチの各サンプルのTotal Lossを記録
            if args.use_hard_mining:
                current_batch_total_losses = current_l1_losses_batch + lp_weight * perceptual_losses_batch
                
                # `original_full_dataset_indices` を `train_dataset` 内の相対インデックスに変換
                # ここでは、batch_size 毎にループして変換する
                for i, full_idx in enumerate(original_full_dataset_indices.cpu().numpy()):
                    # `full_idx_to_subset_idx_map` を使って、full_datasetのインデックスからtrain_dataset内のインデックスを取得
                    subset_idx = full_idx_to_subset_idx_map[full_idx]
                    current_epoch_train_losses[subset_idx] = current_batch_total_losses[i].item()


        # エポック終了時: サンプリング重みを更新 (ハードマイニングが有効な場合のみ)
        if args.use_hard_mining:
            epsilon = 1e-6
            temperature = 10.0 # 調整可能な「温度」パラメータ

            # 損失を指数関数的に重み付けし、正規化
            # ここでは損失の指数関数的な増加を使うが、他の方法も検討可能
            sample_weights = torch.exp(current_epoch_train_losses / temperature)
            sample_weights = sample_weights / sample_weights.sum() # 正規化して確率分布にする

            # 新しいサンプラーをDataLoaderに設定
            sampler = WeightedRandomSampler(sample_weights, num_samples=num_train_samples, replacement=True)  # type: ignore
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

        # ----------------------------------------
        
        avg_l1_loss = running_l1_loss / len(train_loader)
        avg_perceptual_loss = running_perceptual_loss / len(train_loader)
        avg_total_loss = running_total_loss / len(train_loader)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Train Loss: L1={avg_l1_loss:.4f}, Perceptual={avg_perceptual_loss:.4f}, Total={avg_total_loss:.4f}")


        # バリデーション
        model.eval()

        val_running_l1_loss = 0.0
        val_running_perceptual_loss = 0.0
        val_running_total_loss = 0.0
        val_ssim_scores = [] 

        with torch.no_grad():
            for batch_idx, (input_tensor, clean_tensor, mask_tensor, _) in enumerate(_tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)")):
                input_tensor = input_tensor.to(device)
                clean_tensor = clean_tensor.to(device)
                mask_tensor = mask_tensor.to(device)

                output_tensor = model(input_tensor)

                # L1 Loss の計算 with Mask (訓練時と同じロジック)
                l1_loss_per_pixel = l1_loss_fn(output_tensor, clean_tensor)
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

                # Perceptual Loss の計算
                loss_perceptual = torch.tensor(0.0).to(device)
                if lp_weight > 0:
                    perceptual_losses_batch = lpips_loss_fn(output_tensor, clean_tensor)
                    loss_perceptual = torch.mean(perceptual_losses_batch)

                total_loss = loss_l1 + lp_weight * loss_perceptual
                
                val_running_l1_loss += loss_l1.item()
                val_running_perceptual_loss += loss_perceptual.item()
                val_running_total_loss += total_loss.item()
                
                # --- SSIM 計算 ---
                for i in range(input_tensor.shape[0]): 
                    output_img = (output_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    clean_img = (clean_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    current_ssim = ssim_metric(output_img, clean_img, data_range=255, channel_axis=2, win_size=11)
                    val_ssim_scores.append(current_ssim)

        avg_val_l1_loss = val_running_l1_loss / len(val_loader)
        avg_val_perceptual_loss = val_running_perceptual_loss / len(val_loader)
        avg_val_total_loss = val_running_total_loss / len(val_loader)
        avg_val_ssim = np.mean(val_ssim_scores) if val_ssim_scores else 0.0

        # スケジューラーを更新 (バリデーション損失を渡す)
        # scheduler.step(avg_val_total_loss)

        # Save model if Total Loss is improved
        f_best_improved = False
        if avg_val_total_loss < best_avg_val_total_loss:
            best_avg_val_total_loss = avg_val_total_loss
            f_best_improved = True
        if avg_val_ssim > best_avg_val_ssim:
            best_avg_val_ssim = avg_val_ssim
            f_best_improved = True

        f_prev_improved = (avg_val_total_loss < prev_avg_val_total_loss) or (avg_val_ssim > prev_avg_val_ssim)
        
        print(f"Epoch [{epoch+1:4d}/{args.epochs:4d}] {'*' if f_best_improved else '+' if f_prev_improved else '-'} V-Loss: L1={avg_val_l1_loss:.4f}, LPIPS={avg_val_perceptual_loss:.4f}, Total={avg_val_total_loss:.4f}, SSIM={avg_val_ssim:.4f}")

        if f_best_improved:
            model_save_path = f"{args.model_save_dir}/model_{epoch+1:04d}_l1_{avg_val_l1_loss:.2f}_lp_{avg_val_perceptual_loss:.2f}_ssim_{avg_val_ssim:.2f}.pth"
            os.makedirs(args.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

        prev_avg_val_total_loss = avg_val_total_loss
        prev_avg_val_ssim = avg_val_ssim


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の強化版学習スクリプト。")
    parser.add_argument("dataset_dir", type=str, 
                        help="データセットディレクトリ（動画ごとのファイル名のサブディレクトリを含む）")
    parser.add_argument("model_save_dir", type=str,
                        help="モデルの保存先ディレクトリ")
    parser.add_argument("--image_height", type=int, default=192,
                        help="モデルへの入力画像高さ。")
    parser.add_argument("--image_width", type=int, default=256,
                        help="モデルへの入力画像幅。")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("--epochs", type=int, default=100,
                        help="学習エポック数。")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="初期学習率。")
    parser.add_argument("-lp", "--lp_weight", type=float, default=1.0,
                        help="lp_weight")
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
    parser.add_argument("-cbam", "--use_cbam", action="store_true",
                        help="Convolutional Block Attention Module (CBAM) を使用する場合、このフラグを設定。")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show train and progress message")
    parser.add_argument("-hm", "--use_hard_mining", action="store_true", # ハードマイニング用フラグ
                        help="ハードマイニングを有効にする場合、このフラグを設定。")
    parser.add_argument("-rseed", "--random_seed", type=int, required=True,
                        help="Random seed")

    args = parser.parse_args()

    set_seed(args.random_seed)
    train_model(args)