import argparse
import os
import random
import warnings
import numpy as np
import lpips
import pytorch_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ローカルモジュールのインポート
from dataset import RelativeImagePairDataset
from models.unet import UNet
from models.discriminator import Discriminator

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True # for reproducibility in CuDNN
    # torch.backends.cudnn.benchmark = False # for reproducibility in CuDNN

# --- メインの学習関数 ---
def train_model(args):

    dataset_dir = str(args.dataset_dir)
    model_save_dir = str(args.model_save_dir)
    in_channels = int(args.in_channels)
    out_channels = int(args.in_channels)
    in_size = (int(args.image_width), int(args.image_height))
    features = [int(f) for f in args.features.split(',')]
    gan_features = [int(f) for f in args.gan_features.split(',')] if args.gan_features else None
    learning_rate = float(args.learning_rate)
    gan_learning_rate = float(args.gan_learning_rate) if args.gan_learning_rate else learning_rate
    train_split = float(args.train_split)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    num_workers = int(args.num_workers)
    lp_weight = float(args.lp_weight)
    hm_temperature = float(args.hm_temperature) if args.hm_temperature else None # Hard-mining temperature parameter
    gan_weight = float(args.gan_weight) if args.gan_weight else None
    use_se_block = bool(args.use_se_block)
    use_cbam = bool(args.use_cbam)
    log_dir = str(args.log_dir) if args.log_dir else None
    verbose = bool(args.verbose)
    progress = bool(args.progress)

    log_file_path = os.path.join(log_dir, os.path.basename(model_save_dir) + ".log") if log_dir else None
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, mode='a') if log_file_path else None

    def _print(*args, is_verbose=False, to_file=True, to_stdout=True, flush=False):
        if not is_verbose or verbose:
            if to_stdout:
                print(*args, flush=flush)
            if log_file and to_file:
                dt = datetime.now().isoformat(sep=' ', timespec='seconds')
                print(f"{dt}\t", *args, file=log_file, flush=flush)

    _print(args, to_stdout=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"Device: {device}")
    
    # モデルのインスタンス化
    model = UNet(in_channels=in_channels, out_channels=out_channels, in_size=in_size, features=features, 
                 use_se_block=use_se_block and not use_cbam,
                 use_cbam=use_cbam).to(device)
    # 最適化手法と損失関数の定義
    optimizer = pytorch_optimizer.RAdam(model.parameters(), lr=learning_rate)
    
    # Discriminatorの定義
    # Generatorの入力チャネル(3) + ターゲット画像チャネル(3) = 6
    discriminator: Discriminator | None = None
    optimizer_d: pytorch_optimizer.RAdam | None = None
    if gan_weight:
        if not gan_features:
            raise RuntimeError("gan_features is not specified.")
        discriminator = Discriminator(in_channels=in_channels + out_channels, features=gan_features).to(device)
        optimizer_d = pytorch_optimizer.RAdam(discriminator.parameters(), lr=gan_learning_rate, betas=(0.5, 0.999)) # D用のOptimizer (通常はβ1=0.5)

    # データセットの準備
    full_dataset = RelativeImagePairDataset(dataset_dir, in_size, 2 ** len(features))

    # データの分割
    train_size = int(train_split * len(full_dataset))
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # カスタムサンプラーの作成
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_train_samples, replacement=True) # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # ----------------------------------------


    # スケジューラーの定義
    # T_0: 最初の周期の長さ（エポック数）
    # T_mult: 各周期の長さを次の周期でどれだけ伸ばすか（1にすると周期長は一定）
    # eta_min: 学習率の最小値
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6) # 例としてT_0=10 (10エポックで学習率をリセット)
        
    # L1 Loss と Perceptual Loss を組み合わせる
    l1_loss_fn = nn.L1Loss(reduction='none') # reduction='none' でピクセルごとのL1誤差を取得
    
    # 'vgg' は通常、Perceptual Lossで使われるVGG-16/19ベース
    # 'alex' や 'squeeze' も選択可能
    # net_type は 'alex', 'vgg', 'squeeze' から選択。'alex'が推奨されることが多い。
    # cuda=True でGPUを使用 (デフォルトはFalse)

    warnings.simplefilter('ignore', category=UserWarning)

    _lpips_loss_fn = lpips.LPIPS(net='alex', spatial=False).to(device) # spatial=Falseで通常のLPIPS
    _lpips_loss_fn.eval()

    def lpips_loss_fn(scaled_output: Tensor, scaled_clean: Tensor):
        perceptual_losses_batch = _lpips_loss_fn(scaled_output, scaled_clean)
        if perceptual_losses_batch.dim() > 1: # 先ほど修正した部分
            perceptual_losses_batch = perceptual_losses_batch.squeeze()
        return perceptual_losses_batch
        
    # LSGANの損失関数
    def lsgan_loss(predictions, target_is_real):
        if target_is_real:
            target_tensor = torch.ones_like(predictions)
        else:
            target_tensor = torch.zeros_like(predictions)
        return F.mse_loss(predictions, target_tensor) # MSE Lossを使用


    best_avg_total_loss = float('inf')
    prev_avg_total_loss = float('inf')
    best_avg_ssim = 0
    prev_avg_ssim = 0
    
    # 学習ループ
    for epoch in range(epochs):
        model.train()
        val_l1_loss = 0.0
        val_perceptual_loss = 0.0
        val_gan_loss_g = 0.0
        val_discriminator_loss = 0.0
        val_loss_d_real = 0.0
        val_loss_d_fake = 0.0
        val_total_loss = 0.0


        current_epoch_train_losses = torch.zeros(num_train_samples, dtype=torch.float32)

        # DataLoaderから original_indices_in_subset を受け取る
        # これは train_dataset (Subset) 内でのインデックス
        for batch_idx, (input_tensor, clean_tensor, mask_tensor, original_full_dataset_indices) in enumerate(tqdm(train_loader, disable=not progress, desc=f"Epoch {epoch+1} (Train)")):
            input_tensor: Tensor = input_tensor.to(device)
            clean_tensor: Tensor = clean_tensor.to(device)
            mask_tensor: Tensor = mask_tensor.to(device)

            # ---------------------
            # 1. Discriminator の学習
            # ---------------------
            loss_d: Tensor | None = None
            if discriminator and optimizer_d:
                optimizer_d.zero_grad()

                # a. 本物画像に対する損失
                # Discriminatorの入力は (Generatorの入力画像, ターゲット画像)
                real_input_d = torch.cat([input_tensor, clean_tensor], dim=1) # 入力画像を条件として結合
                pred_real = discriminator(real_input_d)
                loss_d_real = lsgan_loss(pred_real, True)

                # b. 偽画像に対する損失
                # Generatorからの出力を使用 (detachしてGeneratorの勾配計算を防ぐ)
                with torch.no_grad(): # Generatorの勾配はここでは計算しない
                    fake_output_g = model(input_tensor).detach()
                # Discriminatorの入力は (Generatorの入力画像, 生成画像)
                fake_input_d = torch.cat([input_tensor, fake_output_g], dim=1) # 入力画像を条件として結合
                pred_fake = discriminator(fake_input_d)
                loss_d_fake = lsgan_loss(pred_fake, False)

                val_loss_d_real += loss_d_real
                val_loss_d_fake += loss_d_fake

                # c. Discriminatorの合計損失と更新
                loss_d = (loss_d_real + loss_d_fake) * 0.5 # 0.5は一般的なスケールファクター
                loss_d.backward()
                optimizer_d.step()
                

            # ---------------------
            # 2. Generator の学習
            # ---------------------
            optimizer.zero_grad() # Generatorのoptimizer (modelのoptimizer)

            output_tensor: Tensor = model(input_tensor) # 再度生成（detachしていない）

            # Calculate L1 Loss
            l1_loss_per_pixel = l1_loss_fn(output_tensor, clean_tensor)
            if l1_loss_per_pixel.dim() == 4:
                l1_loss_per_sample_spatial = torch.mean(l1_loss_per_pixel, dim=1)
            else:
                l1_loss_per_sample_spatial = l1_loss_per_pixel
            masked_l1_loss_per_sample = l1_loss_per_sample_spatial * mask_tensor.squeeze(1)
            current_l1_losses_batch = torch.sum(masked_l1_loss_per_sample, dim=[1,2]) / (torch.sum(mask_tensor.squeeze(1), dim=[1,2]) + 1e-6)
            
            loss_l1 = torch.mean(current_l1_losses_batch)
            total_loss = loss_l1
            current_batch_total_losses: Tensor | None = None
            if hm_temperature:
                current_batch_total_losses = current_l1_losses_batch
            val_l1_loss += loss_l1.item()

            scaled_output_tensor = output_tensor * 2.0 - 1.0
            
            # Calculate LPIPS
            if lp_weight:
                scaled_clean_tensor = clean_tensor * 2.0 - 1.0
                perceptual_losses_batch = lpips_loss_fn(scaled_output_tensor, scaled_clean_tensor)
                loss_perceptual = torch.mean(perceptual_losses_batch)
                total_loss += lp_weight * loss_perceptual
                val_perceptual_loss += loss_perceptual.item()
                if current_batch_total_losses is not None:
                    current_batch_total_losses += lp_weight * perceptual_losses_batch

            # Calculate GAN Loss (Generator)
            # Input for Discriminator: (Input image of Generator, Generated image)
            if discriminator and gan_weight and loss_d is not None:
                scaled_input_tensor = input_tensor * 2.0 - 1.0
                fake_input_for_g = torch.cat([scaled_input_tensor, scaled_output_tensor], dim=1)
                pred_fake_for_g = discriminator(fake_input_for_g)
                loss_g_adversarial = lsgan_loss(pred_fake_for_g, True) # Generatorは「本物と判定してほしい」のでTrue
                total_loss += gan_weight * loss_g_adversarial
                val_gan_loss_g += loss_g_adversarial.item()
                val_discriminator_loss += loss_d.item()
                if current_batch_total_losses is not None:
                    current_batch_total_losses += gan_weight * lsgan_loss(pred_fake_for_g.detach(), True).squeeze()

            val_total_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()
            
            # スケジューラーを更新 (Optimizer.step() の直後が一般的)
            scheduler.step(epochs + batch_idx / len(train_loader)) # type: ignore # 現在のエポック進捗を渡す
            
            # ハードマイニングのための、現在のバッチの各サンプルのTotal Lossを記録
            if current_batch_total_losses is not None:
                # `original_full_dataset_indices` を `train_dataset` 内の相対インデックスに変換
                # ここでは、batch_size 毎にループして変換する
                for i, full_idx in enumerate(original_full_dataset_indices.cpu().numpy()):
                    # `full_idx_to_subset_idx_map` を使って、full_datasetのインデックスからtrain_dataset内のインデックスを取得
                    subset_idx = full_idx_to_subset_idx_map[full_idx]
                    current_epoch_train_losses[subset_idx] = current_batch_total_losses[i].squeeze().item()


        # エポック終了時: サンプリング重みを更新 (ハードマイニングが有効な場合のみ)
        if hm_temperature:
            # 損失を指数関数的に重み付けし、正規化
            # ここでは損失の指数関数的な増加を使うが、他の方法も検討可能
            sample_weights = torch.exp(current_epoch_train_losses / hm_temperature)
            sample_weights = sample_weights / sample_weights.sum() # 正規化して確率分布にする

            # 新しいサンプラーをDataLoaderに設定
            sampler = WeightedRandomSampler(sample_weights, num_samples=num_train_samples, replacement=True)  # type: ignore
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

        # ----------------------------------------
        
        avg_l1_loss = val_l1_loss / len(train_loader)
        avg_perceptual_loss = val_perceptual_loss / len(train_loader)
        avg_gan_loss_g = val_gan_loss_g / len(train_loader)
        avg_discriminator_loss = val_discriminator_loss / len(train_loader)
        avg_loss_d_real = val_loss_d_real / len(train_loader)
        avg_loss_d_fake = val_loss_d_fake / len(train_loader)
        avg_total_loss = val_total_loss / len(train_loader)
        
        _print(f"Epoch [{epoch+1:4d}/{epochs:4d}] Average Train Loss: L1={avg_l1_loss:.4f}, LPIPS={avg_perceptual_loss:.4f}, GAN={avg_gan_loss_g:.4f} (dsc={avg_discriminator_loss:.4f},real={avg_loss_d_real:.4f},fake={avg_loss_d_fake:.4f}), Total={avg_total_loss:.4f}", is_verbose=True)


        # バリデーション
        model.eval()

        val_l1_loss = 0.0
        val_perceptual_loss = 0.0
        val_gan_loss_g = 0.0
        val_discriminator_loss = 0.0
        val_loss_d_real = 0.0
        val_loss_d_fake = 0.0
        val_total_loss = 0.0
        val_ssim_scores = [] 

        with torch.no_grad():
            for batch_idx, (input_tensor, clean_tensor, mask_tensor, _) in enumerate(tqdm(val_loader, disable=not progress, desc=f"Epoch {epoch+1} (Val)")):
                input_tensor: Tensor = input_tensor.to(device)
                clean_tensor: Tensor = clean_tensor.to(device)
                mask_tensor: Tensor = mask_tensor.to(device)

                output_tensor: Tensor = model(input_tensor)

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

                total_loss = loss_l1
                val_l1_loss += loss_l1.item()

                scaled_output_tensor = output_tensor * 2.0 - 1.0

                # Perceptual Loss の計算
                if lp_weight:
                    scaled_clean_tensor = clean_tensor * 2.0 - 1.0
                    perceptual_losses_batch = lpips_loss_fn(scaled_output_tensor, scaled_clean_tensor)
                    loss_perceptual = torch.mean(perceptual_losses_batch)
                    total_loss += lp_weight * loss_perceptual
                    val_perceptual_loss += loss_perceptual.item()
                
                # GAN Loss (Generator) の計算
                # discriminatorへの入力は (input_image, generated_image)
                if discriminator and gan_weight:
                    scaled_input_tensor = input_tensor * 2.0 - 1.0
                    fake_input_for_g = torch.cat([scaled_input_tensor, scaled_output_tensor], dim=1)
                    pred_fake_for_g = discriminator(fake_input_for_g)
                    loss_g_adversarial = lsgan_loss(pred_fake_for_g, True) # Generatorは「本物と判定してほしい」のでTrue
                    total_loss += gan_weight * loss_g_adversarial
                    val_gan_loss_g += loss_g_adversarial.item()

                    # ---------------------
                    # 2. Discriminator の損失計算（評価用、更新はしない）
                    # ---------------------
                    # a. 本物画像に対する損失
                    real_input_d = torch.cat([input_tensor, clean_tensor], dim=1)
                    pred_real = discriminator(real_input_d)
                    loss_d_real = lsgan_loss(pred_real, True)

                    # b. 偽画像に対する損失
                    fake_input_d = torch.cat([input_tensor, output_tensor], dim=1) # output_tensorはすでにdetachされていない
                    pred_fake = discriminator(fake_input_d)
                    loss_d_fake = lsgan_loss(pred_fake, False)
                    
                    val_loss_d_real += loss_d_real.item()
                    val_loss_d_fake += loss_d_fake.item()

                    # c. Discriminatorの合計損失
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

                    val_discriminator_loss += loss_d.item()


                val_total_loss += total_loss.item()
                
                # --- SSIM 計算 ---
                for i in range(input_tensor.shape[0]): 
                    output_img = (output_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    clean_img = (clean_tensor[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    current_ssim = ssim_metric(output_img, clean_img, data_range=255, channel_axis=2, win_size=11)
                    val_ssim_scores.append(current_ssim)

        avg_l1_loss = val_l1_loss / len(val_loader)
        avg_perceptual_loss = val_perceptual_loss / len(val_loader)
        avg_gan_loss_g = val_gan_loss_g / len(val_loader)
        avg_discriminator_loss = val_discriminator_loss / len(val_loader)
        avg_loss_d_real = val_loss_d_real / len(val_loader)
        avg_loss_d_fake = val_loss_d_fake / len(val_loader)
        avg_total_loss = val_total_loss / len(val_loader)
        avg_ssim = np.mean(val_ssim_scores) if val_ssim_scores else 0.0

        # スケジューラーを更新 (バリデーション損失を渡す)
        # scheduler.step(avg_val_total_loss)

        # Save model if Total Loss is improved
        f_best_improved = False
        if avg_total_loss < best_avg_total_loss:
            best_avg_total_loss = avg_total_loss
            f_best_improved = True
        if avg_ssim > best_avg_ssim:
            best_avg_ssim = avg_ssim
            f_best_improved = True

        f_prev_improved = (avg_total_loss < prev_avg_total_loss) or (avg_ssim > prev_avg_ssim)
        
        logtext = f"Epoch [{epoch+1:4d}/{epochs:4d}]"
        logtext += f" {'*' if f_best_improved else '+' if f_prev_improved else '-'}"
        logtext += f" V-Loss: L1={avg_l1_loss:.4f}"
        if lp_weight:
            logtext += f", LPIPS={avg_perceptual_loss:.4f}"
        if gan_weight:
            logtext += f", GAN={avg_gan_loss_g:.4f} (dsc={avg_discriminator_loss:.4f},real={avg_loss_d_real:.4f},fake={avg_loss_d_fake:.4f})"
        logtext += f", Total={avg_total_loss:.4f}, SSIM={avg_ssim:.4f}"
        _print(logtext, flush=True)

        if f_best_improved:
            model_save_path = f"{model_save_dir}/model_ssim_{avg_ssim:.2f}.pth"
            os.makedirs(name=model_save_dir, exist_ok=True)
            model.save(model_save_path, 
                       args=args,
                       epoch=epoch,
                       avg_l1_loss=avg_l1_loss,
                       avg_perceptual_loss=avg_perceptual_loss,
                       avg_total_loss=avg_total_loss,
                       avg_ssim=avg_ssim)

        prev_avg_total_loss = avg_total_loss
        prev_avg_ssim = avg_ssim

    if log_file:
        log_file.close()


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の強化版学習スクリプト。")
    parser.add_argument("dataset_dir", type=str, 
                        help="データセットディレクトリ（動画ごとのファイル名のサブディレクトリを含む）")
    parser.add_argument("model_save_dir", type=str,
                        help="モデルの保存先ディレクトリ")
    parser.add_argument("-inchs", "--in_channels", type=int, default=3,
                        help="Input number of channels")
    parser.add_argument("-outchs", "--out_channels", type=int, default=3,
                        help="Output number of channels")
    parser.add_argument("-imgw", "--image_width", type=int, default=256,
                        help="Input width")
    parser.add_argument("-imgh", "--image_height", type=int, default=192,
                        help="Input height")
    parser.add_argument("-bs", "--batch_size", type=int, default=16,
                        help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="学習エポック数。")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                        help="初期学習率。")
    parser.add_argument("-lp", "--lp_weight", type=float, default=1.0,
                        help="LPIPS weight")
    parser.add_argument("-ts", "--train_split", type=float, default=0.9,
                        help="学習データセットの割合。残りは検証データセット。")
    parser.add_argument("-nw", "--num_workers", type=int, default=(os.cpu_count() or 2) // 2,
                        help="データローダーが使用するワーカースレッド数。")
    parser.add_argument("-ft", "--features", type=str, default="64,128,256",
                        help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("-seb", "--use_se_block", action="store_true",
                        help="Squeeze-and-Excitation (SE) Blockを使用する場合、このフラグを設定。")
    parser.add_argument("-cbam", "--use_cbam", action="store_true",
                        help="Convolutional Block Attention Module (CBAM) を使用する場合、このフラグを設定。")
    parser.add_argument("-gw", "--gan_weight", type=float, default=None,
                        help="GAN weight (None to disable)")
    parser.add_argument("-gft", "--gan_features", type=str, default=None,
                        help="GAN features (required if --gan_weight is not None)")
    parser.add_argument("-glr", "--gan_learning_rate", type=float, default=None,
                        help="GAN learning rate (None to use --learning_rate value)")
    parser.add_argument("-hmt", "--hm_temperature", type=float, default=None,
                        help="Hard-mining temperature parameter (None to disable)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show train and progress message")
    parser.add_argument("-p", "--progress", action="store_true",
                        help="show progress bar")
    parser.add_argument("-log", "--log_dir", type=str, default="log",
                        help="Log directory (None to disable)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="CUDA (GPU) を使用しない場合、このフラグを設定。")
    parser.add_argument("-rseed", "--random_seed", type=int, required=True,
                        help="Random seed")

    args = parser.parse_args()

    set_seed(args.random_seed)
    train_model(args)