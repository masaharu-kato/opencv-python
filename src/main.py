import argparse
import logging
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from os import cpu_count, PathLike
import numpy as np
import lpips
import piqa
import pytorch_optimizer
import torch
import torch.amp.grad_scaler
import torch.amp.autocast_mode
import torch.utils.tensorboard
import torch.nn as nn
from datetime import datetime
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset import Dataset, DatasetOptions
from models.unet import AttentionMethods, ModelOptions, UNet
from utils.option_utils import make_options

PATH_LOGGING = Path("log")
PATH_TENSORBOARD = Path("runs")

@dataclass
class TrainOptions:
    seed: int
    epochs: int
    train_split: float
    learning_rate: float
    batch_size: int
    ga_steps: int # gradient accumulation steps
    lp_weight: float
    hm_temperature: float # Hard mining temperature, 0 for no hard mining

@dataclass
class RuntimeOptions:
    num_workers: int
    verbose: bool
    no_progress: bool

# --- メインの学習関数 ---
def train_model(*,
    model_path: PathLike | None = None,
    model_dir: PathLike | None = None,
    **args
):

    if not model_path:
        model_path = None
        if not model_dir:
            raise ValueError("Either model (model path) or model_dir (model directory to save) must be specified.")
        model_save_dir = Path(model_dir)
    else:
        model_path = Path(model_path)
        model_save_dir = model_path.parent

    model_type_name = model_save_dir.name

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # INFOレベル以上のメッセージを処理
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    PATH_LOGGING.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(PATH_LOGGING / f"{model_type_name}.log")
    file_handler.setLevel(logging.INFO) # ファイルにはINFOレベル以上を書き出す
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout) # 標準出力へ
    console_handler.setLevel(logging.INFO) # コンソールにはINFOレベル以上を書き出す\
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Record uncaught exceptions to the log file
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: \
        logger.exception("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Setup TensorBoard logging
    
    PATH_TENSORBOARD.mkdir(parents=True, exist_ok=True)
    log_dir = PATH_TENSORBOARD / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logging.info(f"TensorBoard log directory: {log_dir}")

    ropts = make_options(RuntimeOptions, {}, args)

    logging.info("#### Starting training script ####")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    logging.info(f"loaded model path: {model_path}")
    logging.info(f"model save directory: {model_save_dir}")
    logging.info(f"args: {args}")
    torch.backends.cudnn.benchmark = True
     
    if not model_path:
        model = UNet(make_options(ModelOptions, {}, args)).to(device)
        cp = {}
    else:
        model, cp = UNet.load(model_path, device)
        
    # Load train options
    opts = make_options(TrainOptions, cp.get('train', None), args)
    
    # Seet random seed if new model
    if model_path is None:
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed) # if using multi-GPU
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        # torch.backends.cudnn.deterministic = True # for reproducibility in CuDNN
        # torch.backends.cudnn.benchmark = False # for reproducibility in CuDNN
    
    if 'optimizer' in cp:
        optimizer = pytorch_optimizer.RAdam(model.parameters())
        optimizer.load_state_dict(cp['optimizer'])
        logging.info('Loaded optimizer state dict.')
    else:
        if model_path is not None:
            logging.warning("No optimizer state found in checkpoint. Using default optimizer settings.")
        optimizer = pytorch_optimizer.RAdam(model.parameters(), lr=opts.learning_rate)
    
    if 'scheduler' in cp:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        scheduler.load_state_dict(cp['scheduler'])
        logging.info('Loaded scheduler state dict. ')
    else:
        if model_path is not None:
            logging.warning("No scheduler state found in checkpoint. Using default scheduler settings.")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    logging.info(f"Optimizer: {optimizer}")
    logging.info(f"Scheduler: {scheduler}")

    # Write graph to TensorBoard if model is new
    if model_path is None:
        dummy_iput = torch.randn(1, 3, model.opts.input_height, model.opts.input_width, device=device) # (B, C, H, W)
        writer.add_graph(model, dummy_iput)

    # Preapre dataset
    full_dataset = Dataset(make_options(DatasetOptions, cp.get('dataset', {}), args), model.opts)

    # データの分割
    train_size = int(opts.train_split * len(full_dataset))
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
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=sampler, num_workers=ropts.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=ropts.num_workers, pin_memory=True)

    # ----------------------------------------
       
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
    
    ssim_metric_piqa = piqa.SSIM(n_channels=3).to(device)

    best_avg_total_loss = float('inf')
    prev_avg_total_loss = float('inf')
    best_avg_ssim = 0
    prev_avg_ssim = 0

    scaler = torch.amp.grad_scaler.GradScaler("cuda")

    init_epoch = cp['epoch'] + 1 if 'epoch' in cp else 0
    
    for epoch in range(init_epoch, opts.epochs):
        model.train()
        train_l1_loss_epoch = 0.0 # 訓練時のL1ロスを記録するための変数 (名前をより明確に)
        train_perceptual_loss_epoch = 0.0 # 訓練時のLPIPSロスを記録するための変数
        train_total_loss_epoch = 0.0 # 訓練時のTotalロスを記録するための変数

        optimizer.zero_grad() 

        current_epoch_train_losses = torch.zeros(num_train_samples, dtype=torch.float32, device='cpu') # CPUに置いておく

        # DataLoaderから original_indices_in_subset を受け取る
        # これは train_dataset (Subset) 内でのインデックス
        # tqdm の desc を調整
        for batch_idx, (input_tensor, clean_tensor, mask_tensor, original_full_dataset_indices) in enumerate(tqdm(train_loader, disable=ropts.no_progress, desc=f"Epoch {epoch+1} (Train)")):
            input_tensor: Tensor = input_tensor.to(device)
            clean_tensor: Tensor = clean_tensor.to(device)
            mask_tensor: Tensor = mask_tensor.to(device)

            with torch.amp.autocast_mode.autocast("cuda"):
                output_tensor: Tensor = model(input_tensor)

                # Calculate L1 Loss
                l1_loss_per_pixel = l1_loss_fn(output_tensor, clean_tensor)
                if l1_loss_per_pixel.dim() == 4:
                    l1_loss_per_sample_spatial = torch.mean(l1_loss_per_pixel, dim=1)
                else:
                    l1_loss_per_sample_spatial = l1_loss_per_pixel
                masked_l1_loss_per_sample = l1_loss_per_sample_spatial * mask_tensor.squeeze(1)
                current_l1_losses_batch = torch.sum(masked_l1_loss_per_sample, dim=[1,2]) / (torch.sum(mask_tensor.squeeze(1), dim=[1,2]) + 1e-6)
                
                loss_l1 = torch.mean(current_l1_losses_batch)
                
                # total_loss を current_loss に変更して、累積ロスと混同しないように
                current_batch_total_loss_sum = loss_l1 # このバッチの総ロス (L1のみの場合)
                current_batch_perceptual_losses_for_hm: Tensor | None = None # ハードマイニング用LPIPS

                scaled_output_tensor = output_tensor * 2.0 - 1.0
                
                # Calculate LPIPS
                scaled_clean_tensor = clean_tensor * 2.0 - 1.0
                perceptual_losses_batch = lpips_loss_fn(scaled_output_tensor, scaled_clean_tensor)
                loss_perceptual = torch.mean(perceptual_losses_batch)
                current_batch_total_loss_sum += opts.lp_weight * loss_perceptual
                current_batch_perceptual_losses_for_hm = opts.lp_weight * perceptual_losses_batch # ハードマイニング用にLPIPS部分も保存

            # 勾配蓄積のための損失のスケーリング
            # current_batch_total_loss_sum は autocast ブロック内で定義されているので、ここではその値を使う
            scaled_loss = current_batch_total_loss_sum / opts.ga_steps
            
            scaler.scale(scaled_loss).backward()
            
            # 勾配蓄積ステップに達した場合、または最後のバッチの場合
            if (batch_idx + 1) % opts.ga_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 勾配をクリア

            # 統計情報の更新 (スケーリング前の損失を加算)
            train_l1_loss_epoch += loss_l1.item() # loss_l1 はスケーリング前の値
            train_perceptual_loss_epoch += loss_perceptual.item() # loss_perceptual はスケーリング前の値
            train_total_loss_epoch += current_batch_total_loss_sum.item() # current_batch_total_loss_sum もスケーリング前の値

            # ハードマイニングのための、現在のバッチの各サンプルのTotal Lossを記録
            if opts.hm_temperature:
                batch_total_losses_for_hm = current_l1_losses_batch # L1部分
                if current_batch_perceptual_losses_for_hm is not None:
                    batch_total_losses_for_hm += current_batch_perceptual_losses_for_hm # LPIPS部分を加算

                for i, full_idx in enumerate(original_full_dataset_indices.cpu().numpy()):
                    subset_idx = full_idx_to_subset_idx_map[full_idx]
                    current_epoch_train_losses[subset_idx] = batch_total_losses_for_hm[i].squeeze().item()


        # エポック終了時: サンプリング重みを更新 (ハードマイニングが有効な場合のみ)
        if opts.hm_temperature:
            sample_weights = torch.exp(current_epoch_train_losses / opts.hm_temperature)
            sample_weights = sample_weights / sample_weights.sum()

            sampler = WeightedRandomSampler(sample_weights, num_samples=num_train_samples, replacement=True) # type: ignore
            train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=sampler, num_workers=ropts.num_workers, pin_memory=True) # train_dataset がどこかで定義されていると仮定

        # 平均ロスの計算
        avg_l1_loss = train_l1_loss_epoch / len(train_loader)
        avg_perceptual_loss = train_perceptual_loss_epoch / len(train_loader)
        avg_total_loss = train_total_loss_epoch / len(train_loader)
        
        logging.info(f"Epoch [{epoch+1:4d}/{opts.epochs:4d}] Average Train Loss: L1={avg_l1_loss:.4f}, LPIPS={avg_perceptual_loss:.4f}, Total={avg_total_loss:.4f}")

        # TensorBoardへのログ記録
        writer.add_scalar('Loss/Train/L1', avg_l1_loss, epoch)
        writer.add_scalar('Loss/Train/LPIPS', avg_perceptual_loss, epoch)
        writer.add_scalar('Loss/Train/Total', avg_total_loss, epoch)

        scheduler.step()

        # バリデーション
        model.eval()

        val_l1_loss = 0.0
        val_perceptual_loss = 0.0
        val_total_loss = 0.0
        val_ssim_scores = [] 

        with torch.no_grad(), torch.amp.autocast_mode.autocast("cuda"):
            for batch_idx, (input_tensor, clean_tensor, mask_tensor, _) in enumerate(tqdm(val_loader, disable=ropts.no_progress, desc=f"Epoch {epoch+1} (Val)")):
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

                # Perceptual Loss の計算
                scaled_output_tensor = output_tensor * 2.0 - 1.0
                scaled_clean_tensor = clean_tensor * 2.0 - 1.0
                perceptual_losses_batch = lpips_loss_fn(scaled_output_tensor, scaled_clean_tensor)
                loss_perceptual = torch.mean(perceptual_losses_batch)
                total_loss += opts.lp_weight * loss_perceptual
                val_perceptual_loss += loss_perceptual.item()

                val_total_loss += total_loss.item()
                
                # --- SSIM 計算 ---
                current_ssim = ssim_metric_piqa(output_tensor, clean_tensor)
                val_ssim_scores.append(current_ssim.item())

                if batch_idx < 4:
                    writer.add_image('Val_Images/Input', input_tensor[0], epoch)
                    writer.add_image('Val_Images/Clean', clean_tensor[0], epoch)
                    writer.add_image('Val_Images/Output', output_tensor[0], epoch)


        avg_l1_loss = val_l1_loss / len(val_loader)
        avg_perceptual_loss = val_perceptual_loss / len(val_loader)
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
        
        logtext = f"Epoch [{epoch+1:4d}/{opts.epochs:4d}]"
        logtext += f" {'*' if f_best_improved else '+' if f_prev_improved else '-'}"
        logtext += f" V-Loss: L1={avg_l1_loss:.4f}"
        logtext += f", LPIPS={avg_perceptual_loss:.4f}"
        logtext += f", Total={avg_total_loss:.4f}, SSIM={avg_ssim:.4f}"
        logging.info(logtext)

        writer.add_scalar('Loss/Val/L1', avg_l1_loss, epoch)
        writer.add_scalar('Loss/Val/LPIPS', avg_perceptual_loss, epoch)
        writer.add_scalar('Loss/Val/Total', avg_total_loss, epoch)
        writer.add_scalar('Metrics/Val_SSIM', avg_ssim, epoch)

        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        if f_best_improved:
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_save_dir / f"model_loss_{avg_total_loss:.2f}.pth"), 
                # in_size, features, attention_methods are saved in model.save()
                train=opts,
                dataset=full_dataset.opts,
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                avg_l1_loss=avg_l1_loss,
                avg_perceptual_loss=avg_perceptual_loss,
                avg_total_loss=avg_total_loss,
                avg_ssim=avg_ssim)

        prev_avg_total_loss = avg_total_loss
        prev_avg_ssim = avg_ssim

        writer.flush()

    writer.close()


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の強化版学習スクリプト。")
    parser.add_argument("-m" ,"--model_path", type=str, help="Model file path (.pth) to load (continue training)")
    parser.add_argument("-md" ,"--model_dir", type=str, help="Model save directory (new model will be saved here)")
    parser.add_argument("-d", "--dataset_dir", type=str, help="データセットディレクトリ（動画ごとのファイル名のサブディレクトリを含む）")
    parser.add_argument("-imgw", "--input_width", type=int, help="Input image width")
    parser.add_argument("-imgh", "--input_height", type=int, help="Input image height")
    parser.add_argument("-bs", "--batch_size", type=int, help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("-gas", "--ga_steps", type=int, help="gradient_accumulation_steps")
    parser.add_argument("-e", "--epochs", type=int, help="学習エポック数。")
    parser.add_argument("-lr", "--learning_rate", type=float, help="初期学習率。")
    parser.add_argument("-lp", "--lp_weight", type=float, help="LPIPS weight")
    parser.add_argument("-ts", "--train_split", type=float, help="学習データセットの割合。残りは検証データセット。")
    parser.add_argument("-nw", "--num_workers", type=int, default=((cpu_count() or 2) - 1), help="データローダーが使用するワーカースレッド数。")
    parser.add_argument("-ft", "--features", type=str, help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("-at", "--attention_method", choices=list(AttentionMethods.__args__), help="Attention methods to use")
    parser.add_argument("-hmt", "--hm_temperature", type=float, help="Hard-mining temperature parameter (No hard-mining if 0).")
    parser.add_argument("-v", "--verbose", action="store_true", help="show train and progress message")
    parser.add_argument("-nop", "--no_progress", action="store_true", help="hide progress bar")
    parser.add_argument("-seed", "--seed", type=int, help="Random seed")

    args = parser.parse_args()

    train_model(**args.__dict__)
