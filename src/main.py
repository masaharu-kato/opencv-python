import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pytorch_optimizer
import torch
import torch.amp.grad_scaler
import torch.amp.autocast_mode
import torch.utils.tensorboard
from datetime import datetime
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from datasets.dataset import DatasetOptions, ImagePairDataset
from datasets.path_pair_groups import PathPairGroups
from losses.loss import LPIPS_MODEL, LPIPS, MaskedL1, SSIM
from models.unet import AttentionMethods, ModelOptions, UNet
from utils.option_utils import make_dataclass_from_args, make_dataclass_from_cp_args

PATH_LOGGING = Path("log")
PATH_TENSORBOARD = Path("runs")

@dataclass
class TrainOptions:
    seed: int
    epochs: int
    train_split: float
    active_train_split: float
    learning_rate: float
    batch_size: int
    ga_steps: int # gradient accumulation steps
    lp_weight: float
    lp_model: LPIPS_MODEL 

def train_model(*,
    model_path: Path | str | None = None,
    model_dir: Path | str | None = None,
    dataset_dir: Path | str | None = None,
    num_workers: int | None = None,
    verbose: bool = False,
    no_progress: bool = False,
    reset_optimizer: bool = False,
    reset_scheduler: bool = False,
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

    logging.info("#### Starting training script ####")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    logging.info(f"loaded model path: {model_path}")
    logging.info(f"model save directory: {model_save_dir}")
    logging.info(f"args: {args}")
    torch.backends.cudnn.benchmark = True

    num_workers = num_workers if num_workers is not None else (os.cpu_count() or 2) - 1


    # --------------------------------------------------
    #   New model configuration
    # --------------------------------------------------
    if model_path is None:
     
        model = UNet(make_dataclass_from_cp_args(ModelOptions, {}, args)).to(device)
        cp = {}
    
        # Load train options
        opts = make_dataclass_from_args(TrainOptions, args)

        # Seet random seed if new model
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed) # if using multi-GPU
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        logging.info(f"Set random seed to {opts.seed}.")
        # torch.backends.cudnn.deterministic = True # for reproducibility in CuDNN
        # torch.backends.cudnn.benchmark = False # for reproducibility in CuDNN

        # Write model graph to TensorBoard
        dummy_input = torch.randn(1, 3, model.opts.input_height, model.opts.input_width, device=device) # (B, C, H, W)
        writer.add_graph(model, dummy_input)

        # Preapre dataset
        dataset_opts = make_dataclass_from_args(DatasetOptions,  args)
        if dataset_dir is None:
            logging.error("No dataset files specified.")
            return
        
        full_dataset_paths = PathPairGroups.from_dir(Path(dataset_dir))
        full_train_paths, val_paths = full_dataset_paths.random_split(opts.train_split)

    # --------------------------------------------------
    #   Load model from checkpoint
    # --------------------------------------------------
    else:
        model, cp = UNet.load(model_path, device, args)
        opts = make_dataclass_from_cp_args(TrainOptions, cp.get('train', None), args)

        if 'dataset' not in cp or any(key not in cp['dataset'] for key in ['full_train_paths', 'val_paths']):
            logging.error("Checkpoint does not contain dataset information.")
            return
        
        dataset_opts = make_dataclass_from_args(DatasetOptions,  args)
        
        full_train_paths = PathPairGroups.from_dump(cp['dataset']['full_train_paths'])
        val_paths = PathPairGroups.from_dump(cp['dataset']['val_paths'])
    
    train_paths, _ = full_train_paths.split(opts.active_train_split)
    train_dataset = ImagePairDataset(train_paths, dataset_opts, model.opts)
    val_dataset = ImagePairDataset(val_paths, dataset_opts, model.opts)

    optimizer = pytorch_optimizer.RAdam(model.parameters(), lr=opts.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    if model_path is not None:
        if not reset_optimizer:
            if 'optimizer' not in cp:
                logging.error("Checkpoint does not contain optimizer state.")
                return
            optimizer.load_state_dict(cp['optimizer'])
            logging.info('Loaded optimizer state dict.')

        if not reset_scheduler:
            if 'scheduler' not in cp:
                logging.error("Checkpoint does not contain scheduler state.")
                return
            scheduler.load_state_dict(cp['scheduler'])
            logging.info('Loaded scheduler state dict. ')

    logging.info(f"Optimizer: {optimizer}")
    logging.info(f"Scheduler: {scheduler}")

    num_train_samples = len(train_dataset)
    sample_weights = torch.ones(num_train_samples, dtype=torch.float32)

    sampler = WeightedRandomSampler(sample_weights, # type: ignore
                                    num_samples=num_train_samples, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ----------------------------------------
    calc_l1 = MaskedL1()
    calc_lpips = LPIPS(opts.lp_model)
    calc_ssim = SSIM()

    best_avg_total_loss = float('inf')
    prev_avg_total_loss = float('inf')
    prev_avg_ssim = 0
    best_avg_ssim = 0

    scaler = torch.amp.grad_scaler.GradScaler("cuda")

    init_epoch = cp['epoch'] + 1 if 'epoch' in cp else 0
    

    for epoch in range(init_epoch, opts.epochs):


        # --------------------------------------------------
        #   Train model
        # --------------------------------------------------
        model.train()

        l1_losses: list[float] = []
        lpips_losses: list[float] = []
        total_losses: list[float] = []

        optimizer.zero_grad() 

        for batch_idx, (input_tensor, clean_tensor, mask_tensor, indexes) in enumerate(tqdm(train_loader, disable=no_progress, desc=f"Epoch {epoch+1} (Train)")):
            input_tensor: Tensor = input_tensor.to(device)
            clean_tensor: Tensor = clean_tensor.to(device)
            mask_tensor: Tensor = mask_tensor.to(device)

            with torch.amp.autocast_mode.autocast("cuda"):
                # Apply the model to the input tensor
                output_tensor: Tensor = model(input_tensor)

                # Calculate loss
                l1_loss = calc_l1(output_tensor, clean_tensor, mask_tensor)
                l1_losses.append(l1_loss.item())
                total_loss = l1_loss

                if opts.lp_weight:
                    lpips_loss = calc_lpips(output_tensor, clean_tensor)
                    lpips_losses.append(lpips_loss.item())
                    total_loss += opts.lp_weight * lpips_loss

                total_losses.append(total_loss.item())

            # Scale the loss for gradient accumulation
            scaled_loss = total_loss / opts.ga_steps
            scaler.scale(scaled_loss).backward()
            
            # Reached gradient accumulation step or last batch
            if (batch_idx + 1) % opts.ga_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Clear gradients after step

        avg_l1_loss = np.mean(l1_losses)
        avg_lpips_loss = np.mean(lpips_losses)
        avg_total_loss = np.mean(total_losses)
        
        logging.info(f"Epoch [{epoch+1:4d}/{opts.epochs:4d}] Average Train Loss: L1={avg_l1_loss:.4f}, LPIPS={avg_lpips_loss:.4f}, Total={avg_total_loss:.4f}")

        writer.add_scalar('Loss/Train/L1', avg_l1_loss, epoch)
        writer.add_scalar('Loss/Train/LPIPS', avg_lpips_loss, epoch)
        writer.add_scalar('Loss/Train/Total', avg_total_loss, epoch)

        scheduler.step()


        # --------------------------------------------------
        #   Validation
        # --------------------------------------------------
        model.eval()

        l1_losses: list[float] = []
        lpips_losses: list[float] = []
        total_losses: list[float] = []
        ssims: list[float] = []

        with torch.no_grad():
            for batch_idx, (input_tensor, clean_tensor, mask_tensor, _) in enumerate(tqdm(val_loader, disable=no_progress, desc=f"Epoch {epoch+1} (Val)")):
                input_tensor: Tensor = input_tensor.to(device)
                clean_tensor: Tensor = clean_tensor.to(device)
                mask_tensor: Tensor = mask_tensor.to(device)

                with torch.amp.autocast_mode.autocast("cuda"):
                    # Apply the model to the input tensor
                    output_tensor: Tensor = model(input_tensor)

                    # Calculate loss
                    l1_loss = calc_l1(output_tensor, clean_tensor, mask_tensor)
                    l1_losses.append(l1_loss.item())
                    total_loss = l1_loss

                    if opts.lp_weight:
                        lpips_loss = calc_lpips(output_tensor, clean_tensor)
                        lpips_losses.append(lpips_loss.item())
                        total_loss += opts.lp_weight * lpips_loss

                    total_losses.append(total_loss.item())

                    ssim = calc_ssim(output_tensor, clean_tensor)
                    ssims.append(ssim.item())

                # if batch_idx < 4:
                #     writer.add_image('Val_Images/Input', input_tensor[0], epoch)
                #     writer.add_image('Val_Images/Clean', clean_tensor[0], epoch)
                #     writer.add_image('Val_Images/Output', output_tensor[0], epoch)

        avg_l1_loss = np.mean(l1_losses)
        avg_lpips_loss = np.mean(lpips_losses)
        avg_total_loss = np.mean(total_losses)
        avg_ssim = np.mean(ssims)

        # スケジューラーを更新 (バリデーション損失を渡す)
        # scheduler.step(avg_val_total_loss)

        # Save model if Total Loss is improved
        f_best_improved = False
        if not np.isnan(avg_total_loss) and avg_total_loss < best_avg_total_loss:
            best_avg_total_loss = avg_total_loss
            f_best_improved = True
        if not np.isnan(avg_ssim) and avg_ssim > best_avg_ssim:
            best_avg_ssim = avg_ssim
            f_best_improved = True

        f_prev_improved = (avg_total_loss < prev_avg_total_loss) or (avg_ssim > prev_avg_ssim)
        
        logtext = f"Epoch [{epoch+1:4d}/{opts.epochs:4d}]"
        logtext += f" {'*' if f_best_improved else '+' if f_prev_improved else '-'}"
        logtext += f" V-Loss: L1={avg_l1_loss:.4f}"
        logtext += f", LPIPS={avg_lpips_loss:.4f}"
        logtext += f", Total={avg_total_loss:.4f}, SSIM={avg_ssim:.4f}"
        logging.info(logtext)

        writer.add_scalar('Loss/Val/L1', avg_l1_loss, epoch)
        writer.add_scalar('Loss/Val/LPIPS', avg_lpips_loss, epoch)
        writer.add_scalar('Loss/Val/Total', avg_total_loss, epoch)
        writer.add_scalar('Metrics/Val_SSIM', avg_ssim, epoch)

        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        if f_best_improved:
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_save_dir / f"model_loss_{avg_total_loss:.2f}.pth"), 
                # in_size, features, attention_methods are saved in model.save()
                train=asdict(opts),
                dataset={
                    'opts': asdict(dataset_opts),
                    'full_train_paths': full_train_paths.dump(),
                    'val_paths': val_paths.dump(),
                },
                epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                avg_l1_loss=avg_l1_loss,
                avg_lpips_loss=avg_lpips_loss,
                avg_total_loss=avg_total_loss,
                avg_ssim=avg_ssim)

        prev_avg_total_loss = avg_total_loss
        prev_avg_ssim = avg_ssim

        writer.flush()

    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像画質向上モデル (UNet) の強化版学習スクリプト。")
    parser.add_argument("-m" ,"--model_path", type=str, help="Model file path (.pth) to load (continue training)")
    parser.add_argument("-md" ,"--model_dir", type=str, help="Model save directory (new model will be saved here)")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Dataset directory containing image pairs (good and bad images).")
    parser.add_argument("-imgw", "--input_width", type=int, help="Input image width")
    parser.add_argument("-imgh", "--input_height", type=int, help="Input image height")
    parser.add_argument("-bs", "--batch_size", type=int, help="学習バッチサイズ。GPUメモリに合わせて調整。")
    parser.add_argument("-gas", "--ga_steps", type=int, help="gradient_accumulation_steps")
    parser.add_argument("-e", "--epochs", type=int, help="学習エポック数。")
    parser.add_argument("-lr", "--learning_rate", type=float, help="初期学習率。")
    parser.add_argument("-lp", "--lp_weight", type=float, help="LPIPS weight")
    parser.add_argument("-lpm", "--lp_model", type=float, choices=list(LPIPS_MODEL.__args__), help="LPIPS Model")
    parser.add_argument("-ats", "--active_train_split", type=float, help="Radio of training data to use for training (0.0-1.0) in the training dataset.")
    parser.add_argument("-ts", "--train_split", type=float, help="Radio of training data to use for training (0.0-1.0). The rest will be used for validation.")
    parser.add_argument("-nw", "--num_workers", type=int, help="データローダーが使用するワーカースレッド数。")
    parser.add_argument("-ft", "--features", type=str, help="UNetの各ステージのチャネル数をカンマ区切りで指定 (例: '64,128,256,512')。")
    parser.add_argument("-at", "--attention_method", choices=list(AttentionMethods.__args__), help="Attention methods to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="show train and progress message")
    parser.add_argument("-nop", "--no_progress", action="store_true", help="hide progress bar")
    parser.add_argument("-ro", "--reset_optimizer", action="store_true", help="Reset optimizer state (ignore checkpoint)")
    parser.add_argument("-rs", "--reset_scheduler", action="store_true", help="Reset scheduler state (ignore checkpoint)")
    parser.add_argument("-seed", "--seed", type=int, help="Random seed")

    args = parser.parse_args()

    train_model(**args.__dict__)
