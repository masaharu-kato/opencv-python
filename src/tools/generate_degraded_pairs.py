import cv2
import numpy as np
import os
import random
import argparse
from tqdm import tqdm # プログレスバー表示用

# --- 劣化関数群 ---

def apply_low_contrast_and_dynamic_range_compression(image, factor=None):
    """
    画像を低コントラスト化し、ダイナミックレンジを圧縮する（ハイライト下げ、シャドー上げ）。
    これにより、全体的な明るさを大きく変えずに「コントラストが低い」状況を再現する。
    factor: None (ランダム), or float (0.5-0.9) - コントラストの度合い
    """
    if factor is None:
        factor = random.uniform(0.2, 0.7) # 0.5 (強い圧縮) から 0.9 (弱い圧縮)
    
    # 画像をfloat32に変換
    img_float = image.astype(np.float32) / 255.0

    # 輝度チャンネルを分離 (BGR -> YCrCb)
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = img_ycrcb[:, :, 0].astype(np.float32) / 255.0 # 輝度チャンネルを0-1に正規化

    # --- ダイナミックレンジ圧縮のためのトーンカーブ的アプローチ ---
    # ハイライトを下げ、シャドーを上げる（S字カーブの逆）
    
    # 圧縮の度合いをランダムに調整
    compress_strength = random.uniform(0.3, 0.7) # 0.3 (弱め) から 0.7 (強め)
    
    # 新しい輝度値を計算
    # y = x + strength * sin(2*pi*x) のようなカーブ
    # より直感的なダイナミックレンジ圧縮：ハイライトを下に、シャドーを上に
    # 0と1に近い値をそれぞれ中央に寄せるイメージ
    # Y' = Y * (1 - compress_strength) + Y_mid * compress_strength
    
    # 具体的な実装例: シグモイド関数を調整したカーブや、単純な線形マッピング
    # ここでは、輝度値を狭い範囲にマッピングし直すことでダイナミックレンジを圧縮します。
    # 例えば、0-1の輝度値を (0.1, 0.9) の範囲に圧縮し、その後0-1に線形伸張
    
    # 入力範囲 (in_min, in_max) を 出力範囲 (out_min, out_max) にマッピング
    in_min = 0.0
    in_max = 1.0
    
    # 出力範囲を決定（中心0.5を保ちつつ、狭くする）
    out_min_offset = (1.0 - factor) / 2.0 # factorが大きいほど、オフセットは小さい
    out_max_offset = (1.0 - factor) / 2.0
    
    out_min = out_min_offset # シャドーが上がる
    out_max = 1.0 - out_max_offset # ハイライトが下がる

    # 線形マッピング: Y' = (Y - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    Y_compressed = (Y - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    Y_compressed = np.clip(Y_compressed, 0, 1) # 0-1の範囲にクリップ

    # 元のYCrCbに戻してBGRに変換
    img_ycrcb[:, :, 0] = (Y_compressed * 255).astype(np.uint8)
    degraded_image = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

    return degraded_image

def apply_highlight_clipping(image, clip_intensity=None):
    """
    画像を全体的に白飛びさせる（ハイライトクリッピング）。
    clip_intensity: None (ランダム), or float (1.1-2.0)
    """
    if clip_intensity is None:
        clip_intensity = random.uniform(1.1, 2.0) # 1.1 (少し) から 2.0 (かなり)
    
    # 画像全体を明るくしてクリッピング
    brightened_image = np.clip(image.astype(np.float32) * clip_intensity, 0, 255).astype(np.uint8)
    return brightened_image

# apply_underexposure は今回の要件に合わないため削除

def apply_defocus_blur(image, kernel_size=None):
    """
    画像をピンボケにする（ガウシアンブラー）。
    kernel_size: None (ランダム), or odd int (3-25)
    """
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_motion_blur(image, kernel_size=None, angle=None):
    """
    画像をモーションブラーにする。
    kernel_size: None (ランダム), or odd int (5-25)
    angle: None (ランダム), or int (0-180)
    """
    if kernel_size is None:
        kernel_size = random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
    if angle is None:
        angle = random.randint(0, 180)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    # 角度に応じた直線カーネルを作成
    # NumPyのix_を使って効率的に値を設定
    if angle == 0 or angle == 180: # 水平
        kernel[center, :] = 1
    elif angle == 90: # 垂直
        kernel[:, center] = 1
    else: # その他の角度
        # ブレの方向をベクトルで表現
        dx = np.cos(np.deg2rad(angle))
        dy = np.sin(np.deg2rad(angle))
        
        # カーネル内の各点について、ブレの線上に乗っているかを判定
        for i in range(kernel_size):
            y_offset = (i - center) * dy
            x_offset = (i - center) * dx
            
            # 中心からの距離に基づいて線上の点を決定
            # より高度なカーネル生成にはcv2.getRotationMatrix2DやBresenham's line algorithmが適
            # ここでは簡易的に、中心から最も近い整数座標に線を引く
            
            # 中心から (dx, dy) 方向に線を引く
            for step in range(-center, center + 1):
                y_coord = int(center + step * dy)
                x_coord = int(center + step * dx)
                if 0 <= y_coord < kernel_size and 0 <= x_coord < kernel_size:
                    kernel[y_coord, x_coord] = 1
    
    kernel = kernel / np.sum(kernel) # 正規化
    
    return cv2.filter2D(image, -1, kernel)

# def add_gaussian_noise(image, sigma=None):
#     """
#     画像にガウシアンノイズを追加する。
#     sigma: None (ランダム), or float (5-30)
#     """
#     if sigma is None:
#         sigma = random.uniform(5.0, 25.0) # 5 (軽度) から 25 (中程度)
    
#     row, col, ch = image.shape
#     mean = 0
#     gauss = np.random.normal(mean, sigma, (row, col, ch))
#     noisy_image = image.astype(np.float32) + gauss
#     return np.clip(noisy_image, 0, 255).astype(np.uint8)

# --- メインスクリプト ---

def generate_degraded_pairs(clean_images_dir, output_degraded_dir, output_clean_dir, num_degradation_per_image=10):
    """
    綺麗な画像ディレクトリから画像を読み込み、様々な劣化を適用してペアを生成する。

    Parameters:
    clean_images_dir (str): 綺麗な画像が保存されているディレクトリのパス。
    output_degraded_dir (str): 生成された劣化画像を保存するディレクトリのパス。
    output_clean_dir (str): 対応する綺麗な画像をコピーして保存するディレクトリのパス。
    num_degradation_per_image (int): 1つの綺麗な画像から生成する劣化パターンの数。
    """

    if not os.path.exists(output_degraded_dir):
        os.makedirs(output_degraded_dir)
        print(f"劣化画像出力ディレクトリを作成しました: {output_degraded_dir}")
    if not os.path.exists(output_clean_dir):
        os.makedirs(output_clean_dir)
        print(f"綺麗な画像出力ディレクトリを作成しました: {output_clean_dir}")

    clean_image_files = [f for f in os.listdir(clean_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not clean_image_files:
        print(f"エラー: {clean_images_dir} に画像ファイルが見つかりません。")
        return

    print(f"\n--- データセット生成開始 ---")
    print(f"処理対象の綺麗な画像数: {len(clean_image_files)}枚")
    print(f"1画像あたりの劣化パターン数: {num_degradation_per_image}種類")
    print(f"予想される劣化画像総数: {len(clean_image_files) * num_degradation_per_image}枚")
    print(f"劣化画像を {output_degraded_dir} に、対応する綺麗な画像を {output_clean_dir} に保存します。\n")

    for i, clean_img_name in enumerate(tqdm(clean_image_files, desc="画像処理中")):
        clean_img_path = os.path.join(clean_images_dir, clean_img_name)
        clean_image = cv2.imread(clean_img_path)

        if clean_image is None:
            print(f"警告: {clean_img_path} を読み込めませんでした。スキップします。")
            continue

        # 綺麗な画像をターゲットディレクトリにコピー（リネームして保存）
        base_name_without_ext = os.path.splitext(clean_img_name)[0]
        target_clean_path = os.path.join(output_clean_dir, f"{base_name_without_ext}.png")
        cv2.imwrite(target_clean_path, clean_image) # PNG形式で統一

        for j in range(num_degradation_per_image):
            degraded_image = clean_image.copy()
            degradation_applied = []

            # 劣化関数のリスト
            # ここでは「低コントラストとダイナミックレンジ圧縮」を主軸に
            # 白飛び、ノイズ、ピンボケ、モーションブラーをランダムに組み合わせる
            
            # --- フェーズ1: コントラスト・明るさの劣化を優先的に適用 ---
            # 常に適用される可能性のある劣化
            core_degradation_candidates = []
            
            # まずは「低コントラスト＆ダイナミックレンジ圧縮」を常に含める
            degraded_image = apply_low_contrast_and_dynamic_range_compression(degraded_image)
            degradation_applied.append("low_contrast_DRC")
            
            # 白飛びを約50%の確率で追加
            if random.random() < 0.5:
                degraded_image = apply_highlight_clipping(degraded_image)
                degradation_applied.append("highlight_clip")
            
            # ノイズを約70%の確率で追加
            # if random.random() < 0.7:
            #     degraded_image = add_gaussian_noise(degraded_image)
            #     degradation_applied.append("noise")

            # --- フェーズ2: ぼけの種類をランダムで追加 ---
            # 「場合によってはさらに若干ピンボケ」を再現
            blur_type = random.choice(["defocus", "motion", "none"]) # ブラーの種類をランダムに選択
            
            if blur_type == "defocus":
                if random.random() < 0.7: # ピンボケは70%の確率で
                    degraded_image = apply_defocus_blur(degraded_image)
                    degradation_applied.append("defocus_blur")
            elif blur_type == "motion":
                if random.random() < 0.4: # モーションブラーは40%の確率で
                    degraded_image = apply_motion_blur(degraded_image)
                    degradation_applied.append("motion_blur")


            # 生成された劣化画像を保存
            degradation_str = "_".join(sorted(list(set(degradation_applied))))
            if not degradation_str: # 何も劣化が適用されなかった場合のフォールバック（理論上は起こらない）
                degradation_str = "no_degradation"
            
            output_filename = f"{base_name_without_ext}_deg_{degradation_str}_{j:02d}.png"
            output_degraded_path = os.path.join(output_degraded_dir, output_filename)
            cv2.imwrite(output_degraded_path, degraded_image)

    print(f"\n--- データセット生成完了 ---")
    print(f"生成された劣化画像の総数: {len(os.listdir(output_degraded_dir))}枚")
    print(f"コピーされた綺麗な画像の総数: {len(os.listdir(output_clean_dir))}枚")


# --- コマンドライン引数パーサー ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="綺麗な画像から劣化画像を生成し、ペアとして保存するスクリプト。")
    parser.add_argument("clean_images_dir", type=str,
                        help="綺麗な画像が保存されているディレクトリのパス。")
    parser.add_argument("--output_degraded_dir", type=str, default="res/degraded_images",
                        help="生成された劣化画像を保存するディレクトリのパス。")
    parser.add_argument("--output_clean_dir", type=str, default="res/clean_targets",
                        help="対応する綺麗な画像をコピーして保存するディレクトリのパス。")
    parser.add_argument("--num_degradation_per_image", type=int, default=10,
                        help="1つの綺麗な画像から生成する劣化パターンの数。")
    
    args = parser.parse_args()
    
    generate_degraded_pairs(args.clean_images_dir, args.output_degraded_dir, 
                             args.output_clean_dir, args.num_degradation_per_image)