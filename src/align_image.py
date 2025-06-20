import sys
import cv2
import numpy as np

def align_images_with_optical_flow(image_a_path, image_g_path):
    # 画像の読み込み
    img_a_color = cv2.imread(image_a_path)
    img_g_color = cv2.imread(image_g_path)

    if img_a_color is None or img_g_color is None:
        print(f"Error: Could not read images {image_a_path} or {image_g_path}")
        return None
    
    # アルファチャンネルを追加してRGBAに変換
    # 初期値として、全て不透明(255)のアルファチャンネルを作成
    alpha_channel_g = np.full(img_g_color.shape[:2], 255, dtype=np.uint8)
    img_g_rgba = cv2.merge([img_g_color[:,:,0], img_g_color[:,:,1], img_g_color[:,:,2], alpha_channel_g])


    # オプティカルフローはグレースケール画像で計算するのが一般的
    img_a_gray = cv2.cvtColor(img_a_color, cv2.COLOR_BGR2GRAY)
    img_g_gray = cv2.cvtColor(img_g_color, cv2.COLOR_BGR2GRAY)

    # Farneback法でオプティカルフローを計算
    # flow は (高さ, 幅, 2) のNumpy配列で、各ピクセルの (dx, dy) 変位ベクトルを表す
    # prev: 最初の画像 (img_a_gray)
    # next: 2番目の画像 (img_g_gray)
    # pyr_scale: 各画像ピラミッドレベルでスケールを減らす比率 (0.5は半分)
    # levels: 画像ピラミッドのレベル数
    # winsize: 各ピクセルが考慮される平均ウィンドウサイズ
    # iterations: 各ピラミッドレベルでの反復回数
    # poly_n: 多項式展開の近似サイズ (通常5または7)
    # poly_sigma: ガウシアンの標準偏差 (poly_n=5なら1.1、poly_n=7なら1.5)
    # flags: 0, cv2.OPTFLOW_FARNEBACK_GAUSSIAN など
    flow = cv2.calcOpticalFlowFarneback(prev=img_a_gray, 
                                        next=img_g_gray, 
                                        flow=None, 
                                        pyr_scale=0.5, 
                                        levels=3, 
                                        winsize=15, 
                                        iterations=3, 
                                        poly_n=5, 
                                        poly_sigma=1.1, 
                                        flags=0)

    # 変位ベクトル場 (flow) からワープ用のマッピング座標を生成
    # meshgrid で画像の各ピクセル座標を生成
    h, w = img_a_gray.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # 各ピクセルの新しい位置を計算: 元の座標 + 推定された変位 (flow)
    # flow[:,:,0] はx方向の変位 (dx)、flow[:,:,1] はy方向の変位 (dy)
    map_x = (x_coords + flow[:,:,0]).astype(np.float32)
    map_y = (y_coords + flow[:,:,1]).astype(np.float32)

    # cv2.remap を使って画像をワープ (img_g_color を img_a_color にアライン)
    # map_x, map_y は、それぞれ出力画像の各ピクセルが、入力画像のどこから値を取ってくるかを示す座標
    # WARP_INVERSE_MAP は、flow が "destination_pixel = source_pixel + flow" ではなく
    # "source_pixel = destination_pixel + flow" (逆方向) を表す場合に使うオプションですが、
    # Farnebackのフローは通常、forward flow (prev -> next) なので、ここでは使いません。
    # map_x と map_y が「出力ピクセル (x',y') に対応する入力ピクセル (x,y)」を直接示しているため、remapの通常の使い方はこれでOKです。
    aligned_g_color = cv2.remap(src=img_g_rgba, 
                                map1=map_x, 
                                map2=map_y, 
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, # 境界外の色
                                borderValue=(0, 0, 0, 0)) # (B, G, R, A) = (0, 0, 0, 0)

    return aligned_g_color

def main():
    input_degraded_image_path = sys.argv[1]
    input_ground_truth_image_path = sys.argv[2]
    output_path = sys.argv[3]
    output_aligned_g = align_images_with_optical_flow(input_degraded_image_path, input_ground_truth_image_path)
    if output_aligned_g is not None:
        cv2.imwrite(output_path, output_aligned_g)


if __name__ == '__main__':
    main()
