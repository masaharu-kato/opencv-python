import argparse
import os
import glob
import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader

import cv2 # OpenCV for Python for image loading and preprocessing

# CUDAライブラリパスをLD_LIBRARY_PATHに追加
CUDA_LIB_PATH = "/local/cuda-12.8/targets/x86_64-linux/lib/"
os.environ["LD_LIBRARY_PATH"] = CUDA_LIB_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

def getCHW(model_path):
    model = onnx.load(model_path)
    input_tensor = model.graph.input[0]
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append(None)  # 未指定の場合はNone
    # ONNXの入力は[N, C, H, W]が多い
    if len(shape) == 4:
        _, c, h, w = shape
        return c, h, w
    elif len(shape) == 3:
        c, h, w = shape
        return c, h, w
    else:
        raise ValueError(f"Unexpected input shape: {shape}")
    

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('input_onnx')
    argp.add_argument('output_onnx')
    argp.add_argument('calib_dir')
    args = argp.parse_args()

    # キャリブレーションデータの画像のパスリスト（例: 動画から抽出した数フレームのパス）
    # 実際には、このリストに実際の画像ファイルのパスを追加してください
    calib_paths = sorted(glob.glob(os.path.join(args.calib_dir, '*.png')))


    # --- キャリブレーションデータパスの確認 ---
    run_quantization(args.input_onnx, args.output_onnx, calib_paths)


# --- キャリブレーションデータローダーの定義 ---
class CalibDataLoader(CalibrationDataReader):
    def __init__(self, model_path, data_paths):
        self.model_path = model_path
        self.data_paths = data_paths
        self.count = 0
        self.input_name = None # モデルの入力名 (後で設定)

        (self.c, self.h, self.w) = getCHW(self.model_path)

        print(f"DataLoader initialized with {len(data_paths)} images.")

    def get_next(self): # type: ignore
        if self.count >= len(self.data_paths):
            print("Finished providing calibration data.")
            return None

        img_path = self.data_paths[self.count]
        self.count += 1
        
        # 画像の読み込みと前処理 (C++コードの前処理と一致させる)
        try:
            # OpenCVで画像を読み込み
            # BGRA/BGRで読み込まれることが多いので、BGR2RGBに変換
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                return self.get_next() # 次の画像へスキップ

            # チャンネル数がモデルと一致するか確認 (もし一致しない場合はエラーまたは変換)
            if img.shape[2] != self.c:
                # 例: BGRからRGBへの変換 (3チャンネルの場合)
                if self.c == 3 and img.shape[2] == 3: # 既にBGRならcvtColor
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif self.c == 3 and img.shape[2] == 4: # BGRAならBGRA2RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif self.c == 1 and img.shape[2] == 3: # BGRをグレースケールに
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    print(f"Error: Unexpected channel count {img.shape[2]} for image {img_path}. Expected {self.c}. Skipping.")
                    return self.get_next()
            
            # リサイズ
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
            # データ型をfloat32に変換し、0-1に正規化
            img = img.astype(np.float32) / 255.0

            # ONNXモデルがNCHW形式（[N, C, H, W]）を期待する場合、軸を入れ替える
            # OpenCVはHWC (Height, Width, Channel) 形式なので、CHW (Channel, Height, Width) に変換
            if self.c == 3:
                img = img.transpose(2, 0, 1) # HWC -> CHW
            elif self.c == 1:
                img = img[np.newaxis, ...] # HW -> 1HW (for grayscale)

            # バッチ次元を追加（モデルがNCHW [1, C, H, W] を期待する場合）
            input_tensor = img[np.newaxis, ...] # CHW -> NCHW (N=1)

            # モデルの入力名を持つ辞書として返す
            # モデルの入力名は、onnx.load(model_fp32_path).graph.input[0].name で確認できる
            if self.input_name is None:
                # モデルをロードして入力名を取得 (初回のみ)
                model = onnx.load(self.model_path)
                self.input_name = model.graph.input[0].name
                print(f"Detected model input name: {self.input_name}")

            return {self.input_name: input_tensor}

        except Exception as e:
            print(f"Error processing image {img_path}: {e}. Skipping.")
            return self.get_next()
        

# --- メインの量子化処理 ---
def run_quantization(model_path, out_path, calib_paths):
    print(f"Loading FP32 model from: {model_path}")
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model) # モデルの健全性チェック
        print("FP32 model loaded and checked successfully.")
    except Exception as e:
        print(f"Error loading or checking FP32 model: {e}")
        return

    # キャリブレーションデータローダーのインスタンスを作成
    # 実際の画像パスリストを渡す
    data_loader = CalibDataLoader(model_path, calib_paths)

    print(f"Starting static quantization to INT8. Output will be saved to: {out_path}")

    # 静的量子化の実行
    # `per_channel=True` はチャネルごとの量子化を有効にし、精度が高まる傾向があります。
    # `weight_type=QuantType.QInt8` は重みを符号付きINT8にします。
    # `activation_type=QuantType.QUInt8` は活性化を符号なしINT8にします（通常推奨）。
    # `optimize_model=True` は量子化前にモデルを最適化します。
    # `quant_format=QuantFormat.QDQ` はQDQ (Quantize-Dequantize) 形式で量子化ノードを挿入します。
    # TensorRTはQDQ形式を好みます。
    quantize_static(
        model_path,
        out_path,
        data_loader,
        quant_format=QuantFormat.QDQ, # TensorRT向けにQDQ形式推奨
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        # optimize_model=True
    )

    print("Quantization complete!")
    print(f"INT8 model saved to: {out_path}")

    # 量子化されたモデルの健全性チェック
    try:
        model_int8 = onnx.load(out_path)
        onnx.checker.check_model(model_int8)
        print("INT8 model loaded and checked successfully.")
    except Exception as e:
        print(f"Error loading or checking INT8 model: {e}")

if __name__ == "__main__":
    main()
