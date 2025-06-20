import cv2
import numpy as np
import sys

def align_images(image_a_path, image_g_path):
    # 画像の読み込み (BGR形式で読み込まれる)
    img_a = cv2.imread(image_a_path)
    img_g = cv2.imread(image_g_path)

    if img_a is None or img_g is None:
        print(f"Error: Could not read images {image_a_path} or {image_g_path}")
        return None

    # グレースケール変換 (特徴点検出のため)
    img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_g_gray = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)

    # 1. 特徴点検出と記述
    # ORB (Oriented FAST and Rotated BRIEF) は、SIFT/SURFに比べて高速で、特許もフリー
    orb = cv2.ORB_create(nfeatures=5000) # 検出する特徴点の最大数を増やすと良いかも

    kp_a, des_a = orb.detectAndCompute(img_a_gray, None) # キーポイントと記述子
    kp_g, des_g = orb.detectAndCompute(img_g_gray, None)

    if des_a is None or des_g is None:
        print(f"Error: Could not find enough keypoints in {image_a_path} or {image_g_path}")
        return None

    # 2. 特徴点マッチング
    # Brute-Force Matcher を使用 (NORM_HAMMING は ORB/BRIEF記述子向け)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # crossCheck=Trueで互いに最適なマッチのみ

    matches = bf.match(des_a, des_g)

    # マッチング結果を距離でソート (良いマッチから順に)
    matches = sorted(matches, key=lambda x: x.distance)

    # 十分な数の良いマッチがあるか確認
    # アフィン変換には最低3点が必要ですが、RANSACのためにはもっと多い方が良い
    MIN_MATCH_COUNT = 10 # 経験的に調整が必要

    if len(matches) > MIN_MATCH_COUNT:
        # 対応点の座標を抽出
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_g[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 3. 外れ値の除去と 4. 変換行列の推定
        # cv2.estimateAffine2D は RANSAC を内部で実行
        # estimateAffine2D は2x3行列 M を返す
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0) 
        
        if M is None:
            print(f"Warning: Could not estimate affine transformation for {image_a_path} and {image_g_path}")
            return None

        # 5. 画像のワープ (正解画像を基準画像に合わせて変形)
        # 基準画像のサイズ (height, width)
        h, w = img_a_gray.shape
        aligned_g = cv2.warpAffine(img_g, M, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) # WARP_INVERSE_MAPでMの逆変換を適用

        return aligned_g
    else:
        print(f"Not enough matches are found for {image_a_path} and {image_g_path} - {len(matches)}/{MIN_MATCH_COUNT}")
        return None

def main():
    input_degraded_image_path = sys.argv[1]
    input_ground_truth_image_path = sys.argv[2]
    output_path = sys.argv[3]
    output_aligned_g = align_images(input_degraded_image_path, input_ground_truth_image_path)
    if output_aligned_g is not None:
        cv2.imwrite(output_path, output_aligned_g)


if __name__ == '__main__':
    main()
