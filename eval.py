#!/usr/bin/env python
import os
import argparse
from PIL import Image
import numpy as np
import json

def compute_localization_metrics(result_dir):
    """
    주어진 결과 디렉토리 내의 png 폴더에서 test_mask와 test_pred 파일들을 읽어
    도로 로컬라이제이션(도로 검출) 평가를 위한 다양한 메트릭을 계산합니다.
    
    각 이미지에서 0보다 큰 픽셀을 도로(1)로 간주하여 이진 마스크를 생성한 후,
    전체 픽셀 단위로 TP, FP, FN, TN을 계산하고,
    precision, recall, F1 score, accuracy, IoU를 산출합니다.
    
    Parameters:
        result_dir (str): 결과 디렉토리 경로
        
    Returns:
        metrics (dict): {
            'precision': ...,
            'recall': ...,
            'f1': ...,
            'accuracy': ...,
            'iou': ...,
            'TP': ...,
            'FP': ...,
            'FN': ...,
            'TN': ...
        }
    """
    png_dir = os.path.join(result_dir, 'png')
    
    # 파일 리스트 생성 및 정렬
    lst_images = sorted([f for f in os.listdir(png_dir) if f.startswith('test_input')])
    lst_labels = sorted([f for f in os.listdir(png_dir) if f.startswith('test_mask')])
    lst_preds = sorted([f for f in os.listdir(png_dir) if f.startswith('test_pred')])
    
    print(f"Number of images: {len(lst_images)}")
    print(f"Number of labels: {len(lst_labels)}")
    print(f"Number of preds: {len(lst_preds)}")
    
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    
    for label_file, pred_file in zip(lst_labels, lst_preds):
        # PIL로 이미지 열기 및 grayscale 변환
        label = np.array(Image.open(os.path.join(png_dir, label_file)).convert('L'))
        pred = np.array(Image.open(os.path.join(png_dir, pred_file)).convert('L'))
        
        # 0보다 큰 값은 도로(1)으로, 0은 배경(0)으로 간주
        label_bin = (label > 0).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)
        
        # 각 이미지에 대해 TP, FP, FN, TN 계산
        TP = ((pred_bin == 1) & (label_bin == 1)).sum()
        FN = ((pred_bin == 0) & (label_bin == 1)).sum()
        FP = ((pred_bin == 1) & (label_bin == 0)).sum()
        TN = ((pred_bin == 0) & (label_bin == 0)).sum()
        
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN
    
    # 메트릭 계산 (0 division 방지)
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN) if (total_TP + total_FP + total_FN + total_TN) > 0 else 0
    iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "iou": iou,
        "TP": int(total_TP),
        "FP": int(total_FP),
        "FN": int(total_FN),
        "TN": int(total_TN)
    }
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute localization (road detection) metrics for semantic segmentation results'
    )
    parser.add_argument('--result_dir', type=str, default='./results_v1',
                        help='Path to results directory (default: ./results_v1)')
    parser.add_argument('--out_fp', type=str, default='localization_metrics.json',
                        help='Output JSON file path (default: localization_metrics.json)')
    args = parser.parse_args()
    
    metrics = compute_localization_metrics(args.result_dir)
    
    print("Localization Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"IoU      : {metrics['iou']:.4f}")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}, TN: {metrics['TN']}")
    
    with open(args.out_fp, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {args.out_fp}")