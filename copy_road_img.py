import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 원본 datasets 폴더 경로 (예시)
datasets_dir = './deepGlobe'  # 실제 경로로 수정
# 대상 폴더 경로 (복사될 폴더의 부모 디렉터리)
dest_dir = '.'  # 현재 디렉터리(또는 원하는 부모 경로)로 지정

def create_dataset_structure(base_path='.', fname_folder='datasets'):
    """
    주어진 base_path 하위에 datasets 폴더 및
    hold, test, train 폴더와 각각의 images, targets 폴더를 생성합니다.
    
    Parameters:
    -----------
    base_path: str
        폴더 구조를 생성할 기준 경로 (예: 현재 디렉터리 '.')
    """
    # 생성할 폴더 구조 정의 ("datasets" 접두사가 포함됨)
    folders = [
        "hold/images",
        "hold/targets",
        "test/images",
        "test/targets",
        "train/images",
        "train/targets",
    ]
    
    for folder in folders:
        target_path = os.path.join(base_path, fname_folder, folder)
        os.makedirs(target_path, exist_ok=True)
        print(f"Created: {target_path}")

# 대상 폴더 내에 "datasets" 폴더가 없으면 생성
if not os.path.exists(os.path.join(dest_dir, "datasets")):
    create_dataset_structure(dest_dir)

def split_save():
    save_data = 'deepGlobe'

    lst_train_images = [f for f in os.listdir(os.path.join(save_data)) if f.endswith('.jpg')]
    lst_train_targets = [f for f in os.listdir(os.path.join(save_data)) if f.endswith('.png')]

    lst_train_images.sort()
    lst_train_targets.sort()

    # 먼저 전체의 80%를 training 세트로, 나머지 20%를 임시 세트로 분할
    train_images, temp_images, train_targets, temp_targets = train_test_split(
        lst_train_images, lst_train_targets, test_size=0.2, random_state=42)

    # 임시 세트를 hold와 test 세트로 각각 50%씩 분할 (전체의 10%씩)
    hold_images, test_images, hold_targets, test_targets = train_test_split(
        temp_images, temp_targets, test_size=0.5, random_state=42)

    print(f"Train set: {len(train_images)} images")
    print(f"Hold set: {len(hold_images)} images")
    print(f"Test set: {len(test_images)} images")

    for filename in train_images:
        save_path = os.path.join('datasets', 'train', 'images', filename)
        img = Image.open(os.path.join(save_data, filename))

        img.save(save_path.split('.jpg')[0]+'.png', 'PNG')

    for filename in train_targets:
        save_path = os.path.join('datasets', 'train', 'targets', filename)
        img = Image.open(os.path.join(save_data, filename)).convert('L')
        gray_np = np.array(img)
        binary_np = (gray_np > 128).astype(np.uint8)
        binary_img = Image.fromarray(binary_np, mode='L')
        binary_img.save(save_path)

    for filename in hold_images:
        save_path = os.path.join('datasets', 'hold', 'images', filename)
        img = Image.open(os.path.join(save_data, filename))

        img.save(save_path.split('.jpg')[0]+'.png', 'PNG')

    for filename in hold_targets:
        save_path = os.path.join('datasets', 'hold', 'targets', filename)
        img = Image.open(os.path.join(save_data, filename)).convert('L')
        gray_np = np.array(img)
        binary_np = (gray_np > 128).astype(np.uint8)
        binary_img = Image.fromarray(binary_np, mode='L')
        binary_img.save(save_path)

    for filename in test_images:
        save_path = os.path.join('datasets', 'test', 'images', filename)
        img = Image.open(os.path.join(save_data, filename))

        img.save(save_path.split('.jpg')[0]+'.png', 'PNG')

    for filename in test_targets:
        save_path = os.path.join('datasets', 'test', 'targets', filename)
        img = Image.open(os.path.join(save_data, filename)).convert('L')
        gray_np = np.array(img)
        binary_np = (gray_np > 128).astype(np.uint8)
        binary_img = Image.fromarray(binary_np, mode='L')
        binary_img.save(save_path)

 # 사용 예시
if __name__ == "__main__":
    split_save()     