import argparse
from PIL import Image
from copy_road_img import create_dataset_structure
import os

parser = argparse.ArgumentParser(description='create a 512x512 xBD building datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--datasets_dir', default='./datasets', type=str, dest='datasets_dir')
parser.add_argument('--save_dir', default='./datasets_512', type=str, dest='save_dir')

args = parser.parse_args()

def save_quarter_crops(path_image, path_label, dir_save, fname_image):
    """
    주어진 원본 이미지와 라벨 이미지에 대해 4분할 crop을 수행하여 
    지정된 디렉토리에 저장하는 함수입니다.
    
    Parameters:
    -----------
    path_image : str
        원본 이미지 파일 경로
    path_label : str
        라벨 이미지 파일 경로
    dir_save : str
        분할된 이미지와 라벨이 저장될 디렉토리 (images와 targets 하위 폴더가 존재해야 함)
    fname_image : str
        분할 저장될 파일의 접두사 (예: "sample"은 "sample_pre_disaster.png"에서 추출)
    """
    # 1. 원본 이미지 열기
    img_input = Image.open(path_image)
    img_label = Image.open(path_label)

    # 2. (width, height) 획득
    width_in, height_in = img_input.size
    width_lb, height_lb = img_label.size

    # 3. 사분면별 crop 영역 정의
    # 왼쪽 상단 (1)
    img_input_1 = img_input.crop((0, 0, width_in // 2, height_in // 2))
    img_label_1 = img_label.crop((0, 0, width_lb // 2, height_lb // 2))

    # 오른쪽 상단 (2)
    img_input_2 = img_input.crop((width_in // 2, 0, width_in, height_in // 2))
    img_label_2 = img_label.crop((width_lb // 2, 0, width_lb, height_lb // 2))

    # 왼쪽 하단 (3)
    img_input_3 = img_input.crop((0, height_in // 2, width_in // 2, height_in))
    img_label_3 = img_label.crop((0, height_lb // 2, width_lb // 2, height_lb))

    # 오른쪽 하단 (4)
    img_input_4 = img_input.crop((width_in // 2, height_in // 2, width_in, height_in))
    img_label_4 = img_label.crop((width_lb // 2, height_lb // 2, width_lb, height_lb))

    # 4. PNG로 저장 (저장 경로: dir_save/images 와 dir_save/targets)
    # 왼쪽 상단 (1)
    img_input_1.save(os.path.join(dir_save, "images", f"{fname_image}_1_sat.png"))
    img_label_1.save(os.path.join(dir_save, "targets", f"{fname_image}_1_mask.png"))
    
    # 오른쪽 상단 (2)
    img_input_2.save(os.path.join(dir_save, "images", f"{fname_image}_2_sat.png"))
    img_label_2.save(os.path.join(dir_save, "targets", f"{fname_image}_2_mask.png"))
    
    # 왼쪽 하단 (3)
    img_input_3.save(os.path.join(dir_save, "images", f"{fname_image}_3_sat.png"))
    img_label_3.save(os.path.join(dir_save, "targets", f"{fname_image}_3_mask.png"))
    
    # 오른쪽 하단 (4)
    img_input_4.save(os.path.join(dir_save, "images", f"{fname_image}_4_sat.png"))
    img_label_4.save(os.path.join(dir_save, "targets", f"{fname_image}_4_mask.png"))
    
    print(f"{fname_image} 파일 crop 완료.")

def process_dataset_crops(datasets_dir, save_dir):
    """
    주어진 datasets 디렉토리 내의 hold, test, train 폴더에 있는
    pre_disaster 이미지를 대상으로 4분할 crop을 수행하여 save_dir에 저장합니다.
    
    Parameters:
    -----------
    datasets_dir : str
        원본 데이터셋 디렉토리 경로 (예: './datasets')
    save_dir : str
        분할된 이미지가 저장될 최종 디렉토리 (예: './datasets_512')
    """
    # 먼저, 저장 디렉토리에 폴더 구조 생성 (외부 모듈의 create_dataset_structure 사용)
    # create_dataset_structure 함수는 fname_folder 매개변수를 받아 저장할 폴더를 생성합니다.
    create_dataset_structure(fname_folder=save_dir)
    
    # hold, test, train 폴더 순회
    for split in ['hold', 'test', 'train']:
        source_dir = os.path.join(datasets_dir, split, 'images')
        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                if filename.endswith("_sat.png"):
                    src_path = os.path.join(source_dir, filename)
                    
                    # 라벨 이미지 경로 생성  
                    # 예를 들어 "_sat.png" → "_mask.png"
                    target_src_path = os.path.join(
                        datasets_dir, split, 'targets',
                        filename.split('_sat.png')[0] + '_mask.png'
                    )
                    
                    # 저장할 경로: save_dir/split (이미지와 타겟 하위 폴더가 존재해야 함)
                    dir_path = os.path.join(save_dir, split)
                    
                    # 파일 접두사 추출 (예: "sample" 추출)
                    fname_image = filename.split('_sat.png')[0]
                    
                    # 사분면별 crop 저장
                    save_quarter_crops(src_path, target_src_path, dir_path, fname_image)
        else:
            print(f"{source_dir} 폴더가 존재하지 않습니다.")

# 사용 예시
if __name__ == "__main__":
    datasets_dir = args.datasets_dir    # 원본 데이터셋 경로
    save_dir = args.save_dir     # 분할 저장될 폴더 경로
    
    process_dataset_crops(datasets_dir, save_dir)