import os
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from dataset import roadDataset

batch_size = 12
# 데이터셋 경로와 변환 정의 (이미지를 Tensor로 변환)
data_dir = './datasets_512'  # 예: './data'
transform = v2.Compose([
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    # 이미지를 [0,1] 범위의 Tensor로 변환
])
target_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

# 예: ImageFolder를 사용한 데이터셋 (폴더 구조가 클래스별로 되어 있음)
dataset = roadDataset(root=os.path.join(data_dir, 'train'), transform=transform, target_transform=target_transform)
loader = DataLoader(dataset, batch_size, shuffle=False)

# 채널별 합계와 제곱합 초기화
mean = torch.zeros(3)
std = torch.zeros(3)
nb_samples = 0

for data, _ in loader:
    # data shape: [batch_size, channels, height, width]
    batch_samples = data.size(0)
    nb_samples += batch_samples

    # [batch_size, channels, height*width]로 변환 후, 채널별 평균과 표준편차 계산
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)

mean /= nb_samples
std /= nb_samples

print("Mean: ", mean)
print("Std: ", std)
