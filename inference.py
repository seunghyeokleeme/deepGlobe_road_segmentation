import argparse
import os
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from dataset import inferenceDataset
from model import UNet
from utils.checkpoint import load

parser = argparse.ArgumentParser(description='Train the deepGlobe road segmentation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
parser.add_argument('--batch_size', default=4, type=int, dest='batch_size')

parser.add_argument('--data_dir', default='./inference_datasets', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, dest='ckpt_dir')
parser.add_argument('--result_dir', default='./inference_results', type=str, dest='result_dir')

args = parser.parse_args()

#parameters
lr = args.lr
batch_size = args.batch_size

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # mac
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('learning rate: %.4e' % lr)
print('batch size: %d' % batch_size)
print('data dir: %s' % data_dir)
print('ckpt dir: %s' % ckpt_dir)
print('result dir: %s' % result_dir)

# create result dir
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.4094, 0.3824, 0.2875],
                        std=[0.1218, 0.1004, 0.0916])])

inference_data = inferenceDataset(input_dir=data_dir, transform=transform)
inference_loader = DataLoader(inference_data, batch_size, shuffle=False)

net = UNet().to(device)

# Loss function 설정하기
fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_pred = lambda output: (torch.sigmoid(output) > 0.5).float()
fn_denorm = lambda x, mean, std: x * torch.tensor(std, device=x.device).view(1, -1, 1, 1) + torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
fn_acc = lambda pred, label: (pred == label).float().mean()

# Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

net, optim, _ = load(ckpt_dir=ckpt_dir, net=net, optim=optim, device=device)

num_batches = len(inference_loader)

with torch.no_grad():
    net.eval()
    for batch, (input, filenames) in enumerate(inference_loader, 1):
        input = input.to(device)
        logits = net(input)
        pred = fn_pred(logits)

        # 평가 사진 저장
        for i, file_name in enumerate(filenames):
            # 예측 결과 저장 (단일 채널)
            pred_np = pred[i].cpu().squeeze().numpy()  # shape: (높이, 너비)
            pred_np = (pred_np * 255).astype('uint8')
            pred_save_path = os.path.join(result_dir, 'png', file_name)
            Image.fromarray(pred_np).save(pred_save_path)
            print(f"Saved prediction: {pred_save_path}")
    