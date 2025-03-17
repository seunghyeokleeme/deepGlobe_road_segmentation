import argparse
import os
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import roadDataset
from model import UNet
from utils.checkpoint import load, save

parser = argparse.ArgumentParser(description='Train the deepGlobe road segmentation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=1e-3, type=float, dest='lr')
parser.add_argument('--batch_size', default=4, type=int, dest='batch_size')
parser.add_argument('--num_epoch', default=100, type=int, dest='num_epoch')

parser.add_argument('--data_dir', default='./datasets', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, dest='ckpt_dir')
parser.add_argument('--log_dir', default='./log', type=str, dest='log_dir')
parser.add_argument('--result_dir', default='./results', type=str, dest='result_dir')

parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')

args = parser.parse_args()

#parameters
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # mac
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('learning rate: %.4e' % lr)
print('batch size: %d' % batch_size)
print('number of epoch: %d' %num_epoch)
print('data dir: %s' % data_dir)
print('ckpt dir: %s' % ckpt_dir)
print('log dir: %s' % log_dir)
print('result dir: %s' % result_dir)
print('mode: %s' % mode)
print('train continue: %s' % train_continue)

# create result dir
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))

if mode == 'train':
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.4094, 0.3824, 0.2875],
                        std=[0.1218, 0.1004, 0.0916])])
    target_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

    train_data = roadDataset(root=os.path.join(data_dir, 'train'), transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)

    val_data = roadDataset(root=os.path.join(data_dir, 'hold'), transform=transform, target_transform=target_transform)
    val_loader = DataLoader(val_data, batch_size, shuffle=False)

else:
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.4094, 0.3824, 0.2875],
                        std=[0.1218, 0.1004, 0.0916])])
    target_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

    test_data = roadDataset(root=os.path.join(data_dir, 'test'), transform=transform, target_transform=target_transform)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

net = UNet().to(device)

# Loss function 설정하기
fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_pred = lambda output: (torch.sigmoid(output) > 0.5).float()
fn_denorm = lambda x, mean, std: x * torch.tensor(std, device=x.device).view(1, -1, 1, 1) + torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
fn_acc = lambda pred, label: (pred == label).float().mean()

# Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'), comment='train')
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'), comment='val')
writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'), comment='test')
    
def train_loop(dataloader, model, fn_loss, optim, epoch, writer):
    num_batches = len(dataloader)
    train_loss, correct = 0.0, 0

    model.train()

    for batch, (input, mask) in enumerate(dataloader, 1):
        input = input.to(device)
        mask = mask.to(device)
        
        logits = model(input)
        loss = fn_loss(logits, mask)

        # backward pass
        loss.backward()
        optim.step()
        optim.zero_grad()

        train_loss += loss.item()
        
        pred = fn_pred(logits)
        acc = fn_acc(pred, mask)
        correct += acc.item()

        print("TRAIN: EPOCH %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f"%
                (epoch, batch, num_batches, loss.item(), acc.item()))

        # input_img = fn_denorm(input, mean=[0.4094, 0.3824, 0.2875], std=[0.1218, 0.1004, 0.0916])

        # # Tensorboard 저장하기
        # writer.add_images("input", img_tensor=input_img, global_step=(num_batches*(epoch -1)+batch))
        # writer.add_images("mask", img_tensor=mask, global_step=(num_batches*(epoch -1)+batch))
        # writer.add_images("predict", img_tensor=pred, global_step=(num_batches*(epoch -1)+batch))
    
    train_loss /= num_batches
    train_accuracy = correct / num_batches
    writer.add_scalar('Loss', train_loss, epoch)
    writer.add_scalar('Accuracy', train_accuracy, epoch)
    
    return train_loss, train_accuracy

def eval_loop(dataloader, model, fn_loss, epoch, writer):
    num_batches = len(dataloader)
    eval_loss, correct = 0.0, 0

    model.eval()

    with torch.no_grad():
        for batch, (input, mask) in enumerate(dataloader, 1):
            input = input.to(device)
            mask = mask.to(device)
            
            logits = model(input)
            loss = fn_loss(logits, mask)

            eval_loss += loss.item()
            
            pred = fn_pred(logits)
            acc = fn_acc(pred, mask)
            correct += acc.item()

            print("VALID: EPOCH %04d | BATCH %04d / %04d | LOSS %.4f | ACC %.4f"%
                    (epoch, batch, num_batches, loss.item(), acc.item()))

            # input_img = fn_denorm(input, mean=[0.4094, 0.3824, 0.2875], std=[0.1218, 0.1004, 0.0916])

            # # Tensorboard 저장하기
            # writer.add_images("input", img_tensor=input_img, global_step=(num_batches*(epoch -1)+batch))
            # writer.add_images("mask", img_tensor=mask, global_step=(num_batches*(epoch -1)+batch))
            # writer.add_images("predict", img_tensor=pred, global_step=(num_batches*(epoch -1)+batch))
        
    eval_loss /= num_batches
    eval_accuracy = correct / num_batches
    writer.add_scalar('Loss', eval_loss, epoch)
    writer.add_scalar('Accuracy', eval_accuracy, epoch)
    
    return eval_loss, eval_accuracy

st_epoch = 0

if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim, device=device)
    
    for epoch in range(st_epoch+1, num_epoch + 1):
        loss, acc = train_loop(dataloader=train_loader, model=net, fn_loss=fn_loss, optim=optim, epoch=epoch, writer=writer_train)
        val_loss, val_acc = eval_loop(dataloader=val_loader, model=net, fn_loss=fn_loss, epoch=epoch, writer=writer_val)

        print(f"Epoch {epoch} summary: Train Loss: {loss:.4f} | Train Accuracy: {100*acc:.1f}% | Val Loss: {val_loss:.4f} | Val Accuracy: {100*val_acc:.1f}%")
    
        if epoch % 5 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
else:
    net, optim, _ = load(ckpt_dir=ckpt_dir, net=net, optim=optim, device=device)

    num_batches = len(test_loader)
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        net.eval()
        for batch, (input, mask) in enumerate(test_loader, 1):
            input = input.to(device)
            mask = mask.to(device)
            
            logits = net(input)
            loss = fn_loss(logits, mask)

            test_loss += loss.item()
            
            pred = fn_pred(logits)
            acc = fn_acc(pred, mask)
            correct += acc.item()

            print("Test: BATCH %04d / %04d | LOSS %.4f | ACC %.4f"%
                    (batch, num_batches, loss.item(), acc.item()))

            input_img = fn_denorm(input, mean=[0.4094, 0.3824, 0.2875], std=[0.1218, 0.1004, 0.0916])
            # Tensorboard 저장하기
            writer_test.add_images("input", img_tensor=input_img, global_step=batch)
            writer_test.add_images("mask", img_tensor=mask, global_step=batch)
            writer_test.add_images("predict", img_tensor=pred, global_step=batch)
        
            # 평가 사진 저장
            for i in range(pred.size(0)):
                # 입력 이미지 저장 (3채널)
                input_np = input_img[i].cpu().permute(1, 2, 0).numpy()  # (채널, 높이, 너비) -> (높이, 너비, 채널)
                input_np = (input_np * 255).astype('uint8')  # [0,1] -> [0,255]
                input_save_path = os.path.join(result_dir, 'png', f"test_input_batch{batch:04d}_img{i:02d}.png")
                Image.fromarray(input_np).save(input_save_path)

                # 마스크 이미지 저장 (단일 채널)
                mask_np = mask[i].cpu().squeeze().numpy()  # shape: (높이, 너비)
                mask_np = (mask_np * 255).astype('uint8')
                mask_save_path = os.path.join(result_dir, 'png', f"test_mask_batch{batch:04d}_img{i:02d}.png")
                Image.fromarray(mask_np).save(mask_save_path)

                # 예측 결과 저장 (단일 채널)
                pred_np = pred[i].cpu().squeeze().numpy()  # shape: (높이, 너비)
                pred_np = (pred_np * 255).astype('uint8')
                pred_save_path = os.path.join(result_dir, 'png', f"test_pred_batch{batch:04d}_img{i:02d}.png")
                Image.fromarray(pred_np).save(pred_save_path)
      
    test_loss /= num_batches
    test_accuracy = correct / num_batches

    print('AVERAGE TEST: LOSS %.4f | ACC %.4f'% (test_loss, 100*test_accuracy))
    writer_test.close()