import os
import random
import torchvision
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from DM import *
from utils import prepare_dataloaders


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


seed = 1949
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
# dataloader
batch_size = 64
train_dataloader, test_dataloader = prepare_dataloaders(batch_size=batch_size)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

fixed_eeg_train, fixed_img_train, _, _ = next(iter(train_dataloader))
fixed_eeg_train = Variable(fixed_eeg_train.type(FloatTensor)).to(device)
fixed_img_train = Variable(fixed_img_train.type(FloatTensor)).to(device)

fixed_eeg_test, fixed_img_test, _, _ = next(iter(test_dataloader))
fixed_eeg_test = Variable(fixed_eeg_test.type(FloatTensor)).to(device)
fixed_img_test = Variable(fixed_img_test.type(FloatTensor)).to(device)

fixed_img = torch.cat((fixed_img_train[:32], fixed_img_test[:32]), dim=0)
fixed_eeg = torch.cat((fixed_eeg_train[:32], fixed_eeg_test[:32]), dim=0)

img_save_path = "image_save"
os.makedirs(img_save_path, exist_ok=True)
model_save_path = "model_save"
os.makedirs(model_save_path, exist_ok=True)

torchvision.utils.save_image(fixed_img.data, os.path.join(img_save_path, 'fixed.png'), nrow=8, normalize=True)

model = Unet().to(device)
diffusion = GaussianDiffusion(model, timesteps=50, c_l=1, loss_type='l1').to(device)
scaler = GradScaler()

optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

# ----------------------------------------------------------------------------
for epoch in range(1, 801):
    run_loss = 0
    diffusion.train()
    for j, (eeg, img, _, _) in enumerate(train_dataloader):
        # input
        eeg = Variable(eeg.type(FloatTensor)).to(device)
        real_imgs = Variable(img.type(FloatTensor)).to(device)

        with autocast():
            loss = diffusion(x=real_imgs, label=eeg)

        run_loss += loss.item()

        scaler.scale(loss / 2).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    print("epoch: {}    loss: {:.4f}".format(epoch, run_loss))
    if epoch % 20 == 0:
        sampled_images = diffusion.sample(batch_size=64, label=fixed_eeg)
        torchvision.utils.save_image(sampled_images.data, os.path.join(img_save_path, 'epoch_%04d.png' % epoch), nrow=8, normalize=True)
    if epoch % 200 == 0:
        torch.save(diffusion, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))
