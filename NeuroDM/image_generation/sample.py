import os
import random
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from DM import *


"""
generate sampling images
"""

class EEGDataset(Dataset):
    def __init__(self, path):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data = np.load(path, allow_pickle=True)

    def __getitem__(self, index):
        item = self.data[index]['eeg'], \
               self.transform(Image.fromarray(self.data[index]['image'].numpy())), \
               self.data[index]['label'], \

        return item

    def __len__(self):
        return len(self.data)


seeds = random.sample(range(10000), 1)
seed = seeds[0]
print("seed =", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_path = 'model_save/XXX.pth'
diffusion = torch.load(model_path).eval().to(device)
batch_size = 64
eeg_path = "../data/feature/test.npy"
dataset = EEGDataset(eeg_path)
dataloader = DataLoader(dataset, batch_size=batch_size)
FloatTensor = torch.cuda.FloatTensor

img_save_path = "sample"
os.makedirs(img_save_path, exist_ok=True)
for i in range(40):
    os.makedirs(os.path.join(img_save_path, 'class_%02d' % i), exist_ok=True)

count = [0] * 40


for i, (eeg, image, label) in enumerate(dataloader):
    eeg = Variable(eeg.type(FloatTensor)).to(device)
    image = Variable(image.type(FloatTensor)).to(device)
    sampled_images = diffusion.sample(batch_size=len(eeg), label=eeg)
    for j in range(len(eeg)):
        pair = torch.cat((image[j:j+1], sampled_images[j:j+1]), dim=0)
        count[label[j]] += 1
        torchvision.utils.save_image(pair.data, os.path.join(img_save_path, 'class_%02d' % label[j].data, '%04d.png' % count[label[j]]), nrow=2, normalize=True)

