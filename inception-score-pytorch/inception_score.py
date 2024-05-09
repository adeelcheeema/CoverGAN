import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as TF
from torchvision.models.inception import inception_v3
from PIL import Image
import numpy as np
from scipy.stats import entropy
import pathlib



class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path)
        # cover_path = 'E:\AdeelCoverGAN\Image Generation\scene_generation\outputs\cover_without_name/'+path.stem+'.jpg'
        # if img.size[0]==512:
        #     imagemain = np.asarray(img)
        #     image2 = imagemain[:, 256:, :]
        #     image = imagemain[:, :256, :]
        #
        #     for ii in range(0, 255):
        #         for jj in range(0, 255):
        #             if ((np.sum(image2[ii][jj]) / 3) > 6):
        #                 image[ii][jj] = image2[ii][jj]
        #     img_j = Image.fromarray(np.asarray(image))
        #     img_j.save(cover_path)

        img = self.transforms(img)
        return img

def inception_score(path, cuda=False, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
   
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor


    dataset = ImagePathDataset(files,transforms=TF.Compose([
                                 TF.Resize(256),
                                 TF.ToTensor(),
                                 TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             )

    N = len(dataloader)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div

    split_arr = [1,2,3,5,10]
    for split in split_arr:
        split_scores = []
        for k in range(split):
            part = preds[k * (N // split): (k+1) * (N // split), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        print("mean :" + str(np.mean(split_scores)) + " -- std :" + str(np.std(split_scores)) + " -- Split : " + str(split))
    # return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    # cifar = dset.CIFAR10(root='data/', download=True,
    #                          transform=transforms.Compose([
    #                              transforms.Scale(32),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )
    #
    # IgnoreLabelDataset(cifar)

    print("Calculating Inception Score...")
    inception_score("E:\AdeelCoverGAN\Image Generation\scene_generation\outputs\cover_without_name", cuda=True, batch_size=1, resize=True, splits=1)
