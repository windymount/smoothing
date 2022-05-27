import scipy.io as sio
from scipy.stats import norm
import numpy as np
import torch
from core import Smooth, LipFlexSmooth
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def get_topk_examples(matfile, idx_offset, k=1):
    """
    
    Get top-k largest radius examples for each class.
    
    """
    matdict = sio.loadmat(matfile)
    radius = matdict['hard'].squeeze()
    labels = matdict['labels']
    NUM_CLASSES = 10
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = CIFAR10(
        root='data', train=True, download=True, transform=transform_train)
    results = []
    for i in range(NUM_CLASSES):
        cls_idx = np.where(labels == i)[1]
        radius_cls = radius[cls_idx]
        radius_sorted_idx = np.argsort(radius_cls)[-k:]
        results.append(cls_idx[radius_sorted_idx]+idx_offset)
    items = np.concatenate(results)
    target_imgs = []
    for i in items:
        target_imgs.append(trainset[i][0])
    return torch.stack(target_imgs).reshape((len(items), -1))


def generate_sigma(x, lam, sigma0, refer_set):
    """
    
    Generate sigma using NN.
    
    """
    x = x.reshape(-1)
    min_dist = torch.min(torch.norm(refer_set - x, dim=1)).item()
    return sigma0 - min_dist * lam


def generate_Lipschitz_sigma(refset_pth, lam, sigma0):
    # Read refset
    with open(refset_pth, 'r') as f:
        refset = torch.load(f)
        refset = refset.reshape((refset.shape[0], -1))
    sigma_generator = lambda x: generate_sigma(x, lam, sigma0, refset)
    return sigma_generator


def certify_with_flexible_sigma(model, device, dataset, num_classes, ref_set, matfile=None,
            start_img=0, num_img=500, skip=1, sigma0=0.25, lam=0.05, N0=100, N=100000,
            alpha=0.001, batch=1000, verbose=False,
            grid=(0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25)):
    print('===certify(N={}, lambda={})==='.format(N, lam))

    model.eval()
    smoothed_net = Smooth(model, num_classes,
                        sigma0)
    Lip_generator = generate_Lipschitz_sigma(ref_set, lam, sigma0)
    Lip_net = LipFlexSmooth(Lip_generator, lam, 724, model, num_classes, sigma0)

    radius_flex = np.zeros((num_img,), dtype=float)
    radius_hard = np.zeros((num_img,), dtype=float)
    num_grid = len(grid)
    cnt_grid_flex = np.zeros((num_grid + 1,), dtype=int)
    cnt_grid_hard = np.zeros((num_grid + 1,), dtype=int)
    s_flex, s_hard = 0.0, 0.0
    for i in range(num_img):
        img, target = dataset[start_img + i * skip]
        img = img.to(device)
        # Get sigma
        print('Certifying example {:d}/{:d}: sample sigma {:.3f}.'.format(i+1, num_img, smoothed_net.sigma))
        p_hard, r_hard = smoothed_net.certify(
                img, N0, N, alpha, batch)
        correct = int(p_hard == target)
        p_flex, r_flex = Lip_net.certify(
                img, N0, N, alpha, batch)
        correct_fl = int(p_flex == target)
        if verbose:
            if correct == 1:
                print('Correct: 1. Radius: {}.'.format(r_hard))
            else:
                print('Correct: 0.')
            if correct_fl == 1:
                print('Flex Correct: 1. Radius: {}.'.format(r_flex))
            else:
                print('Flex Correct: 0.')
        radius_flex[i] = r_flex if correct_fl == 1 else -1
        radius_hard[i] = r_hard if correct == 1 else -1
        if correct == 1:
            cnt_grid_hard[0] += 1
            s_hard += r_hard
            for j in range(num_grid):
                if r_hard >= grid[j]:
                    cnt_grid_hard[j + 1] += 1
        if correct_fl == 1:
            cnt_grid_flex[0] += 1
            s_flex += r_flex
            for j in range(num_grid):
                if r_flex >= grid[j]:
                    cnt_grid_flex[j + 1] += 1

    print('===Certify Summary===')
    print('Total Image Number: {}'.format(num_img))
    print('Flex summary')
    print('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_flex[0], cnt_grid_flex[0] / num_img * 100))
    for j in range(num_grid):
        print('Radius: {}  Number: {}  Acc: {}'.format(
            grid[j], cnt_grid_flex[j + 1], cnt_grid_flex[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_flex / num_img))
    print('Hard summary')
    print('Radius: 0.0  Number: {}  Acc: {}'.format(
        cnt_grid_hard[0], cnt_grid_hard[0] / num_img * 100))
    for j in range(num_grid):
        print('Radius: {}  Number: {}  Acc: {}'.format(
            grid[j], cnt_grid_hard[j + 1], cnt_grid_hard[j + 1] / num_img * 100))
    print('ACR: {}'.format(s_hard / num_img))
    if matfile is not None:
        sio.savemat(matfile, {"flex": radius_flex, "hard": radius_hard})


if __name__ == "__main__":
    examples = get_topk_examples("mat/440.mat", 500, k=5)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = CIFAR10(
        root='data', train=True, download=True, transform=transform_train)
    x, _ = trainset[10]
    generate_sigma(x, 0.05, 0.5, examples)
