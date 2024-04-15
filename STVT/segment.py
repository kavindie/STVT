import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
import numpy as np
from PIL import Image

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda" if use_cuda else "cpu"

def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    #x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    #c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def segment(features, dataset):
    k =10
    cl,c = KMeans(features[..., 0,0], k)
    # cl =[4700]
    # c =[k, 512]
    plt.close()
    plt.hist(cl, bins=k)
    plt.xlabel('Class number')
    plt.ylabel('Number of frames per class')
    plt.title('Histogram of frames per class depending on the features')
    plt.savefig('img.png')
    plt.close()

    selected_frames = []
    for class_num in range(k):
        frames_in_class = torch.zeros_like(cl)
        for i, frame_num in enumerate(cl):
            #print(i)
            if frame_num == class_num:
                frames_in_class[i] = 1
        indices = torch.argwhere(frames_in_class == 1)
        random_index = np.random.choice(len(indices))
        selected_frames.append(indices[random_index])

    selected_frames = torch.as_tensor(selected_frames)
    selected_frames_v_i = torch.sort(selected_frames)
    columns = 5
    rows = int(k / columns)
    fig, axes = plt.subplots(rows, columns)
    for i, image_num in enumerate(torch.sort(selected_frames)[0]):
        # Read the image using PIL
        img = Image.open(f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{dataset}/Images/frame_{image_num}.jpg')

        r = i //  columns
        c = i%columns
        # Display the image in the corresponding subplot
        axes[r,c].imshow(img)
        axes[r,c].axis('off')  # Turn off axis labels

    fig.suptitle(f'Frames selected randomly per class and plotted chronologically', fontsize=6)