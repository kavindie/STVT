import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import h5py 



def plotImgs(dataset, selected_i, value):
    number_of_cols = 10
    for j in range(selected_i.shape[0]//number_of_cols):
        fig, axes = plt.subplots(1, number_of_cols, figsize=(15, 2))
        for i, image_num in enumerate(selected_i[number_of_cols*j:number_of_cols*(j+1)]):
            img = Image.open(f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{dataset}/Images/frame_{image_num}.jpg')
            axes[i].imshow(img)
            axes[i].axis('off')
        fig.suptitle(f'First {number_of_cols*(j+1)} Images in a Grid with value {v.float()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'test_{number_of_cols*(j+1)}_{v.float()}.png')
        plt.close()


def knapsack(weights, values, capacity):
    """values = profit"""
    n = len(weights)
    dp = torch.zeros((n + 1, capacity + 1), dtype=torch.float32)
    selected_items = torch.zeros((n + 1, capacity + 1), dtype=torch.bool)
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i, w] = torch.max(values[i-1] + dp[i-1, (w-weights[i-1].long())], dp[i-1,w])
                selected_items[i, w] = True
            else:
                dp[i, w] = dp[i - 1, w]
    
    selected_indices = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if type(w) == torch.Tensor:
            w = int(w)
        if selected_items[i, w]:
            selected_indices.append(i - 1)
            w -= weights[i - 1]
        i -= 1

    selected_items_indices = torch.tensor(selected_indices)
    
    return dp[n, capacity], selected_items_indices



def readh5File(file_dir):
    with h5py.File(file_dir, "r") as f:
        for key in f.keys():
            if key == 'video_11':
                video = f[key]
                features = torch.as_tensor(video['feature'][:])
                label = torch.as_tensor(video['label'][:])
                weight = torch.as_tensor(video['n_frame_per_seg'][:])
                cps = torch.as_tensor(video['change_points'][:])
                true_summary_arr_20 = torch.as_tensor(video['user_summary'][:])
    return features, label, weight, cps, true_summary_arr_20


def select_keyshots(predicted_list, video_number_list,image_name_list, dataset, sequence=16):
    """Adding stuff in the original function
    args.dataset = dataset
    args.sequence = 16
    target_list = 
    """
    with h5py.File(f'./STVT/STVT/datasets/datasets/{dataset}.h5', "r") as f:
        for key in f.keys():
            if key == 'video_11':
                video = f[key]
    predicted_single_video = []
    predicted_single_video_list = []
    target_single_video = []
    target_single_video_list = []
    video_single_list = list(set(video_number_list))
    eval_arr = []

    for i in range(len(image_name_list)):
        if image_name_list[i] == 1 and i!=0:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)
            predicted_single_video = []
            target_single_video = []

        predictedL = [predicted_list[i]]
        predicted_single_video += predictedL
        targetL = list(map(int, str(target_list[i])))
        target_single_video += targetL

        if i == len(image_name_list)-1:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)
    video_single_list_sort = sorted(video_single_list)
    True_all_video_len = 0
    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file['video_' + index]
        fea_sequencelen = (len(video['feature'][:])//sequence)*sequence
        True_all_video_len += fea_sequencelen

    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file['video_' + index]
        cps = video['change_points'][:]
        vidlen = int(cps[-1][1]) + 1
        weight = video['n_frame_per_seg'][:]
        fea_sequencelen = (len(video['feature'][:])//sequence)*sequence
        for ckeck_n in range(len(video_single_list_sort)):
            dif = True_all_video_len-len(predicted_list)
            if len(predicted_single_video_list[ckeck_n]) == fea_sequencelen or len(predicted_single_video_list[ckeck_n]) == fea_sequencelen-dif:
                pred_score = np.array(predicted_single_video_list[ckeck_n])
                up_rate = vidlen//len(pred_score)
                # print(up_rate)
                break
        #pred
        pred_score = upsample(pred_score, up_rate, vidlen)
        pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
        _, selected = knapsack(pred_value, weight, int(0.15 * vidlen))
        selected = selected[::-1]
        key_labels = np.zeros((vidlen,))
        for i in selected:
            key_labels[cps[i][0]:cps[i][1]] = 1
        pred_summary = key_labels.tolist()
        true_summary_arr_20 = video['user_summary'][:]
        eval_res = [eval_metrics(pred_summary, true_summary_1) for true_summary_1 in true_summary_arr_20]
        eval_res = np.mean(eval_res, axis=0).tolist() if dataset == "TVSum" else np.max(eval_res, axis=0).tolist()
        eval_arr.append(eval_res)
        
    return eval_arr




if __name__ == "__main__":
    # Original data
    file_path = '/scratch2/kat049/tmp/ydata-tvsum50-anno.tsv' 
    df = pd.read_csv(file_path, sep='\t')
    target_id = 'i3wAGJaaktw' # What is the target, I want video 11 = i3wAGJaaktw
    filtered_df = df[df.iloc[:, 0] == target_id] # Video ID / Category / Annotations
    anno = np.asarray(filtered_df.iloc[:,2])  # Annotation
    anno_array  = [] 
    for i in range(anno.shape[0]): 
        k_i = np.asarray(list(map(int, anno[i].split(',')))) 
        anno_array.append(k_i) 
    anno_array = torch.as_tensor(np.asarray(anno_array)) # Annotation of shape [20,4700]
    avgs = anno_array.float().mean(dim=0)
    avgs_0_1 = avgs/5

    # Processed data
    file_dir = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/datasets/TVSum.h5'
    features, label, weight, cps, true_summary_arr_20 = readh5File(file_dir)
 








# do select_keyshots
predicted_list = avgs_0_1
video_number_list = torch.ones_like(avgs_0_1)*11
image_name_list = torch.arange(avgs_0_1.shape[0])
dataset = 'TestDataset_11'
sequence=16, 
select_keyshots(predicted_list, video_number_list, image_name_list, dataset)


# weights = torch.ones(4700)
# capacity = int(4700*.15)
# kk, selected_items_indices = knapsack(weights, avgs_0_1, capacity)


def pick_frames_per_val(avgs):
    sorted_avgs = avgs.sort(descending=True)
    value_check = None
    selected_i = []
    for val,idx in zip(sorted_avgs[0], sorted_avgs[1]):
        if value_check is None: # Initializing
            value_check = val
        else:
            if val == value_check: # Same value
                selected_i.append(idx)
            else:
                selected_i = torch.as_tensor(np.asarray(selected_i))
                frames = torch.zeros(4700)
                frames[selected_i]  = 1
                plotImgs('TestDataset_11', selected_i, val)
                # plt.plot(frames)
                # plt.title(f'For the values {val}, I found {selected_i.shape[0]} frames')
                # plt.savefig(f'{val}.png')
                # plt.close()
                value_check = val
                selected_i = []



