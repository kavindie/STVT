import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import h5py 
import ruptures as rpt
from train import val, parse_args
from transfer import build_load_model
from STVT.knapsack import knapsack
from STVT.datasets.TVSum import TVSum
from STVT.datasets.NewDataset import NewDataset
from STVT.datasets.TVSum import TVSum
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import torch.nn as nn
from KTS.cpd_nonlin import cpd_nonlin


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


# def knapsack(weights, values, capacity):
#     """values = profit"""
#     n = len(weights)
#     dp = torch.zeros((n + 1, capacity + 1), dtype=torch.float32)
#     selected_items = torch.zeros((n + 1, capacity + 1), dtype=torch.bool)
    
#     for i in range(1, n + 1):
#         for w in range(1, capacity + 1):
#             if weights[i - 1] <= w:
#                 dp[i, w] = torch.max(values[i-1] + dp[i-1, (w-weights[i-1].long())], dp[i-1,w])
#                 selected_items[i, w] = True
#             else:
#                 dp[i, w] = dp[i - 1, w]
    
#     selected_indices = []
#     i, w = n, capacity
#     while i > 0 and w > 0:
#         if type(w) == torch.Tensor:
#             w = int(w)
#         if selected_items[i, w]:
#             selected_indices.append(i - 1)
#             w -= weights[i - 1]
#         i -= 1

#     selected_items_indices = torch.tensor(selected_indices)
    
#     return dp[n, capacity], selected_items_indices



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



def normalize_to_zero_one(numbers):
    """
    Normalize a PyTorch tensor of shape [20, 4700] to the range 0 to 1 independently for each column.

    Parameters:
    - numbers: PyTorch tensor of shape [20, 4700].

    Returns:
    - normalized_numbers: Normalized PyTorch tensor.
    """

    # Find the minimum and maximum values for each column
    min_values, _ = torch.min(numbers, dim=0)
    max_values, _ = torch.max(numbers, dim=0)

    # Normalize each column to the range 0 to 1
    normalized_numbers = (numbers - min_values) / (max_values - min_values)

    return normalized_numbers

def detect_cps(X, cps_number=10):
    """
    params:
        X : A tensor of shape [number_ofitems, feature_dim]
        cps_number = amound of change points needed
    Apply KCPD Algorithm
    1. KernelCPD() 
        kernels can be -linear, gaussian (rbf), cosine(cosine)
        min_size 	int minimum segment length. 2
         jump 	int not considered, set to 1. 	1
    2. predict()
        Return the optimal breakpoints. 
        Must be called after the fit method. 
        pen 	float 	penalty value (>0). Defaults to None. Not considered if n_bkps is not None. 
    """
    model = "l2"
    algo = rpt.KernelCPD(kernel="rbf", min_size=2, jump=5).fit(X)
    result = algo.predict(n_bkps=cps_number)
    return result

    n = X.shape[0]
    m = cps_number
    K = np.dot(X, X.T)
    cps_gen, _ = cpd_nonlin(K, m, lmin=1, lmax=10000)
    cps_gen = np.insert(cps_gen, 0, 0) # adding 0 in the begging
    cps_gen = np.insert(cps_gen, len(cps_gen), n)  # adding 1 in the begging
    plt.close()
    plt.plot(X)
    mi = np.min(X)
    ma = np.max(X)
    for cp in cps_gen:
        plt.plot([cp, cp], [mi, ma], 'r')
    plt.savefig('img.png')
    plt.close()
    return cps_gen


def convert_to_keyshots(frame_scores, max_duration, cps_number=10):
    """
    Convert frame-level importance scores into keyshots.

    Parameters:
    - frame_scores: 1D array representing importance scores for each frame.
    - max_duration: Maximum total duration of keyshots.

    Returns:
    - keyshot_frames: Indices of selected keyframe frames.
    """

    # Perform temporal segmentation (you can replace this with your own segmentation method)
    # For illustration, we assume you already have change points
    change_points = detect_cps(frame_scores, cps_number=cps_number)
    # This returns eg [5, ... , Final frame number] with len being cps_number+1
    change_points_mod = np.insert(change_points, 0, 0)

    print("Change points are calculated")
    # Compute interval-level scores by averaging scores within each interval
    interval_scores = []
    for i in range(len(change_points_mod) - 1):
        start_index = change_points_mod[i]
        end_index = change_points_mod[i + 1]
        interval_score = frame_scores[start_index:end_index].mean()
        interval_scores.append(interval_score)

    # Rank intervals in descending order by their scores
    ranked_intervals = np.argsort(interval_scores)[::-1]

    # Select intervals in order until the total duration is below the threshold (knapsack algorithm)
    selected_keyshots = []
    total_duration = 0
    for i in ranked_intervals:
        start_index = change_points[i]
        end_index = change_points[i + 1]
        interval_duration = end_index - start_index
        if total_duration + interval_duration <= max_duration:
            selected_keyshots.append((start_index, end_index))
            total_duration += interval_duration
        else:
            break

    # Pick the frame with the highest importance score within each keyshot as a keyframe
    keyshot_frames = []
    for keyshot in selected_keyshots:
        start_index, end_index = keyshot
        if frame_scores.shape[1] > 1:
            keyframe_index = start_index + np.argmax(frame_scores[start_index:end_index].mean(axis=1))
        else:
            keyframe_index = start_index + np.argmax(frame_scores[start_index:end_index])
        keyshot_frames.append(keyframe_index)
    change_points = np.asarray(change_points)
    selected_keyshots = np.asarray(selected_keyshots)
    keyshot_frames = np.asarray(keyshot_frames)
    return change_points, selected_keyshots, keyshot_frames

def keyshots_to_labels(selected_keyshots, labels_gt, length=None):
    if labels_gt is None and length is None:
        print("Either give the ground truth label or the length of the video")
        return None
    
    if labels_gt is not None:
        length = labels_gt.shape[0]
    
    label = np.zeros(length)

    for start, end in selected_keyshots:
        label[start : end + 1] = 1

    label = np.asarray(label)

    if labels_gt is not None:  
        plt.close()
        plt.step(range(length), label , where='mid')
        plt.step(range(length), labels_gt, where='mid')
        plt.legend(['Calculated', 'Ground Truth'])
        plt.savefig('img.png')
        plt.close()
    
    return label

def draw_keyframes(keyshot_frames, dataset='TestDataset_11'):
    plt.close()
    top_frame_num = keyshot_frames.shape[0]
    fig, axes = plt.subplots(1, top_frame_num)
    for i, image_num in enumerate(keyshot_frames):
        # Read the image using PIL
        img = Image.open(f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{dataset}/Images/frame_{image_num}.jpg')

        # Display the image in the corresponding subplot
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis labels

    # Set a common title for the grid
    fig.suptitle(f'Top {top_frame_num} images as per manual cal of original annotations', fontsize=6)

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()
    plt.savefig(f'img.png')
    plt.close()


def val_my_method(model, val_loader, epoch, args):
    
    model.eval()
    if epoch == -1:
        epoch = args.epochs - 1
    with tqdm(
        total=len(val_loader), desc='Validate Epoch #{}'.format(epoch + 1)
    ) as t:
        with torch.no_grad():
            predicted_multi_list = []
            video_number_list = []
            image_number_list = []
            for data, video_number, image_number in val_loader:
                predicted_list = []
                if args.cuda:
                    #data = data.to(torch.device("cuda:2"))
                    data = data.cuda()
                output = model(data)
                video_number = video_number
                image_number = image_number
                multi_output = output
                for sequence in range(args.sequence):
                    output = multi_output[sequence]
                    predicted_ver2 = []
                    sigmoid = nn.Sigmoid()
                    outputs_sigmoid = sigmoid(output)
                    for s in outputs_sigmoid:
                        predicted_ver2.append(float(s[1]))
                    predicted_list.append(predicted_ver2)
                t.update(1)
                predicted_list = torch.Tensor(predicted_list).permute(1,0)
                predicted_list = torch.Tensor(predicted_list).reshape(args.val_batch_size*args.sequence)
                video_number = video_number.reshape(args.val_batch_size*args.sequence)
                image_number = image_number.reshape(args.val_batch_size*args.sequence)
                predicted_multi_list += predicted_list.tolist()
                video_number_list += video_number.tolist()
                image_number_list += image_number.tolist()

            predicted_multi_list = [float(i) for i in predicted_multi_list]

            return predicted_multi_list
        
if __name__ == "__main__":
    # Running with the saved model
    epoch = 1
    args = parse_args()
    args.dataset = 'TrainDataset_3'
    args.val_batch_size = 1
    model_path = '/scratch2/kat049/Git/STVT/STVT/STVT/model/TVSum/model_save_name_roundtimes/TVSum_99_0.6389088961807157.pth'
    #model_path = '/scratch2/kat049/Git/STVT/STVT/STVT/model/TVSum/model_2_roundtimes/TVSum_5_0.6717641997941182.pth'
    dataset_path = f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/datasets/{args.dataset}.h5'
    loaded_model = build_load_model(args, model_path)
    if 'TVSum' not in args.dataset and 'SumMe' not in args.dataset:
        test_data = NewDataset(file_dir=dataset_path, patch_number=16)
        test_loader = DataLoader(dataset=test_data, batch_size=args.val_batch_size, shuffle=False, drop_last=True)
        predicted_multi_list = val_my_method(loaded_model, test_loader, epoch, args)
    else:
        args.test_dataset = ','.join(map(str, [i for i in range(11, 12)]))  #making the dataset only 12 
        args.video_amount = [11]
        _, test_loader, _ = TVSum(args)
        predicted_multi_list, target_multi_list = val(loaded_model, test_loader, epoch, args)

    
    vidlen = len(predicted_multi_list)
    max_duration = int(vidlen*.15)
    pred_score = np.asarray(predicted_multi_list)[..., None]
    change_points, selected_keyshots, keyshot_frames = convert_to_keyshots(pred_score, max_duration, cps_number=30)
    up_rate = vidlen//len(pred_score)
    
    # change points go from [0, ...., last frame]
    cps = []
    for i in range(len(change_points)-1):
        if cps == []:
            cps += [[0, change_points[i]-1]]
        cps += [[change_points[i], change_points[i+1]-1]]
    cps += [[change_points[-1], vidlen]]
    cps = np.asarray(cps)
      
    weight = []
    for s,e in cps:
        weight += [(e - s + 1)]
    weight = np.asarray(weight)
    
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
    _, selected = knapsack(pred_value, weight, int(0.15 * vidlen))
    selected = selected[::-1]
    key_labels = np.zeros((vidlen,))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1

    

    # Original data
    file_path = '/scratch2/kat049/tmp/ydata-tvsum50-anno.tsv' 
    df = pd.read_csv(file_path, sep='\t')
    target_id = 'J0nA4VgnoCo' # video_3 'i3wAGJaaktw' # What is the target, I want video 11 = i3wAGJaaktw
    filtered_df = df[df.iloc[:, 0] == target_id] # Video ID / Category / Annotations
    anno = np.asarray(filtered_df.iloc[:,2])  # Annotation
    anno_array  = [] 
    for i in range(anno.shape[0]): 
        k_i = np.asarray(list(map(int, anno[i].split(',')))) 
        anno_array.append(k_i) 
    anno_array = torch.as_tensor(np.asarray(anno_array)) # Annotation of shape [20,4700]
    normalized_anno_array = normalize_to_zero_one(anno_array)
    mean_anno_array = normalized_anno_array.mean(dim=0)

    max_duration = int(normalized_anno_array.shape[1]*.15) #15% of the video
    # X = normalized_anno_array[0,:][..., None].numpy() # Test 1
    # X = mean_anno_array[..., None].numpy() # Test 2
    #change_points, selected_keyshots, keyshot_frames = convert_to_keyshots(X, max_duration, cps_number=30)

    # Processed data
    file_dir = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/datasets/TVSum.h5'
    features, label, weight, cps, true_summary_arr_20 = readh5File(file_dir)
    X = features.numpy() #Test 3
    change_points, selected_keyshots, keyshot_frames = convert_to_keyshots(X, max_duration, cps_number=30)


    t = true_summary_arr_20.mean(dim=0)
    t = normalize_to_zero_one(t)
    plt.bar(range(4700), label)
    plt.bar(range(4700), t)
    plt.legend(['Label', 'Mean of each person'])
    plt.title('Based on the processed data')
    plt.savefig('img.png')
    plt.close()
 









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



