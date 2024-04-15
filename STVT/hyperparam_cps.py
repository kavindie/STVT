from train import parse_args, val
from transfer import build_load_model
from STVT.datasets.TVSum import TVSum
import torch
import ruptures as rpt
from matplotlib import pyplot as plt
import numpy as np
from STVT.knapsack import knapsack
from STVT.eval import eval_metrics, upsample
import h5py

# TODO This will not work without the following mods
"""
Under eval.py inside the function select_keyshots add the following at the end
    return eval_arr, pred_summary
    #TODO get rid of the returning pred_summary

Under train.py in fucntion val
Instead of  
eval_res = select_keyshots(predicted_multi_list, video_number_list, image_number_list, target_multi_list, args)
do
eval_res, pred_summary = select_keyshots(predicted_multi_list, video_number_list, image_number_list, target_multi_list, args)

Add the following return staement
return pred_summary, predicted_multi_list, np.asarray(target_multi_list), video_number_list, image_number_list
and quote all below fscore_k = 0 
"""



best_f1_score = 0

def tune_hyperparam(preds, target_multi_list, kernel, min_size, jump, cps_number=30):
            # New additions as per their code
        model = "l2"
        pred_array = np.asarray(preds)[...,None]
        algo = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=jump).fit(pred_array)
        # result = algo.predict(n_bkps=cps_number) # This will always have the last frame by not the first one
        result = algo.predict(pen=1)
        print(f'kernel={kernel}, min_size:{min_size}, jump:{jump}, cps_number:{cps_number}')

        # If you want to plot
        plt.close()
        plt.plot(preds, alpha=0.5)
        k = np.zeros(len(preds) + 1) 
        k[result] = 1 
        plt.plot(k, 'r')
        plt.xlabel('Frame #')
        plt.ylabel('Importance score')
        plt.savefig('img.png')
        plt.close()

        cps = [[result[i-1], result[i]-1] for i in range(1, len(result))]
        cps.insert(0, [0, result[0]-1])

        return cps 
        cps = torch.as_tensor(cps)
        vidlen = int(cps[-1][1]) + 1
        weights = [sublist[1] - sublist[0] + 1 for sublist in cps]
        weights = torch.as_tensor(weights)
        pred_mean = torch.as_tensor([preds[cp[0]:cp[1]].mean() for cp in cps])

        _, selected = knapsack(pred_mean, weights, int(args.summary_prop * vidlen))
        selected = selected[::-1]
        key_labels = torch.zeros((vidlen,))
        for i in selected:
            key_labels[cps[i][0]:cps[i][1]] = 1
        pred_summary = key_labels.tolist()
        #pred_summary = torch.as_tensor(pred_summary)
        _,_, fscore = eval_metrics(pred_summary, target_multi_list)
        return fscore, pred_summary


def draw(predicted_multi_list, pred_summary, fscore, i):
     plt.plot(predicted_multi_list)
     plt.plot(pred_summary)
     plt.legend('Prediction', 'Selection')
     plt.title(f'F1 score is {fscore}')
     plt.savefig(f'img_{i}.png')
     plt.close()


def select_keyshots_my_version(predicted_list, video_number_list,image_name_list,target_list,args):
    global best_f1_score
    data_path = './STVT/STVT/datasets/datasets/'+str(args.dataset)+".h5" #hacky './STVT/datasets/datasets/'+str(args.dataset)+".h5"
    data_file = h5py.File(data_path)

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
        fea_sequencelen = (len(video['feature'][:])//args.sequence)*args.sequence
        True_all_video_len += fea_sequencelen

    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file['video_' + index]

        cps = tune_hyperparam(np.asarray(predicted_list), target_list, kernel=args.k, min_size=args.min_size, jump=args.jump, cps_number=args.cps_number)
        cps[-1][1] = len(video['feature'][:]) -1
        vidlen = int(cps[-1][1]) + 1
        weight = [sublist[1] - sublist[0] + 1 for sublist in cps]
        weight = torch.as_tensor(weight)
        fea_sequencelen = (len(video['feature'][:])//args.sequence)*args.sequence
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
        _,_, fscore = eval_metrics(pred_summary[:len(target_multi_list)], target_multi_list)

        if fscore > best_f1_score:
             best_f1_score = fscore
             print(f'Best f1 score at kernel, min_sie, jump, cps_no:{args.k, args.min_size, args.jump, args.cps_number}')
        
        print(f'fscore {fscore}')
        return fscore, pred_summary


def get_sum_of_cost(algo, n_bkps):
    """Return the sum of costs for the change points `bkps`"""
    bkps = algo.predict(n_bkps=n_bkps)
    return algo.cost.sum_of_costs(bkps)


def get_sum_of_cost_with_penality(algo, penalty_value):
    """Return the sum of costs for the change points `bkps`"""
    bkps = algo.predict(pen=penalty_value)
    return algo.cost.sum_of_costs(bkps)


def fig_ax(figsize=(20, 5), dpi=150):
    """Return a (matplotlib) figure and ax objects with given size."""
    return plt.subplots(figsize=figsize, dpi=dpi)

if __name__ == "__main__":
    args = parse_args()
    model_path = '/scratch2/kat049/Git/STVT/STVT/STVT/model/TVSum/model_save_name_roundtimes/TVSum_99_0.6389088961807157.pth'
    dataset = 'TVSum'

    args.model_path = model_path
    args.dataset = dataset
    args.batch_size = 1
    args.val_batch_size = 1
    args.cuda = False #not args.no_cuda and torch.cuda.is_available()
    args.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu
    args.summary_prop = 0.15
    
    dataset_path = f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/datasets/{args.dataset}.h5'
    loaded_model = build_load_model(args, model_path)

    epoch = 1
    args.test_dataset = ','.join(map(str, [i for i in range(3, 4)]))  #making the dataset only 3 
    args.video_amount = [3]
    _, test_loader, _ = TVSum(args)
    pred_summary, predicted_multi_list, target_multi_list, video_number_list, image_number_list = val(loaded_model, test_loader, epoch, args)

    precision, recall, fscore = eval_metrics(pred_summary[:len(target_multi_list)], target_multi_list)
    draw(predicted_multi_list, pred_summary, fscore, 0)
    print(f"Original method f1 score {fscore}")
    

    # # Method 1
    # # Using the elbow method to find the cps_number
    # algo = rpt.KernelCPD(kernel="linear").fit(np.asarray(predicted_multi_list))
    # # Choose the number of changes (elbow heuristic)
    # n_bkps_max = 20  # K_max
    # # Start by computing the segmentation with most changes.
    # # After start, all segmentations with 1, 2,..., K_max-1 changes are also available for free.
    # _   = algo.predict(n_bkps_max)

    # array_of_n_bkps = np.arange(1, n_bkps_max + 1)

    # fig, ax = fig_ax((7, 4))
    # ax.plot(
    #     array_of_n_bkps,
    #     [get_sum_of_cost(algo=algo, n_bkps=n_bkps) for n_bkps in array_of_n_bkps],
    #     "-*",
    #     alpha=0.5,
    # )
    # ax.set_xticks(array_of_n_bkps)
    # ax.set_xlabel("Number of change points")
    # ax.set_title("Sum of costs")
    # ax.grid(axis="x")
    # ax.set_xlim(0, n_bkps_max + 1)
    # plt.savefig('img_max_points.png')
    # plt.close()



    # # Method 2
    # # Using penality
    # algo = rpt.KernelCPD(kernel="linear").fit(np.asarray(predicted_multi_list))
    # # Choose the number of changes (elbow heuristic)
    # max_penality = 10  # K_max
    # # Start by computing the segmentation with most changes.
    # # After start, all segmentations with 1, 2,..., K_max-1 changes are also available for free.
    # _   = algo.predict(max_penality)

    # array_of_penalities = np.arange(1, max_penality + 1)
    # fig, ax = fig_ax((7, 4))
    # ax.plot(
    #     array_of_penalities,
    #     [get_sum_of_cost_with_penality(algo=algo, penalty_value=penality) for penality in array_of_penalities],
    #     "-*",
    #     alpha=0.5,
    # )
    # ax.set_xticks(array_of_n_bkps)
    # ax.set_xlabel("Penality value")
    # ax.set_title("Sum of costs")
    # ax.grid(axis="x")
    # ax.set_xlim(0, n_bkps_max + 1)
    # plt.savefig('img_penality.png')
    # plt.close()
    
    args.k = 'linear'
    args.min_size = 2
    args.jump = 1
    args.cps_number = 10
    fscore, pred_summary_2 = select_keyshots_my_version(predicted_multi_list, video_number_list, image_number_list, target_multi_list, args)
                    
    # By looking at it the cost of 10 cps is close to 150
    # with penality 1 cost is less than 80?

    print("todo here")

   