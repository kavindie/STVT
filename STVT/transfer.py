import torch
from train import parse_args
from STVT.build_model import build_model
import torch.backends.cudnn as cudnn
from STVT.build_dataloader import build_dataloader
from STVT.datasets.NewDataset import NewDataset
from STVT.datasets.TVSum import TVSum
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from STVT.eval import select_keyshots
from matplotlib import pyplot as plt
from PIL import Image

def build_load_model(args, path):
    model = build_model(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True

    if args.cuda:
        model.cuda()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    saved_keys = torch.load(path)
    model.load_state_dict(saved_keys)
    return model

if __name__ == "__main__":
    args = parse_args()
    args.dataset = 'TVSum'
    use_sigmoid = True
    model_path = '/scratch2/kat049/Git/STVT/STVT/STVT/model/TVSum/model_save_name_roundtimes/TVSum_99_0.6389088961807157.pth'
    dataset_path = f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/datasets/{args.dataset}.h5'
    loaded_model = build_load_model(args, model_path)
    if 'TVSum' not in args.dataset and 'SumMe' not in args.dataset:
        test_data = NewDataset(file_dir=dataset_path, patch_number=16)
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    else:
        args.test_dataset = ','.join(map(str, [i for i in range(1, 2)]))  #making the dataset only 1 
        args.video_amount = [1]
        train_loader, test_loader, In_target = TVSum(args)

        
    with tqdm(total=len(test_loader), desc='Testing') as t:
        with torch.no_grad():
            predicted_multi_list = []
            video_number_list = []
            image_number_list = []
            for data, video_number, image_number in test_loader:
                predicted_list = []
                if args.cuda:
                    data = data.cuda()
                output = loaded_model(data)
                video_number = video_number
                image_number = image_number
                multi_output = output

                if use_sigmoid:
                    for sequence in range(args.sequence):
                        output = multi_output[sequence]
                        predicted_ver2 = []
                        sigmoid = nn.Sigmoid()
                        outputs_sigmoid = sigmoid(output)
                        for s in outputs_sigmoid:
                            predicted_ver2.append(float(s[1]))
                        predicted_list.append(predicted_ver2)
                else:
                    softmax = nn.Softmax(dim=-1)
                    predicted_list.append(torch.argmax(softmax(multi_output), dim=-1, keepdim=True))

                t.update(1)
                predicted_list = torch.Tensor(predicted_list).permute(1,0)
                predicted_list = torch.Tensor(predicted_list).reshape(args.val_batch_size*args.sequence)
                video_number = video_number.reshape(args.val_batch_size*args.sequence)
                image_number = image_number.reshape(args.val_batch_size*args.sequence)
                predicted_multi_list += predicted_list.tolist()
                video_number_list += video_number.tolist()
                image_number_list += image_number.tolist()

            predicted_multi_list = [float(i) for i in predicted_multi_list]
            # target_multi_list = [torch.zeros_like(torch.tensor(predicted_multi_list))]
            # eval_res = select_keyshots(predicted_multi_list, video_number_list, image_number_list, target_multi_list, args)
            # fscore_k = 0
            # for i in eval_res:
            #     fscore_k+=i[2]
            # fscore_k/= len(list(args.test_dataset.split(",")))
            # pd_F_measure_k.append(fscore_k)

            preds = torch.as_tensor(predicted_multi_list)
            top_frame_num = 10
            hk
            indices = preds.topk(top_frame_num)[1]
            for i, image_num in enumerate(indices):
                # Read the image using PIL
                img = Image.open(f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{args.dataset}/Images/frame_{image_num}.jpg')

                # Display the image in the corresponding subplot
                axes[i].imshow(img)
                axes[i].axis('off')  # Turn off axis labels

            # Set a common title for the grid
            fig.suptitle(f'Top {top_frame_num} Images in a Grid', fontsize=16)

            # Adjust layout to prevent clipping of titles
            plt.tight_layout()
            plt.savefig(f'frame_imp_{args.dataset}.png')


    
"""How to run
run /scratch2/kat049/Git/STVT/STVT/STVT/datasets/video_processing.py 
    params:
        path = path/to/video
	    saving_path = path/to/save/extracted/images -> Need to create this path too

run /scratch2/kat049/Git/STVT/STVT/STVT/datasets/processNewDataset.py
    params:
        folder_path = path/to/save/extracted/images

run /scratch2/kat049/Git/STVT/STVT/transfer.py
    params:
        args.dataset = '?'
"""