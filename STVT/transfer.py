import cv2
import torch
from train import parse_args
from STVT.build_model import build_model
import torch.backends.cudnn as cudnn
from STVT.build_dataloader import build_dataloader
from STVT.datasets.NewDataset import NewDataset
from STVT.datasets.TVSum import TVSum
from STVT.knapsack import knapsack
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
from STVT.eval import select_keyshots
from matplotlib import pyplot as plt
from PIL import Image
import ruptures as rpt
import os
import imageio
import numpy as np

def build_load_model(args, path):
    model = build_model(args)
    #args.cuda = False #not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True

    if args.cuda:
        model.to(torch.device("cuda"))
    #device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

    saved_keys = torch.load(path, map_location=args.device)
    model.load_state_dict(saved_keys)
    return model

if __name__ == "__main__":
    args = parse_args()
    args.dataset = 'NewDataset'
    batch_size = 1
    use_sigmoid = True
    model_path = '/scratch2/kat049/Git/STVT/STVT/STVT/model/TVSum/model_save_name_roundtimes/TVSum_99_0.6389088961807157.pth'
    dataset_path = f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/datasets/{args.dataset}.h5'

    print("Loading Model")
    loaded_model = build_load_model(args, model_path)
    print("Model Loaded")

    if 'TVSum' not in args.dataset and 'SumMe' not in args.dataset:
        test_data = NewDataset(file_dir=dataset_path, patch_number=16)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        args.test_dataset = ','.join(map(str, [i for i in range(11, 12)]))  #making the dataset only 12 
        args.video_amount = [11]
        args.val_batch_size = 1
        train_loader, test_loader, In_target = TVSum(args)

        
    with tqdm(total=len(test_loader), desc='Testing') as t:
        with torch.no_grad():
            predicted_multi_list = []
            video_number_list = []
            image_number_list = []
            for vals in test_loader:
                if 'TVSum' not in args.dataset and 'SumMe' not in args.dataset:
                    data, video_number, image_number = vals[0], vals[1], vals[2]
                else:
                    data, target, video_number, image_number = vals[0], vals[1], vals[2], vals[3]
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
                predicted_list = torch.Tensor(predicted_list).reshape(batch_size*args.sequence)
                video_number = video_number.reshape(batch_size*args.sequence)
                image_number = image_number.reshape(batch_size*args.sequence)
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

            # New additions as per their code
            model = "l2"
            cps_number = 30 # TODO need to see how this changes the behaviour
            pred_score = np.asarray(preds)[...,None]
            algo = rpt.KernelCPD(kernel="rbf", min_size=2, jump=5).fit(pred_score)
            result = algo.predict(n_bkps=cps_number) # This will always have the last frame by not the first one

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
            cps = torch.as_tensor(cps)
            vidlen = int(cps[-1][1]) + 1
            weights = [sublist[1] - sublist[0] + 1 for sublist in cps]
            weights = torch.as_tensor(weights)
            pred_value = torch.as_tensor([preds[cp[0]:cp[1]].mean() for cp in cps])

            _, selected = knapsack(pred_value, weights, int(0.15 * vidlen))
            selected = selected[::-1]
            key_labels = torch.zeros((vidlen,))
            for i in selected:
                key_labels[cps[i][0]:cps[i][1]] = 1
            pred_summary = key_labels.tolist()
            pred_summary = torch.as_tensor(pred_summary)

            #End of their code
            print("Creating a video")
            image_paths = []
            for i, value in enumerate(key_labels):
                if value == 1:
                    image_path = os.path.join(f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{args.dataset}/Images/', f'frame_{i}.jpg')
                    image_paths.append(image_path)

            # Create Video
            images = [cv2.imread(image_path) for image_path in image_paths]
            # Get dimensions of the first image
            height, width, _ = images[0].shape

            # Define video codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('output.avi', fourcc, 30, (width, height))
            for image in images:
                video_writer.write(image)
            
            video_writer.release()
            print("Video created")
            # Create GIF
            # images = []
            # for image_path in image_paths:
            #     if value == 1:
            #         images.append(Image.open(image_path))

            # # Optimize GIF creation parameters
            # duration = 100  # Adjust frame duration as needed
            # loop = 0  # Set loop count to 0 for infinite loop (or adjust as needed)

            # # Save the list of images as a GIF with optimized parameters
            # images[0].save('output.gif', save_all=True, append_images=images[1:], duration=duration, loop=loop, optimize=True, quality=95)


            # Getting key frames
            keyshots = []
            for i in selected:
                start_index = cps[i][0]
                end_index = cps[i][1]
                keyframe_index = start_index + np.argmax(preds[start_index:end_index])
                imp = preds[start_index:end_index].max()
                keyshots.append([keyframe_index, imp])

            # Draw first 20 keyshots
            images = []
            plt.figure(figsize=(16, 5))

            for i, frame in enumerate(keyshots):
                if i > 19:
                    break
                image = Image.open(os.path.join(f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{args.dataset}/Images/', f'frame_{int(frame[0])}.jpg')).convert("RGB")

                plt.subplot(4, 5, len(images) + 1)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])

                images.append(image)

            plt.tight_layout()
            plt.savefig('img.png')

            # Sorting
            keyshots = np.asarray(keyshots)
            sorted_indices = np.argsort(keyshots[:, 1])[::-1]
            sorted_array = keyshots[sorted_indices]
            # # Create the GIF from the images
            # with imageio.get_writer('output.gif', mode='I') as writer:
            #     for image_path in image_paths:
            #         image = imageio.imread(image_path)
            #         writer.append_data(image)

            # top_frame_num = 10
            # fig, axes = plt.subplots(1, top_frame_num)
            # indices = preds.topk(top_frame_num)[1]
            # for i, image_num in enumerate(indices):
            #     # Read the image using PIL
            #     img = Image.open(f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{args.dataset}/Images/frame_{image_num}.jpg')

            #     # Display the image in the corresponding subplot
            #     axes[i].imshow(img)
            #     axes[i].axis('off')  # Turn off axis labels

            # # Set a common title for the grid
            # fig.suptitle(f'Top {top_frame_num} Images in a Grid', fontsize=16)

            # # Adjust layout to prevent clipping of titles
            # plt.tight_layout()
            # plt.savefig(f'frame_imp_{args.dataset}.png')


            # # Apply KCPD algorithm
            # model = "l2"  # The cost function for the KCPD algorithm
            # algo = rpt.KernelCPD(kernel="rbf", min_size=2, jump=5).fit(preds.numpy())
            # # kernels can be -linear, gaussian (rbf), cosine(cosine)
            # # min_size 	int minimum segment length. 2
            # # jump 	int not considered, set to 1. 	1
            # result = algo.predict(pen=10) #Return the optimal breakpoints. Must be called after the fit method. pen 	float 	penalty value (>0). Defaults to None. Not considered if n_bkps is not None. 



    
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