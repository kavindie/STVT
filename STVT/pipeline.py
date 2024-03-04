import argparse
import cv2
from tqdm import tqdm
import torch
import os
import h5py
from torch.utils.data import DataLoader
import torch.nn as nn
import ruptures as rpt 
import numpy as np
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from PIL import Image

from openai import OpenAI
import io
import base64
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

from train import parse_args
from STVT.build_model import build_model
from STVT.datasets.NewDataset import NewDataset
from STVT.datasets.processNewDataset import Predictor, process_image
from STVT.knapsack import knapsack


# Define command-line arguments
# Todo figure out how to do this
parser = argparse.ArgumentParser(description='Create a Summary instance.')
parser.add_argument('--video_path', type=str, help='Path to where the video is saved.', required=True)
parser.add_argument('--image_path', type=str, help='Path to where the extracted images are saved.', required=True)


# Cleaning the code

class Summarize:
    def __init__(self):
        self.h5_file =  f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/datasets/{args.dataset}.h5'
        self.model = None
        self.data = None
        self.data_loader = None
        self.fps = None
        self.features = None

        # First extract the frames of the video

        # Create the directory if not existance
        if not os.path.exists(args.image_path):
            # If it doesn't exist, create the directory
            print(f"{args.image_path} does not exist. Creating one")
            os.makedirs(args.image_path)

        self.process_video()

        # Second create and load dataset
        if not os.path.exists(self.h5_file):
            self.create_dataset()
        self.load_dataset()

        # Third the output from the model
        self.load_model()

        # Generate predcitions
        with h5py.File(self.h5_file, 'a') as f:
            if 'preds' in f:
                preds = torch.asarray(f['preds'][:])
            else:
                preds = self.generate_preds(open_file=f)

        if self.features is None:
            with h5py.File(self.h5_file, 'r') as hf:
                self.features = torch.asarray(hf['features'][:])
                self.features = self.features[...,0,0]
        
        # # Get cluster features
        # pca_features, closest_points_indices = self.get_cluster_features(self.features)
        # self.draw_imgs(closest_points_indices, title='img_clusters.png')
        pca = PCA(n_components=args.pca_comps, random_state=42)
        pca.fit(self.features)
        x = pca.transform(self.features)

        # generate cps
        cps, weights, selected, key_labels, pred_summary, _, _ = self.generate_cps(preds, pca_features=x)
        _, _, _, _, _, keyshots, sorted_keyshots               = self.generate_cps(preds, pca_features=x, use_nbkps=True)
        # Generate video
        self.generate_video(key_labels)
        
        # See what ChatGPT do
        # f = min(len(closest_points_indices), len(keyshots))
        self.generate_text(keyshots=keyshots)
        # self.generate_text(keyshots=closest_points_indices)

        
        print("Complete")
        
    # Function to extract frames 
    def process_video(self): 
        print("Starting to process the video")
        # Path to video file 
        vidObj = cv2.VideoCapture(args.video_path) 

        # How many frames 
        total_frames  = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Used as counter variable 
        count = 0
        
        # how mnay fps
        self.fps = vidObj.get(cv2.CAP_PROP_FPS)

        # fps_to_keep = 1 #TODO maybe change
        # interval = int(round(fps / fps_to_keep))

        # #
        # current_frame = 0

        # for _ in tqdm(range(total_frames )):
        #     if  current_frame < total_frames:
        #         vidObj.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        #         success, image = vidObj.read() 
        #         if success:
        #             image_path_name = f'{args.image_path}/frame_{count}.jpg'
        #             if not os.path.exists(image_path_name):
        #                 cv2.imwrite(f'{args.image_path}/frame_{count}.jpg', image) 
        #             count += 1
        #         current_frame += interval

        # checks whether frames were extracted 
        success = 1
        while success: 
            for i in tqdm(range(total_frames )):
                # vidObj object calls read 
                # function extract frames 
            
                success, image = vidObj.read() 
                
                # resize and save
                # image = ResizeImages(image, size=224)
                
                # Saves the frames with frame-count 
                image_path_name = f'{args.image_path}/frame_{count}.jpg'
                if not os.path.exists(image_path_name):
                    cv2.imwrite(f'{args.image_path}/frame_{count}.jpg', image) 
                
                count += 1
            if count == total_frames :
                break
        print(f"Created {count} frames")

        print("Video processed")

    def create_dataset(self):
        predictor = Predictor().to(args.device)
    
        image_features = []
        
        # Process each .jpg file in the folder
        file_list = os.listdir(args.image_path)
        progress_bar = tqdm(file_list, desc="Processing Files", unit="file")
        for file_name in progress_bar:
            if file_name.endswith(".jpg"):
                image_path = os.path.join(args.image_path, file_name)
                features = process_image(image_path, model=predictor, device=args.device)
                image_features.append(features)
        
        image_features = torch.stack(image_features)
        self.features = image_features

        # Save the features to an h5 file
        with h5py.File(self.h5_file, 'w') as hf:
            hf.create_dataset('features', data=image_features.cpu().numpy())

    def load_dataset(self):
        self.data = NewDataset(file_dir=self.h5_file, patch_number=args.sequence)
        self.data_loader = DataLoader(dataset=self.data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    def load_model(self):
        print("Loading Model")

        self.model = build_model(args)
        cudnn.benchmark = True

        if args.cuda:
            self.model.to(torch.device("cuda"))
        
        saved_keys = torch.load(args.model_path, map_location=args.device)
        self.model.load_state_dict(saved_keys)
        
        print("Model Loaded Successfully")

    def generate_preds(self, open_file):
        with tqdm(total=len(self.data_loader), desc='Testing') as t:
            with torch.no_grad():
                predicted_multi_list = []
                for d in self.data_loader:
                    data = d[0]
                    predicted_list = []
                    if args.cuda:
                        data = data.cuda()
                    output = self.model(data)
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
                    predicted_list = torch.Tensor(predicted_list).reshape(args.batch_size*args.sequence)
                    predicted_multi_list += predicted_list.tolist()
                    
                predicted_multi_list = [float(i) for i in predicted_multi_list]
                preds = torch.as_tensor(predicted_multi_list)

                # save the preds for future
                open_file.create_dataset('preds', data=preds)

                return preds

    def draw_imgs(self, keyshots, title='img.png'):
        # Draw first 20 keyshots
        images = []
        plt.figure(figsize=(16, 5))

        for i, frame in enumerate(keyshots):
            if i > 19:
                break
            image = Image.open(os.path.join(args.image_path, f'frame_{int(frame[0])}.jpg')).convert("RGB")

            plt.subplot(4, 5, len(images) + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images.append(image)

        plt.tight_layout()
        plt.savefig(title)
        plt.close()

    def generate_cps(self, preds, pca_features=None, use_nbkps=False):

            vidlen = len(self.features) # this is going to be greater than the preds
            last_pred_val = preds[-1]
            preds = torch.cat((preds, last_pred_val.repeat(vidlen - len(preds))))
            
            if pca_features is None:    
                pca_features = preds
                pred_array = np.asarray(preds)[...,None]
            else:
                pred_array = np.asarray(pca_features)
            
            algo = rpt.KernelCPD(kernel="linear").fit(pred_array)
            penalty = True

            if use_nbkps:
                result = algo.predict(n_bkps=args.cps_number) # This will always have the last frame by not the first one
                cps = [[result[i-1], result[i]-1] for i in range(1, len(result))]
                cps.insert(0, [0, result[0]-1])
                cps = torch.as_tensor(cps)
            else:
                for _ in range(2):
                    if penalty:
                        result = algo.predict(pen=1) # This will always have the last frame by not the first one
                    else:
                        result = algo.predict(n_bkps=args.cps_number) # This will always have the last frame by not the first one   
                    
                    cps = [[result[i-1], result[i]-1] for i in range(1, len(result))]
                    cps.insert(0, [0, result[0]-1])
                    cps = torch.as_tensor(cps)
                    if len(cps) < args.cps_number: 
                        penalty = False
                    else:
                        break
            # If you want to plot
            plt.close()
            plt.plot(preds, alpha=0.5)
            k = np.zeros(len(pred_array) + 1) 
            k[result] = 1 
            plt.plot(k, 'r')
            plt.xlabel('Frame #')
            plt.ylabel('Importance score')
            plt.savefig('imp_score.png')
            plt.close()

            weights = [sublist[1] - sublist[0] + 1 for sublist in cps]
            weights = torch.as_tensor(weights)
            pred_mean = torch.as_tensor([preds[cp[0]:cp[1]].mean() for cp in cps])

            _, selected = knapsack(pred_mean, weights, int(args.summary_prop * vidlen))
            selected = selected[::-1]
            key_labels = torch.zeros((vidlen,))
            for i in selected:
                key_labels[cps[i][0]:cps[i][1]] = 1
            pred_summary = key_labels.tolist()
            pred_summary = torch.as_tensor(pred_summary)

            # Selecting only args.num_cluster amount of keyshots 
            # keyshot_indices = np.ones(len(selected))
            # feature_keyshots = np.ones((len(selected), self.features.shape[-1]))
            # keyshots_imps = np.ones(len(selected))
            # for selection_count, i in enumerate(selected):
            #     start_index = cps[i][0]
            #     end_index = cps[i][1]
            #     keyframe_index = start_index + np.argmax(preds[start_index:end_index])
            #     imp = preds[keyframe_index]

            #     keyshot_indices[selection_count] = keyframe_index
            #     feature_keyshots[selection_count] = self.features[keyframe_index]
            #     keyshots_imps[selection_count] = imp
                
            # _, closest_points_indices = self.get_cluster_features(feature_keyshots)
            # closest_points_indices  = closest_points_indices[:, 0].astype(int)

            # keyshots = np.ones((args.clusters, 2))
            # keyshots[:,0] = keyshot_indices[closest_points_indices]
            # keyshots[:,1] = keyshots_imps[closest_points_indices]
            
            keyshots = []
            for i in selected:
                start_index = cps[i][0]
                end_index = cps[i][1]
                keyframe_index = start_index + np.argmax(preds[start_index:end_index])
                imp = preds[start_index:end_index].max()
                keyshots.append([keyframe_index, imp])
            

            # Draw first 20 keyshots
            self.draw_imgs(keyshots, title='img_cps.png')
            
            # Sorting
            keyshots = np.asarray(keyshots)
            sorted_indices = np.argsort(keyshots[:, 1])[::-1]
            sorted_keyshots = keyshots[sorted_indices]

            return cps, weights, selected, key_labels, pred_summary, keyshots, sorted_keyshots

    def generate_video(self, key_labels):
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
            video_writer = cv2.VideoWriter('output.avi', fourcc, self.fps, (width, height))
            for image in images:
                video_writer.write(image)
            
            video_writer.release()
            print("Video created")

    # def get_cluster_features(self, features):
    #     pca = PCA(n_components=args.pca_comps, random_state=42)
    #     pca.fit(features)
    #     x = pca.transform(features)
        
        
    #     kmeans = KMeans(
    #         init="random",
    #         n_clusters=args.clusters,
    #         #n_init=10,
    #         #max_iter=300,
    #         #random_state=42
    #     )

    #     kmeans.fit(x)
    #     cluster_centers = kmeans.cluster_centers_
    #     #labels = kmeans.predict(x)
    #     distances = euclidean_distances(x, cluster_centers)
    #     closest_points_indices = distances.argmin(axis=0)
    #     closest_points_indices.sort()
    #     #closest_points = features[closest_points_indices]
    #     closest_points_indices = np.stack((closest_points_indices, np.ones_like(closest_points_indices)*(1/len(closest_points_indices))), axis=-1)
    #     return x, closest_points_indices


    def generate_text(self, keyshots):
        base64Frames = []
        for frame_num, _ in keyshots:
            image_path = f'{args.image_path}/frame_{int(frame_num)}.jpg'
            with open(image_path, "rb") as img_file:
                    img_bytes = img_file.read()
            #image = Image.open(io.BytesIO(img_bytes))
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            base64Frames.append(img_base64)

        # TODO : Need to hide this
        client=OpenAI(api_key='XXX')
        PROMPT_MESSAGE_1 = [
            {
                "role": "user",
                "content": [
                    "Title: Object Identification Challenge\n \
                    Introduction: Welcome to the object identification challenge! \
                    In this activity, you will be shown images from a scene and your task is to identify the objects within it.\
                    Scene Description: You are a robot navigating through an underground tunnel, \
                    conducting a search and rescue mission. You are interested in locating the following artifacts: \
                    survivors wearing yellow high-visibility jackets, cell phones, red backpacks, drills, fire extinguishers, gas, vent, \
                    helmets, blue ropes and cubes.\n Task: Identify as many objects as you can from the list above. \
                    List the objects in the order you notice them. If you do not notice any simply say 'No objects deteced' \n \
                    Remember to pay attention to details and think creatively!\n\
                    ",
                    *map(lambda x: {"image": x}, base64Frames[:10]),
                ],
            },
        ]
        PROMPT_MESSAGE_2= [
            {
                "role": "user",
                "content": [
                    "Title: Object Identification Challenge\n \
                    Introduction: Welcome to the object identification challenge! \
                    In this activity, you will be shown images from a scene and your task is to identify the objects within it.\
                    Scene Description: You are a robot navigating through an underground tunnel, \
                    conducting a search and rescue mission. You are interested in locating the following artifacts: \
                    survivors wearing yellow high-visibility jackets, cell phones, red backpacks, drills, fire extinguishers, gas, vent, \
                    helmets, blue ropes and cubes.\n Task: Identify as many objects as you can from the list above. \
                    List the objects in the order you notice them. If you do not notice any simply say 'No objects deteced' \n \
                    Remember to pay attention to details and think creatively!\n\
                    ",
                    *map(lambda x: {"image": x}, base64Frames[10:20]),
                ],
            },
        ]

        prompt_messages = [PROMPT_MESSAGE_1, PROMPT_MESSAGE_2]
        for p in prompt_messages:
            params = {
                "model": "gpt-4-vision-preview",
                "messages": p,
                "max_tokens": 500,
                # "temperature": 0.4,
                # "n":2, 
                # "presence_penalty":0.8,
            }

            result = client.chat.completions.create(**params)
            print(result.choices[0].message.content)

        

if __name__ == '__main__': 
    video_path = '/scratch2/kat049/tmp/camera0-1024x768-002.mp4'
    image_path = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/camera0-1024x768-002/Images'
    model_path = '/scratch2/kat049/Git/STVT/STVT/STVT/model/TVSum/model_save_name_roundtimes/TVSum_99_0.6389088961807157.pth'

    args = parse_args()
    args.video_path = video_path
    args.image_path = image_path
    args.model_path = model_path
    args.dataset = image_path.split('/')[-2]
    args.batch_size = 1
    args.val_batch_size = 1
    args.cuda = False #not args.no_cuda and torch.cuda.is_available()
    args.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu
    args.cps_number = 19 # TODO need to see how this changes the behaviour
    args.summary_prop = .15 #TODO if you need to change the summary propotion
    args.pca_comps = 100
    args.clusters = 20
    

    Summarize()
