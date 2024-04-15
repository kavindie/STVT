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
        
        # Get pca features
        pca = PCA(n_components=args.pca_comps, random_state=42)
        pca.fit(self.features)
        x = pca.transform(self.features)

        # generate cps
        cps, weights, selected, key_labels, pred_summary, keyshots, sorted_keyshots = self.generate_cps(preds, pca_features=x)

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
        self.features = image_features[..., 0, 0]

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

    def generate_cps(self, preds, pca_features=None):

            vidlen = len(self.features) # this is going to be greater than the preds
            last_pred_val = preds[-1]
            preds = torch.cat((preds, last_pred_val.repeat(vidlen - len(preds))))
            
            if pca_features is None:    
                pca_features = preds
                pred_array = np.asarray(preds)[...,None]
            else:
                pred_array = np.asarray(pca_features)
            
            algo = rpt.KernelCPD(kernel="cosine", min_size=15).fit(pred_array)
            penalty = True

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

            keyshot_indices = []
            for i in selected:
                start_index = cps[i][0]
                end_index = cps[i][1]
                keyframe_index = start_index + np.argmax(preds[start_index:end_index])
                keyshot_indices.append(keyframe_index)
            
            keyshot_indices = np.asarray(keyshot_indices)
            kmeans = KMeans(init="k-means++", n_clusters=args.clusters, n_init=10)
            x = pca_features[keyshot_indices.astype(int)]
            kmeans.fit(x)
            
            cluster_labels = kmeans.labels_  # Get cluster labels for each data point
            cluster_centers = kmeans.cluster_centers_ # Get cluster centers
            representative_points = []
            for i in range(args.clusters):  # 20 clusters
                indices = np.where(cluster_labels == i)[0] # Get indices of data points in the cluster
                distances = np.linalg.norm(x[indices] - cluster_centers[i], axis=1)  # Calculate distances of all data points in the cluster to the cluster center
                closest_point_index = indices[np.argmin(distances)] # Get the index of the point closest to the cluster center
                representative_points.append(keyshot_indices[closest_point_index]) # Add the closest point to the list of representative points

            representative_points.sort()
            keyshots = []
            for i in representative_points:
                keyframe_index = i
                imp = preds[i]
                keyshots.append([keyframe_index, imp])

            
            # Draw first 20 keyshots
            self.draw_imgs(keyshots, title=f'{args.output_folder}/img_cps_new.png')
            
            # Sorting
            keyshots = np.asarray(keyshots)
            sorted_indices = np.argsort(keyshots[:, 1])[::-1]
            sorted_keyshots = keyshots[sorted_indices]

            return cps, weights, selected, key_labels, pred_summary, keyshots, sorted_keyshots

    def generate_video(self, key_labels):
            #End of their code
            print("Creating a summary video")
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
            video_writer = cv2.VideoWriter(f'{args.output_folder + "/output_new"}.avi', fourcc, self.fps, (width, height))
            for image in images:
                video_writer.write(image)
            
            video_writer.release()
            print("Video created")

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
        history = ""
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
            history += result.choices[0].message.content
            #print(result.choices[0].message.content)
        
        output_file = f'{args.output_folder}/output_text_new.txt'

        # Open the output text file in write mode
        with open(output_file, 'w') as file:
            # Write the text to the file
            file.write(history)

        

if __name__ == '__main__': 
    model_path = '/scratch2/kat049/Git/STVT/STVT/STVT/model/TVSum/model_save_name_roundtimes/TVSum_99_0.6389088961807157.pth' 
    
    args = parse_args() 
    args.model_path = model_path 
    args.batch_size = 1 
    args.val_batch_size = 1 
    args.cuda = False #not args.no_cuda and torch.cuda.is_available() 
    args.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu 
    args.cps_number = 40 # TODO need to see how this changes the behaviour 
    args.summary_prop = .15 #TODO if you need to change the summary propotion 
    args.pca_comps = 100 
    args.clusters = 20 


    # robot = 'bear'
    # camera_no = 'all' 
    # image_path = f'/scratch2/kat049/Git/STVT/STVT/STVT/datasets/{robot}_{camera_no}/Images' 
    # h5_files = [
    #         '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/bear_camera0-1024x768-002',
    #         '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/bear_camera0-1024x768-003',
    #         '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/bear_camera0-1024x768-004',
    #         '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/bear_camera0-1024x768-005',
    #         '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/bear_camera0-1024x768-006',
    #         '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/bear_camera0-1024x768-007',
    #         ]

    # # Initialize lists to store features and predictions
    # all_features = []
    # all_preds = []

    # # Read data from each .h5 file and append to lists
    # for filename in h5_files:
    #     with h5py.File(filename, 'r') as file:
    #         features = file['features'][:]
    #         preds = file['preds'][:]
    #         all_features.append(features)
    #         all_preds.append(preds)

    # # Concatenate lists along the first axis to create a single dataset
    # merged_features = np.concatenate(all_features, axis=0)
    # merged_preds = np.concatenate(all_preds, axis=0)

    # # Create a new HDF5 file to store merged data
    # output_file = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/bear_camera0-1024x768-002'
    # with h5py.File(output_file, 'w') as file:
    #     # Write merged data to the new HDF5 file
    #     file.create_dataset('features', data=merged_features)
    #     file.create_dataset('preds', data=merged_preds)
    #     # Loop for videos 
    
    directory = '/scratch2/kat049/tmp/bear' 


    # List to store the names of .mp4 files 
    mp4_files = [] 
    # Iterate through all files and directories in the specified directory 
    for filename in os.listdir(directory): 
        # Check if the file ends with .mp4 
        if filename.endswith('.mp4'): 
            # If yes, append the file name to the list 
            mp4_files.append(os.path.join(directory, filename)) 

    for video_path in mp4_files: 
        if 'camera0-1024x768-000' in video_path or 'camera0-1024x768-001' in video_path:
            continue


        args.video_path = video_path 
        args.image_path = image_path 
        args.robot = robot 
        args.camera_no =  camera_no 
        args.dataset = image_path.split('/')[-2] 
        args.output_folder = f'/scratch2/kat049/Git/STVT/outputs/{args.robot}/{args.camera_no}'
        
        Summarize()
        # if not os.path.exists(args.output_folder):
        #     print(f"Processing {video_path}")
        #     # If it doesn't exist, create the directory
        #     print(f"{args.output_folder} does not exist. Creating one")
        #     os.makedirs(args.output_folder)
             
        # else:
        #     continue

        
    
    #video_path = '/scratch2/kat049/tmp/camera0-1024x768-002.mp4'
    #image_path = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/camera0-1024x768-002/Images'
    
