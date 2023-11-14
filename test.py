import torch
from torchvision import transforms
from model import VEATIC_baseline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_model(weights_path, device):
    '''
    function to load model with saved weights
    Params: weights_path (string), device (torch device)
    returns: model (VEATIC_baseline obj)
    '''
    model = VEATIC_baseline()  
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_frames(data_path, transform):
    '''
    function to transform images into a stack of tensors
    Params: data_path (string), transform (Transform obj)
    Returns: torch.stack(frames) (sequence of tensors)
    '''
    frames = []
    frame_filenames = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')])
    for filename in frame_filenames:
        image = Image.open(filename).convert('RGB')
        image = transform(image)
        frames.append(image)
    return torch.stack(frames)

def run_model(frames_path, device):
    '''
    function of main routine meant for easier OOP integration
    Params: frames_path (string), device (torch device)
    returns: all_preds (2-d array of predictions)
    '''
    weights_path = '/Users/vedphadke/Documents/GitHub/VEATIC/one_stream_vit.pth'  # Update this path as necessary

    model = load_model(weights_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    frames_tensor = process_frames(frames_path, transform)
    preds = []

    # Testing loop in groups of 5
    for i in range(0, len(frames_tensor), 5):
        subset_frames = frames_tensor[i:i+5].to(device)
        
        # Padding if necessary
        if subset_frames.size(0) < 5:
            padding = torch.zeros(5 - subset_frames.size(0), *subset_frames.size()[1:])
            subset_frames = torch.cat((subset_frames, padding), dim=0)

        subset_frames_batch = subset_frames.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            predictions = model(subset_frames_batch)
            preds.append(predictions.cpu().numpy())

    all_preds = np.concatenate(preds, axis=0)

    # returning preds
    return all_preds

def visualize_results(valence_arousal):
    '''
    void function to visualize model predictions of valence and arousal
    Params: valence_arousal (2-d numpy array)
    '''
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(valence_arousal[:, 0])
    ax[0].set_title('Valence over Time')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Valence')

    ax[1].plot(valence_arousal[:, 1])
    ax[1].set_title('Arousal over Time')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Arousal')

    plt.show()

if __name__ == '__main__':
    weights_path = '/Users/vedphadke/Documents/GitHub/VEATIC/one_stream_vit.pth'
    frames_path = '/Users/vedphadke/Desktop/DSU_Projects/Social_Intelligence/frame_save_test/0'

    device = torch.device('cpu')

    # load weights
    model = load_model(weights_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),      #choosing 224x224 because of resNet50 model input size
        transforms.ToTensor(),
    ])


    frames_tensor = process_frames(frames_path, transform)
    preds = []

    # testing loop in groups of 5
    for i in range(0, len(frames_tensor), 5):
        subset_frames = frames_tensor[i:i+5].to(device)
        
        # padding if necessary
        if subset_frames.size(0) < 5:
            padding = torch.zeros(5 - subset_frames.size(0), *subset_frames.size()[1:])
            subset_frames = torch.cat((subset_frames, padding), dim=0)

        subset_frames_batch = subset_frames.unsqueeze(0)  #Add batch dimension
    

        with torch.no_grad():
            predictions = model(subset_frames_batch)
            preds.append(predictions.cpu().numpy())

    all_preds = np.concatenate(preds, axis=0)

    visualize_results(all_preds)

    np.savetxt('valence_arousal_output.csv', all_preds, delimiter=',', header='Valence,Arousal', comments='')
