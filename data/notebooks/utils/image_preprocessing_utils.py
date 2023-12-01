import os 
from skimage import io
from tqdm import tqdm 
import torch 

def readImage(image_path):
    img = io.imread(image_path)
    return img

def countImages(folder_path):
    return len(os.listdir(folder_path))

def transformImage(img, transforms):
    img_transformed = transforms(img.copy())
    return img_transformed

def getRepresentation(image, model):
    image = image.unsqueeze(0)
    with torch.no_grad():  # No need to track gradients when making predictions
        representation = model(image)
    return representation

def saveRepresentations(read_folder_path, save_folder_path, model, transforms, device):
    
    # Total number of images
    total_images = int(countImages(read_folder_path)/2)

    # set model to device
    model.to(device)
    model.eval()

    # Looping through all images 
    for i in tqdm(range(total_images)):

        ## Read paths
        read_path = os.path.join(read_folder_path, str(i) + '.tif')
        save_path = os.path.join(save_folder_path, str(i) + '.pt')

        ## Read image
        img = readImage(read_path)

        ## Transform image & put to DEVICE
        img_transformed = transformImage(img, transforms)
        img_transformed = img_transformed.to(device)
        
        ## Get representation
        representation = getRepresentation(img_transformed, model)

        ## Squeeze first dimension
        representation = representation.squeeze(0).detach().cpu()

        ## Save representation
        saveArray(representation, save_path)

def saveArray(array, path):
    torch.save(array, path)

def loadArray(path):
    return torch.load(path)