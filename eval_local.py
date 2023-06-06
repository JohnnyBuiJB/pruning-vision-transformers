# Import libraries
from datasets import load_dataset, Image # need to import Image for local files to be decoded (handled automatically)
from transformers import ViTForImageClassification
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Define path to your cache where imagenet stored
DATA_PATH = './hpc-share/.cache'

# Set hyperparameters
batch_size = 32
num_classes = 1000

# Define ImageNet Normalization values (from PyTorch)
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset (from local cache path) - currently only using 10% of val data (!!CHANGE THIS!!)
dset = load_dataset('imagenet-1k', split='validation', cache_dir=DATA_PATH)

# Define preprocessing for each image (pulled from huggingface deit inference time processing)
transform = transforms.Compose([
    transforms.Resize((256,256)), # Resize image to 256x256
    transforms.CenterCrop((224, 224)), # Center crop image to 224x224
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD) # Normalize image using imagenet values
])

# Custom collate function to handle PIL images
def collate_fn(batch):
    images = [transform(img_path["image"].convert('RGB')) for img_path in batch] # create list of images 
    labels = [item["label"] for item in batch] # create list of labels
    return {"image": torch.stack(images), "label": labels} # return dict of images and labels


def eval_model(model):
    # Create a DataLoader to batch the dataset
    data_loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Define values for loop
    correct = 0 # hold number of correct preds
    n_check = 75 # how often to print out checkpoints (number of batches to do before each print)
    n_batches = round(len(dset)/(batch_size*n_check)) # how many total batches in the data set

    model.to(device) # move model to device
    model.eval() # put model in evaluation mode

    print(f"\nStarting evaluation loop using {device}:\n")
    start = time.time() # start time

    for i, batch in enumerate(data_loader, 1):
        images = batch["image"].to(device) # get current batch images
        labels = batch["label"] # get current batch labels (always on cpu)
        #inputs = feature_extractor(images=images, return_tensors="pt").to(device) # extract features for input (move to gpu)
        with torch.no_grad():
            outputs = model(images) # get model output
        logits = outputs.logits.to('cpu') # get logits (move back to cpu - labels on cpu, correct array on cpu)
        predicted_class_idx = logits.argmax(-1).numpy() # get predicted class index (array)
        correct += sum(predicted_class_idx == labels) # append to correct list (array)

        if i % n_check == 0:
            print(f"Batch {int(i/n_check)}/{n_batches} complete - {(i)*batch_size} imgs.")

    end = time.time()
    print("\nTime for eval:", round((end - start)/60, 2), "minutes.")
    print("Final Validation Accuracy:", round(correct/len(dset), 4)*100, "%")

    return

def main():
    # Define model (DeiT base - could change this to load pruned model)
    model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
    model.config.num_classes = num_classes

    eval_model(model) # run model through validation set (dset)

    return

if __name__ == "__main__":
    main()
    

    
    
    
