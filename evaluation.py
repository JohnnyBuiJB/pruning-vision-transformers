!pip install datasets # install dataset loader
!pip install transformers # install vision transformers
!huggingface-cli login # needed to get imagenet-1k (need account with hugging face)

from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
import time
import torch

BATCH_SIZE = 32

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model (DeiT base)
model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
model = model.to(device)

# Define feature extractor
feature_extractor = ViTImageProcessor.from_pretrained('facebook/deit-base-patch16-224')

# Load data set stream (need auth token from hugging face - create account)
dset = load_dataset('imagenet-1k', split='validation', streaming=True, use_auth_token=True)

# batch data set
def group_batch(batch):
    return {k: [v] for k, v in batch.items()}
    
dset_b = dset.map(group_batch, batched=True, batch_size=BATCH_SIZE)

# Evalution loop
correct = [] # hold T/F from predictions

start = time.time() # start time

for i, img in enumerate(iter(dset_b), 1):
  cur_img = img['image'] # get current image
  cur_b = [img.convert('RGB') for img in cur_img] # (had to convert to rgb due to some grayscale imgs)
  cur_idx = img['label'] # get current label
  inputs = feature_extractor(images=cur_b, return_tensors="pt").to(device) # extract features for input (move to gpu)
  with torch.no_grad():
    outputs = model(**inputs) # get model output
  logits = outputs.logits.to('cpu') # get logits (move back to cpu - labels on cpu, correct array on cpu)
  predicted_class_idx = logits.argmax(-1).numpy() # get predicted class index (array)
  correct += list(predicted_class_idx == cur_idx) # append to correct list (array)

  print(f"Batch {i} complete, {(i)*32} imgs.") # print batch, number of images evaluated
  if i == 10: # only do first 10 batches - 320 imgs (just for checking right now)
    break

end = time.time()

print("Time for eval:", round(end - start, 3), "seconds.") # time of eval
print("Accuracy:", round(sum(correct)/len(correct), 4)*100, "%") # accuracy

