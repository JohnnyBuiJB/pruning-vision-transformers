import os
import torch
import datasets
from datasets import load_dataset, list_datasets
import re
import shutil

TRAIN_DATA_PATH = '/nfs/hpc/share/buivy/pruning-vision-transformers/train'
VAL_DATA_PATH = '/nfs/hpc/share/buivy/pruning-vision-transformers/val'

DATA_PATH = '/nfs/hpc/share/buivy/pruning-vision-transformers/.cache'

def huggingface_to_imagenet_folder(huggingface, folder, split):
    dset = load_dataset(path='imagenet-1k', split=split, use_auth_token=True,
                        cache_dir=huggingface, num_proc=4)

    num_added_png = 0
    for i in range(36000, len(dset)):
        path = os.path.join(folder, str(dset[i]['label']))
        if not os.path.exists(path):
            os.mkdir(path)

        fn = 'img%d.png' % i
        file_path = os.path.join(path, fn)
        if not os.path.exists(file_path):
            dset[i]['image'].save(file_path, 'PNG')
            num_added_png += 1
            
        if num_added_png > 0 and num_added_png % 10000 == 0:
            print(num_added_png, "pngs were added")
    
def recursive_count_files(root):
    num_file = 0

    for i in os.listdir(root):
        path = os.path.join(root, i)
        for fn in os.listdir(path):
            file_path = os.path.join(path, fn)
            if os.path.isfile(file_path):
                num_file += 1
    print(num_file)



# 1. Move files to folders
def move():
    for class_index in os.listdir(VAL_DATA_PATH):
        val_class_path = os.path.join(VAL_DATA_PATH, class_index)
        
        for img_name in os.listdir(val_class_path):
            img_index, _ = re.split('\.', img_name[3:])
            
            if img_index >= 567842:
                img_path = os.path.join(val_class_path, img_name)
                new_img_path = os.path.join(TRAIN_DATA_PATH, class_index, img_name)
                shutil.move(img_path, new_img_path)
    

# for dir in os.listdir(TRAIN_DATA_PATH):
#     path = os.path.join(TRAIN_DATA_PATH, dir)
#     if os.path.isfile(path):
#         folder_name, fn = re.split('img', dir)
#         fn = 'img' + fn
#         folder_path = os.path.join(TRAIN_DATA_PATH, folder_name)
        
#         if not os.path.exists(folder_path):
#             os.mkdir(folder_path)
        
#         # print(path)
#         # print(os.path.join(folder_path, fn))
#         shutil.move(path, os.path.join(folder_path, fn))
# 876img316283.png


# file = open('log.txt', 'w')
# file.writelines(os.listdir(TRAIN_DATA_PATH))
# file.close()

if __name__ == '__main__':
    # move()
    recursive_count_files(VAL_DATA_PATH)
    # huggingface_to_imagenet_folder(DATA_PATH, VAL_DATA_PATH, 
    #                            datasets.Split.VALIDATION)