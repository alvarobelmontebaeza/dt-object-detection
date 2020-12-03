import torch
import torchvision
from engine import train_one_epoch
import utils

import transforms as T

class Dataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # Load all npz files and extract data
        self.dataset_files = list(sorted(filter(lambda x: "npz" in x, os.listdir(root + "/data_collection/dataset"))))
        self.dataset_size = len(self.dataset_files)
    
    def __getitem__(self,idx):
        # Check that required idx is smaller than dataset size
        if idx >= self.dataset_size:
            print('ERROR: Required index %s out of bounds.' % idx)
            print('Dataset size is %s' % self.dataset_size)
            exit()
        # Load the item required by the idx parameter
        file = str(idx) + '.npz'
        with np.load(f'./dataset/{file}') as data:
            # Retrieve data
            img, boxes, classes = tuple([data[f"arr_{i}"] for i in range(3)])
            # Convert to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            img_id = torch.tensor([idx])
            # Define target dictionary
            target = {}
            target['boxes'] = boxes
            target['labels'] = classes
            target['image_id'] = img_id
            # Perform transforms if required
            if self.transforms is not None:
                img, target = self.transforms(img,target)

        # Return image and data dictionary.
        return img, target

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of `./weights`!
    pass

if __name__ == "__main__":
    main()