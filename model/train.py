import numpy as np
import os

import torch
import torchvision
from engine import train_one_epoch
import utils

import transforms as T
import model
from model import Model

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
    
    def __len__(self):
        return self.dataset_size

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # Check for GPU availability during training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Create dataset object, giving the root as the parent directory
    dataset = Dataset('.', get_transform(train=True))
    # Define training dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    # Define model
    model = Model()
    # Move model to the right device
    model.to(device)
    # Construct optimizer. (maybe try Adam?)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # Constructs learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    # Set number of epochs
    num_epochs = 10

    # Main training loop
    for epoch in range(num_epochs):
        # Try for one epoch and print every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # Update learning rate
        lr_scheduler.step()
    
    print('Training complete! Saving weights...')
    torch.save(model.state_dict(), './weights')
    print('Weights saved!')

if __name__ == "__main__":
    main()