import os
import pandas as pd
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset
from avalanche.benchmarks.classic import CLStream51

import os
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CUB200Dataset(Dataset):
    def __init__(self, data_path, is_train=True, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.loader = default_loader
        
        # Load image list
        images_file = os.path.join(data_path, 'images.txt')
        self.images = pd.read_csv(images_file, sep=' ', names=['img_id', 'img_path'])
        
        # Load train/test split
        split_file = os.path.join(data_path, 'train_test_split.txt')
        split_data = pd.read_csv(split_file, sep=' ', names=['img_id', 'is_training_image'])
        self.images = self.images.merge(split_data, on='img_id')
        
        # Filter based on train/test flag
        self.images = self.images[self.images['is_training_image'] == int(is_train)]
        
        # Load class labels
        labels_file = os.path.join(data_path, 'image_class_labels.txt')
        labels = pd.read_csv(labels_file, sep=' ', names=['img_id', 'class_id'])
        self.images = self.images.merge(labels, on='img_id')
        
        # Load class names
        classes_file = os.path.join(data_path, 'classes.txt')
        self.classes = pd.read_csv(classes_file, sep=' ', names=['class_id', 'class_name'])
        self.class_to_idx = {cls_id: idx for idx, cls_id in enumerate(self.classes['class_id'])}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, 'images', self.images.iloc[idx]['img_path'])
        class_id = self.images.iloc[idx]['class_id']
        
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, self.class_to_idx[class_id]

def load_cub200_dataset(data_path, batch_size=32, image_size=224):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CUB200Dataset(data_path, is_train=True, transform=transform)
    test_dataset = CUB200Dataset(data_path, is_train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


def get_dataset(dataset_name, data_path, train=True, transform=None):
    if dataset_name == 'cifar100':
        return datasets.CIFAR100(root=data_path, train=train, download=True, transform=transform)
    elif dataset_name == 'cub200':
        
        return load_cub200_dataset(data_path, batch_size=32, image_size=224)
    elif dataset_name == 'fgvc_aircraft':
        split = 'trainval' if train else 'test'
        return datasets.FGVCAircraft(root=data_path, split=split, annotation_level='variant', transform=transform, download=True)
    # elif dataset_name == 'stream51':
    #     return CLStream51(data_path, train=train, transform=transform)
    elif dataset_name == 'tinyimagenet':
        # Assume TinyImageNet is organized in a way compatible with ImageFolder
        split = 'train' if train else 'val'
        return datasets.ImageFolder(root=os.path.join(data_path, split), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")