import torch
from torchvision import transforms
from avalanche.benchmarks import nc_benchmark, ni_benchmark
from avalanche.benchmarks.classic.ccub200 import SplitCUB200
from avalanche.benchmarks.classic import CLStream51
from datasets.datasets import get_dataset
from datasets.cir_sampling import cir_sampling_cifar100, cir_sampling_tinyimagenet

def get_transforms(dataset_name):
    if dataset_name == "stream51":
        _mu, _std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset_name == "cifar100":
        _mu, _std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    else:  # ImageNet stats for other datasets
        _mu, _std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    if dataset_name in ["cub200", "fgvc_aircraft"]:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mu, std=_std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mu, std=_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mu, std=_std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mu, std=_std),
        ])
    
    return train_transform, test_transform

def get_dataset_benchmark(dataset_name, data_path, train_transform, test_transform):
    train_set = get_dataset(dataset_name, data_path, train=True, transform=train_transform)
    test_set = get_dataset(dataset_name, data_path, train=False, transform=test_transform)
    return train_set, test_set

def get_nc_benchmark(train_set, test_set, n_experiences=10):
    return nc_benchmark(
        train_set, test_set,
        n_experiences=n_experiences,
        shuffle=True,
        seed=1234,
        task_labels=False
    )

def get_ni_benchmark(train_set, test_set, n_experiences=10):
    return ni_benchmark(
        train_set, test_set,
        n_experiences=n_experiences,
        shuffle=True,
        seed=1234
    )

def get_cir_benchmark(dataset_name, data_path):
    cir_params = {
        'n_e': 500,
        's_e': 3500,
        'p_a': 1.0,
        'sampler_type': "random",
        'use_all_samples': False,
        'dist_first_occurrence': {'dist_type': 'geometric', 'p': 0.01},
        'dist_recurrence': {'dist_type': 'fixed', 'p': 0.2},
        'seed': 1234
    }
    
    if dataset_name == "cifar100":
        return cir_sampling_cifar100(dataset_root=data_path, **cir_params)
    elif dataset_name == "tinyimagenet":
        return cir_sampling_tinyimagenet(dataset_root=data_path, **cir_params)
    else:
        raise ValueError(f"CIR benchmark not supported for dataset: {dataset_name}")

def get_stream51_benchmark():
    return CLStream51(scenario="class_instance", seed=10, bbox_crop=True)

def get_cl_benchmark(dataset_name, data_path, benchmark_type, n_experiences):
    train_transform, test_transform = get_transforms(dataset_name)
    
    if dataset_name == "stream51":
        return get_stream51_benchmark()
    
    if dataset_name not in ["cifar100", "tinyimagenet", "cub200", "fgvc_aircraft"]:
        raise ValueError(f"Unsupported dataset for CL benchmark: {dataset_name}")
    
    train_set, test_set = get_dataset_benchmark(dataset_name, data_path, train_transform, test_transform)
    
    if benchmark_type == "nc":
        return get_nc_benchmark(train_set, test_set, n_experiences)
    elif benchmark_type == "ni":
        return get_ni_benchmark(train_set, test_set, n_experiences)
    elif benchmark_type == "cir":
        return get_cir_benchmark(dataset_name, data_path)
    else:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}")