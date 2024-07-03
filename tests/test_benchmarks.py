import pytest
from data.datasets import get_cl_benchmark, get_transforms

@pytest.mark.parametrize("dataset_name", ["cifar100", "tinyimagenet"])
@pytest.mark.parametrize("benchmark_type", ["nc", "ni", "cir"])
def test_cl_benchmark(dataset_name, benchmark_type, tmp_path):
    train_transform, test_transform = get_transforms(dataset_name)
    
    benchmark = get_cl_benchmark(
        dataset_name, 
        str(tmp_path), 
        train_transform, 
        test_transform, 
        benchmark_type,
        n_experiences=5
    )
    
    assert hasattr(benchmark, 'train_stream'), f"{benchmark_type} benchmark for {dataset_name} does not have train_stream"
    assert hasattr(benchmark, 'test_stream'), f"{benchmark_type} benchmark for {dataset_name} does not have test_stream"
    
    # Check if the streams have the correct number of experiences
    assert len(benchmark.train_stream) == 5, f"Train stream for {dataset_name} {benchmark_type} benchmark does not have 5 experiences"
    
    # Check the first experience
    first_experience = benchmark.train_stream[0]
    assert hasattr(first_experience, 'dataset'), f"Experience for {dataset_name} {benchmark_type} benchmark does not have dataset"
    assert len(first_experience.dataset) > 0, f"Dataset for {dataset_name} {benchmark_type} benchmark experience is empty"
    
    # Check if the experience dataset returns the correct data format
    first_item = first_experience.dataset[0]
    assert isinstance(first_item, tuple), f"Dataset item for {dataset_name} {benchmark_type} benchmark is not a tuple"
    assert len(first_item) == 2, f"Dataset item for {dataset_name} {benchmark_type} benchmark does not have 2 elements"
    
    image, label = first_item
    assert image.shape == (3, 224, 224), f"Image shape for {dataset_name} {benchmark_type} benchmark is incorrect"
    assert isinstance(label, int), f"Label for {dataset_name} {benchmark_type} benchmark is not an integer"

def test_stream51_benchmark(tmp_path):
    train_transform, test_transform = get_transforms("stream51")
    
    benchmark = get_cl_benchmark(
        "stream51", 
        str(tmp_path), 
        train_transform, 
        test_transform, 
        "nc",  # Stream51 only supports class-incremental scenario
        n_experiences=5
    )
    
    assert hasattr(benchmark, 'train_stream'), "Stream51 benchmark does not have train_stream"
    assert hasattr(benchmark, 'test_stream'), "Stream51 benchmark does not have test_stream"
    
    # Check if the streams have experiences
    assert len(benchmark.train_stream) > 0, "Train stream for Stream51 benchmark is empty"
    
    # Check the first experience
    first_experience = benchmark.train_stream[0]
    assert hasattr(first_experience, 'dataset'), "Experience for Stream51 benchmark does not have dataset"
    assert len(first_experience.dataset) > 0, "Dataset for Stream51 benchmark experience is empty"
    
    # Check if the experience dataset returns the correct data format
    first_item = first_experience.dataset[0]
    assert isinstance(first_item, tuple), "Dataset item for Stream51 benchmark is not a tuple"
    assert len(first_item) == 2, "Dataset item for Stream51 benchmark does not have 2 elements"
    
    image, label = first_item
    assert image.shape == (3, 224, 224), "Image shape for Stream51 benchmark is incorrect"
    assert isinstance(label, int), "Label for Stream51 benchmark is not an integer"