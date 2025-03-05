import json
import urllib.request

# URL to ImageNet class index file
imagenet_class_index_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

def load_imagenet_labels():
    """Load ImageNet class labels from the official URL."""
    try:
        with urllib.request.urlopen(imagenet_class_index_url) as response:
            class_index = json.loads(response.read().decode())
        return {int(k): v[1] for k, v in class_index.items()}  # Return dictionary mapping index to label
    except Exception as e:
        print(f"Error loading ImageNet labels: {e}")
        return {}

def get_imagenet_label(class_idx):
    """Retrieve the ImageNet label given a class index."""
    labels = load_imagenet_labels()
    return labels.get(class_idx, "Unknown class index")

# Example usage
if __name__ == "__main__":
    class_idx = int(input("Enter ImageNet class index (0-999): "))
    label = get_imagenet_label(class_idx)
    print(f"Class {class_idx}: {label}")
