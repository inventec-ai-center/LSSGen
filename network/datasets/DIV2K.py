from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

class DIV2KDataset(Dataset):
    def __init__(
        self, 
        image_dir: Path | str, 
        test_run: bool = False,
        transform=None
    ):
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)

        self.image_dir = image_dir

        self.transform = transform

        self.image_files = []
        for ext in ('*.jpg', '*.png'):
            self.image_files.extend(image_dir.glob(ext))

        if test_run:
            self.image_files = self.image_files[:10]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "name": image_path.stem}
    
class DIV2KScalingDataset(Dataset):
    def __init__(
        self, 
        image_dir: Path | str, 
        test_run: bool = False,
        resolution: int | tuple[int, int] = 256,
        target_resolution: int | tuple[int, int] = 512,
        transform=None,
        target_type: str = "origin" # "origin", "nearest"
    ):
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)

        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        if isinstance(target_resolution, int):
            target_resolution = (target_resolution, target_resolution)

        self.image_dir = image_dir
        self.resolution = resolution
        self.target_resolution = target_resolution
        self.target_type = target_type

        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomCrop(
                    target_resolution if target_type == "origin" else resolution, 
                    pad_if_needed=True, 
                    padding_mode="reflect"
                ),
            ])
        else:
            self.transform = transform

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.resizer = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST)
        self.resizer_back = transforms.Resize(target_resolution, interpolation=transforms.InterpolationMode.NEAREST)

        self.image_files = []
        for ext in ('*.jpg', '*.png'):
            self.image_files.extend(image_dir.glob(ext))

        if test_run:
            self.image_files = self.image_files[:10]


    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        
        image = self.transform(image)
        tar_image = image if self.target_type == "origin" else self.resizer_back(image)
        image = self.resizer(image) if self.target_type == "origin" else image

        image = self.to_tensor(image)
        tar_image = self.to_tensor(tar_image)

        return {
            "image": image, 
            "target": tar_image,
            "name": image_path.stem
        }