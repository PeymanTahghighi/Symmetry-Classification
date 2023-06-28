import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

EPSILON = 1e-5
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu';
BATCH_SIZE = 1;
IMAGE_SIZE =880
WARMUP_EPOCHS = 10;
D_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\Diaphragm'
ST_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Sternum'
SP_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\TipsSP'
SR_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs'
IMAGE_DATASET_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final'

pre_transforms = A.Compose([
    #A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.Sequential([
            A.RandomGamma((200.0,500.0), p=1.0),
            A.GaussNoise((50,200), mean = 10, p=0.75)
            ]),
        # A.Sequential([
        #     A.RandomGamma((200.0,500.0), p=1.0),
        #     A.GaussNoise((50,200), mean = 10, p=0.75)
        # ]),
        
    ], p = 1.0),
])

hist_flip_resize_transforms = A.Compose(
[
    #A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), always_apply=True, p = 1.0),
    #A.HorizontalFlip(p=0.5),
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
],
additional_targets={'mask': 'mask'}
)

to_tensor_transforms = A.Compose(
[
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)

valid_transforms = A.Compose(
    [
    #A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), always_apply=True, p = 1.0),
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ]
)


overexposure_transforms = A.Compose([
    A.OneOf([
        A.Sequential([
            A.RandomGamma((200.0,500.0), p=1.0),
            A.GaussNoise((50,200), mean = 10, p=0.75)
            ]),      
    ], p = 1.0),
])

underexposure_transforms = A.Compose([
    #A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.Sequential([
            A.RandomGamma((200.0,500.0), p=1.0),
            A.GaussNoise((50,200), mean = 10, p=0.75)
        ]),
        
    ], p = 1.0),
])

train_transforms = A.Compose(
[
    A.HorizontalFlip(p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), always_apply=True, p = 1.0),
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)
