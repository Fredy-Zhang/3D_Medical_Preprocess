import torch
from monai.apps import DecathlonDataset
from monai.transforms import (
    AddChanneld,
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    ToTensord
)


def colon(root, workers, section, roi_size, spacing=(1, 1, 1), amp=False):
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    if section not in ["training", "validation"]:
        raise ValueError(f"{section} is not correct input,training|validation.")

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-30, a_max=165.82,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Resized(keys=["image"], spatial_size=roi_size, mode="trilinear", anti_aliasing=True),
            ToTensord(keys=["image"]),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    # section="training" or "validation"
    dataset = DecathlonDataset(
            root_dir=root,
            task="Task10_Colon",
            section=section,  # validation
            cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
            num_workers=workers,
            download=False,  # Set download to True if the datasets hasnt been downloaded yet
            seed=0,
            transform=transforms,
        )

    return dataset
