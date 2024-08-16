import torch
from monai.apps import DecathlonDataset
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)


def brats(root, workers, section, roi_size, spacing=(1, 1, 1), amp=False):
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    if section not in ["training", "validation"]:
        raise ValueError(f"{section} is not correct input,training|validation.")

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear")),
            CenterSpatialCropd(keys=["image"], roi_size=roi_size),
            ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
            EnsureTyped(keys="image", dtype=compute_dtype),
        ]
    )
    # section="training" or "validation"
    dataset = DecathlonDataset(
            root_dir=root,
            task="Task01_BrainTumour",
            section=section,  # validation
            cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
            num_workers=workers,
            download=False,  # Set download to True if the datasets hasnt been downloaded yet
            seed=0,
            transform=transforms,
        )

    return dataset
