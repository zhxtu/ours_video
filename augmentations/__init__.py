import logging

from augmentations.augmentations import (
    RandomCrop,
    RandomResizedCrop,
    RandomHorizontallyFlip,
    RandomVerticallyFlip,
    Scale,
    RandomScale,
    RandomRotate,
    RandomTranslate,
    CenterCrop,
    Compose,
    ColorJitter,
    ColorNorm,
    Totensor
)

logger = logging.getLogger("ptsemseg")

key2aug = {
    "rcrop": RandomCrop,
    "rscrop": RandomResizedCrop,
    "hflip": RandomHorizontallyFlip,
    "vflip": RandomVerticallyFlip,
    "scale": RandomScale,
    "rotate": RandomRotate,
    "translate": RandomTranslate,
    "ccrop": CenterCrop,
    "jitter": ColorJitter,
    "colornorm": ColorNorm,
    "totensor":Totensor
}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        # logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)