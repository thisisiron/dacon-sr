from PIL import Image
import albumentations as A

from configuration.const import logger


def get_transforms(augment_type='default', image_norm='imagenet'):

    if image_norm == 'imagenet':
        logger.info('Using ImageNet mean/std Norm.')
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        logger.info('Using Zero-centerd Norm.')  # -1 ~ 1
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if augment_type == 'default':
        train_transform = A.Compose([A.HorizontalFlip(),
                                     # A.VerticalFlip(),
                                     A.RandomRotate90(),
                                     A.Normalize(mean=mean, std=std)],
                                     additional_targets={'image2':'image'})

    return train_transform
