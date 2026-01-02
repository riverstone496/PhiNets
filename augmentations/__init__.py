from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .phinet_aug import PhiNetTransform
from .trisiam_aug import TriSiamTransform
def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None):

    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'hiposiam':
            augmentation = PhiNetTransform(image_size)
        elif name == 'hiposiampred':
            augmentation = PhiNetTransform(image_size)
        elif name == 'trisiam' or name == 'phinet_aug' or name == 'phinetmom_aug' or name=='phinetmomcos_aug':
            augmentation = TriSiamTransform(image_size)
        elif name == 'hiposiamlatent' or 'phinet' in name or 'rmsimsiam' in name or 'byol' in name or 'barlowtwins' in name or 'dino' in name or name == 'moco':
            augmentation = PhiNetTransform(image_size)
        elif name == 'hiposiamlatent_rotation':
            augmentation = PhiNetTransform(image_size, apply_rotation=True)
        elif name == 'hiposiamlatent_noflip':
            augmentation = PhiNetTransform(image_size, apply_flip=False)
        elif name == 'sidae':
            augmentation = PhiNetTransform(image_size)
        elif name == 'simsiamrank1':
            augmentation = SimSiamTransform(image_size)
        elif name == 'simsiamTWD':
            augmentation = SimSiamTransform(image_size)
        elif name == 'dino':
            augmentation = SimSiamTransform(image_size)
        # elif name == 'byol':
        #     augmentation = BYOL_transform(image_size)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








