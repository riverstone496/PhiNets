from .simsiam import SimSiam
from .trisiam import TriSiamLatent
from .phinet import PhiNet
from .eyephinet import EyePhiNet
from .phinet_mom import PhiNetMom
from .phinet_mom_cos import PhiNetMomCos
from .phinet_cos import PhiNetCos
from .phinet_slow import PhiNetSlow
from .phinet_slow_cos import PhiNetSlowCos
from .eye_phinet_mom_cos import EyePhiNetMomCos
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2, wideresnet28_cifar_variant1

def get_backbone(backbone, castrate=True, use_timm=False, pretrained=False):
    if use_timm:
        import timm
        bname = backbone
        backbone = timm.create_model(backbone, pretrained=pretrained)
        if castrate and 'vit' in bname:
            backbone.output_dim = backbone.head.in_features
            backbone.head = torch.nn.Identity()
        if castrate and 'resnet' in bname:
            backbone.output_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()
    else:
        backbone = eval(f"{backbone}()")

        if castrate:
            backbone.output_dim = backbone.fc.in_features
            backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg,simplicial_model='DCT'):    

    if model_cfg.model_name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinet':
        model =  PhiNet(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinet_aug':
        model =  PhiNet(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinetcos':
        model =  PhiNetCos(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'eyephinet':
        model =  EyePhiNet(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinetmom':
        model =  PhiNetMom(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio, beta=model_cfg.beta_enc)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinetmom_aug':
        model =  PhiNetMom(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio, beta=model_cfg.beta_enc)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinetmomcos':
        model =  PhiNetMomCos(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio, beta=model_cfg.beta_enc)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'eyephinetmomcos':
        model =  EyePhiNetMomCos(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio, beta=model_cfg.beta_enc)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinetmomcos_aug':
        model =  PhiNetMomCos(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio, beta=model_cfg.beta_enc)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinetslow':
        model =  PhiNetSlow(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio, beta=model_cfg.beta_enc)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'phinetslowcos':
        model =  PhiNetSlowCos(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio, beta=model_cfg.beta_enc)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.model_name == 'trisiam':
        model =  TriSiamLatent(backbone=get_backbone(model_cfg.backbone, use_timm=model_cfg.use_timm, pretrained = model_cfg.pretrained), mse_loss_ratio=model_cfg.mse_loss_ratio, ori_loss_ratio = model_cfg.ori_loss_ratio)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    else:
        raise NotImplementedError
    return model






