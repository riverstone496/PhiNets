from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch
from .lr_scheduler import LR_Scheduler
from .lion import Lion

def get_optimizer(name, model, lr, momentum, weight_decay, beta2=0.95, preconditioning_compute_steps=30, train_eps=1e-8):

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay, eps=train_eps)
    elif name == 'lion':
        optimizer = Lion(parameters, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay)
    elif name == 'lars_simclr': # Careful
        optimizer = LARS_simclr(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(
            torch.optim.SGD(
                parameters,
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            ),
            trust_coefficient=0.001, 
            clip=False
        )
    else:
        raise NotImplementedError
    return optimizer



