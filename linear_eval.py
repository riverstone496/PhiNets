import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
import wandb

def main(args, state_dict=None, global_epoch=None, slow=False):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), 
            train=True, 
            memory=True,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )


    model = get_backbone(args.backbone, use_timm=args.use_timm, pretrained=args.pretrained)
    if args.dataset_name == 'cifar10':
        out_features = 10
    elif args.dataset_name == 'cifar100':
        out_features = 100
    elif args.dataset_name == 'cifar5m':
        out_features = 10
    elif args.dataset_name == 'stl10':
        out_features = 10
    elif args.dataset_name == 'ImageNet':
        out_features = 1000
    classifier = nn.Linear(in_features=model.output_dim, out_features=out_features, bias=True).to(args.device)

    if state_dict is None:
        assert args.eval_from is not None
        state_dict = torch.load(args.eval_from, map_location='cpu')['state_dict']
    if slow:
        msg = model.load_state_dict({k[15:]:v for k, v in state_dict.items() if k.startswith('slow_encoder.0.')}, strict=True)
    else:
        msg = model.load_state_dict({k[9:]:v for k, v in state_dict.items() if k.startswith('backbone.')}, strict=True)
    
    # print(msg)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier, 
        lr=args.eval.base_lr*args.eval.batch_size/256, 
        momentum=args.eval.optimizer.momentum, 
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/256, 
        args.eval_num_epochs, args.eval.base_lr*args.eval.batch_size/256, args.eval.final_lr*args.eval.batch_size/256, 
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.eval_num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval_num_epochs}', disable=True)
        
        for idx, (images, labels) in enumerate(local_progress):

            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})
    if slow:
        wandb.log({
        'epoch':global_epoch,
        'local_epoch':epoch,
        "eval/slow_loss":loss_meter.val,
        "eval/slow_loss_avg":loss_meter.avg,
        "eval/lr":lr,
    })
    else:
        wandb.log({
            'epoch':global_epoch,
            'local_epoch':epoch,
            "eval/loss":loss_meter.val,
            "eval/loss_avg":loss_meter.avg,
            "eval/lr":lr,
        })

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(images.to(args.device))
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0])
    print(f'Accuracy = {acc_meter.avg*100:.2f}')
    if slow:
        wandb.log({
            'epoch':global_epoch,
            'eval/Slow_Accuracy':acc_meter.avg*100,
        })
    else:
        wandb.log({
            'epoch':global_epoch,
            'eval/Accuracy':acc_meter.avg*100,
        })



if __name__ == "__main__":
    main(args=get_args())
















