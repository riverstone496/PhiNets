import os
import torch
import torch.nn.functional as F 
from utils import *
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import knn_monitor
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
import argparse
import wandb
from tools.weight_comp import calculate_similarity

def main(device, args):
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train_batch_size,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=True,
            memory=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train_batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train_batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args,simplicial_model=args.train.simplicial_model).to(device)
    model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train_optimizer_name, model, 
        lr=args.train_base_lr*args.train_batch_size*args.accumulation_steps/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train_weight_decay,
        preconditioning_compute_steps=args.preconditioning_compute_steps,
        beta2=args.train_beta2,
        train_eps=args.train_eps)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train_warmup_epochs, args.train.warmup_lr*args.accumulation_steps*args.train_batch_size/256, 
        args.train.num_epochs, args.train_base_lr*args.accumulation_steps*args.train_batch_size/256, args.train.final_lr*args.accumulation_steps*args.train_batch_size/256, 
        int(len(train_loader)//args.accumulation_steps),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    #logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 
    update_iters = 0
    # Start training
    #global_progress = tqdm(range(0, args.train_epochs), desc=f'Training')
    model.zero_grad()
    if 'phinetmom' in args.model_name:
        log_dict = {'epoch':0, 'update_iters':0}
        log_dict.update(calculate_similarity(model.module))
        wandb.log(log_dict)
    for epoch in range(0,args.train_epochs):
        model.train()
        #local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2,images_ori), labels) in enumerate(train_loader):

            #print(images1.shape)
            
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True),images_ori.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            loss_cos = data_dict['loss_cos'].mean()
            loss_mse = data_dict['loss_mse'].mean()
            loss.backward()
            if update_iters%args.accumulation_steps==args.accumulation_steps-1:
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
            current_lr = lr_scheduler.get_lr()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            update_iters += 1
            
            if args.dataset_name=='cifar5m' and args.train.knn_monitor and epoch % args.train.knn_interval == 0 and idx % 100000 == 0: 
                accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, epoch, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress, device=args.device) 
                print('start knn', idx, '/', len(train_loader.dataset), accuracy)
                log_dict = {"acc": accuracy, "update_iters":update_iters, "loss": loss, "loss_cos": loss_cos, "loss_mse": loss_mse, "epoch":epoch, "lr":current_lr}
                if 'phinetmom' in args.model_name:
                    with torch.no_grad():
                        log_dict.update(calculate_similarity(model.module))
                    #    slow_accuracy = knn_monitor(model.module.slow_encoder, memory_loader, test_loader, epoch, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress, device=args.device) 
                    #log_dict['slow_acc']=slow_accuracy
                wandb.log(log_dict)
            #local_progress.set_postfix(data_dict)
            #logger.update_scalers(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, epoch, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress, device=args.device) 
            print(accuracy)
            log_dict = {"acc": accuracy, "loss": loss, "loss_cos": loss_cos, "loss_mse": loss_mse, "epoch":epoch, "lr":current_lr}
            if 'phinetmom' in args.model_name:
                with torch.no_grad():
                    log_dict.update(calculate_similarity(model.module))
                #    slow_accuracy = knn_monitor(model.module.slow_encoder, memory_loader, test_loader, epoch, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress, device=args.device) 
                #log_dict['slow_acc']=slow_accuracy
            wandb.log(log_dict)
        state_dict=model.module.state_dict()
        if args.dataset_name=='cifar5m' and (not args.eval_last):
            linear_eval(args, state_dict, global_epoch=epoch)
            if 'phinetmom' in args.model_name:
                linear_eval(args, state_dict, global_epoch=epoch, slow=True)
    if args.dataset_name!='cifar5m' or args.eval_last:
        linear_eval(args, state_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', type=str, help="xxx.yaml",default='./configs/hiposiam_cifar_seed0.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--ckpt_dir', type=str, default='./result/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')

    parser.add_argument('--model_name', type=str, default='phinetmom_aug')
    parser.add_argument('--backbone', type=str, default='resnet18_cifar_variant1')
    parser.add_argument('--proj_layers', type=int, default=2)

    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--image_size', type=int, default=32)

    parser.add_argument('--mse_loss_ratio', type=float, default=1)
    parser.add_argument('--ori_loss_ratio', type=float, default=0)
    parser.add_argument('--beta_enc', type=float, default=0.99)

    parser.add_argument('--train_optimizer_name', type=str, default='sgd')
    parser.add_argument('--train_base_lr', type=float, default=0.03)
    parser.add_argument('--train_beta2', type=float, default=0.95)
    parser.add_argument('--train_eps', type=float, default=1e-8)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--train_weight_decay', type=float, default=0.0005)
    parser.add_argument('--train_epochs', type=int, default=800)
    parser.add_argument('--train_warmup_epochs', type=float, default=10)
    parser.add_argument('--seed',type=int, default=1234)
    parser.add_argument('--preconditioning_compute_steps', type=int, default=30)
    parser.add_argument('--eval_num_epochs', type=int, default=100)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.2)
    
    parser.add_argument('--use_timm', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--eval_last', action='store_true', default=False)
    
    args = parser.parse_args()
    if args.model_name == 'simsiam':
        args.model_name = 'hiposiamlatent'
        args.mse_loss_ratio = 0
        args.ori_loss_ratio = 0
    args_out = get_args(args)
    wandb.init(
    # set the wandb project where this run will be logged
    entity=os.environ.get('WANDB_ENTITY', None),
    project=os.environ.get('WANDB_PROJECT', None),
    # track hyperparameters and run metadata
    config=vars(args_out).copy()
    )

    print(args.model)
    print(args_out.train.simplicial_model)

    main(device=args_out.device, args=args_out)

    wandb.finish()
    print('Finish')












