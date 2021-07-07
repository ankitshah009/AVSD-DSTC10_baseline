import numpy as np
# import tensorboardX as tensorboard
import torch
from torch.utils import tensorboard as tensorboard
from torch.utils.data import DataLoader

from datasets.captioning_dataset import AVSD10Dataset
from epoch_loops.captioning_epoch_loops import (greedy_decoder, teacher_forced_decoder, save_model, training_loop, validation_1by1_loop, validation_next_word_loop)
from loss.label_smoothing import LabelSmoothing
from model.captioning_module import BiModalTransformer, Transformer
from utilities.captioning_utils import timer


def train_cap(cfg):
    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # preventing PyTorch from allocating memory on the default device (cuda:0) when the desired 
    # cuda id for training is not 0.
    torch.cuda.set_device(cfg.device_ids[0])

    exp_name = cfg.exp_name

    train_dataset = AVSD10Dataset(cfg, 'train', get_full_feat=False)
    val_dataset = AVSD10Dataset(cfg, 'val', get_full_feat=False)
    
    # make sure that DataLoader has batch_size = 1!
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.dont_collate)
    val_loader = DataLoader(val_dataset, collate_fn=val_dataset.dont_collate)

    if cfg.pretrained_cap_model_path is not None:
        cap_model_cpt = torch.load(cfg.pretrained_cap_model_path, map_location='cpu')
        model_cfg = cap_model_cpt['config']
    else:
        cap_model_cpt = None
        model_cfg = cfg

    if cfg.modality == 'audio_video':
        model = BiModalTransformer(model_cfg, train_dataset)
    elif cfg.modality in ['video', 'audio']:
        model = Transformer(model_cfg, train_dataset)

    criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)
    
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.lr, (cfg.beta1, cfg.beta2), cfg.eps,
                                     weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), cfg.lr, cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    
    if cfg.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_reduce_factor, patience=cfg.lr_patience
        )
    else:
        scheduler = None

    model.to(torch.device(cfg.device))
    model = torch.nn.DataParallel(model, cfg.device_ids)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')

    if cap_model_cpt is not None:
        model.load_state_dict(cap_model_cpt['model_state_dict'])

    if cfg.to_log:
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    # keeping track of the best model 
    best_metric = 0
    # "early stopping" thing
    num_epoch_best_metric_unchanged = 0

    for epoch in range(cfg.epoch_num):
        print(f'The best metrict was unchanged for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch+cfg.early_stop_after-num_epoch_best_metric_unchanged}')
        print(f'Started @ {cfg.curr_time}; Current timer: {timer(cfg.curr_time)}')
        
        # stop training if metric hasn't been changed for cfg.early_stop_after epochs
        if num_epoch_best_metric_unchanged == cfg.early_stop_after:
            break

        # train
        training_loop(cfg, model, train_loader, criterion, optimizer, epoch, TBoard)
        # validation (next word)
        val_loss = validation_next_word_loop(
            cfg, model, val_loader, greedy_decoder, criterion, epoch, TBoard, exp_name
        )
        if scheduler is not None:
            scheduler.step(val_loss)

        # validation (1-by-1 word)
        if epoch >= cfg.one_by_one_starts_at:
            val_metrics, duration = validation_1by1_loop(
                cfg, model, val_loader, teacher_forced_decoder, epoch, TBoard
            )
            if cfg.to_log:
                for metric, score in val_metrics.items():
                    TBoard.add_scalar('val/metrics/' + metric, score * 100, epoch)
                TBoard.add_scalar('val/duration_of_1by1', duration / 60, epoch)
                TBoard.add_scalar('val/loss', val_loss, epoch)
                # saving the model if it is better than the best so far
                if best_metric < val_metrics['Bleu_4']:
                    best_metric = val_metrics['Bleu_4']
                    save_model(cfg, epoch, model, optimizer, val_loss,
                               val_metrics, train_dataset.trg_voc_size)
                    # reset the early stopping criterion
                    num_epoch_best_metric_unchanged = 0
                else:
                    num_epoch_best_metric_unchanged += 1
            # Output the results
            print('-' * 25)
            for metric, score in val_metrics.items():
                print('| %s: %2.4f' % (metric, 100 * score))
            print('-' * 25)
            print('duration_of_1by1:', duration / 60, epoch)

    print(f'{cfg.curr_time}')
    print(f'best_metric: {best_metric}')
    if cfg.to_log:
        TBoard.close()
