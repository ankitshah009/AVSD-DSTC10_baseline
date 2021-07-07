import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.captioning_dataset import AVSD10Dataset
from epoch_loops.captioning_epoch_loops import (teacher_forced_decoder, validation_1by1_loop)
from model.captioning_module import BiModalTransformer, Transformer
from utilities.captioning_utils import timer


def eval_cap(cfg):
    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(cfg.device_ids[0])

    test_dataset = AVSD10Dataset(cfg, 'test', get_full_feat=False)
    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.dont_collate)

    cap_model_cpt = torch.load(cfg.pretrained_cap_model_path, map_location='cpu')
    model_cfg = cap_model_cpt['config']
    if cfg.modality == 'audio_video':
        model = BiModalTransformer(model_cfg, test_dataset)
    elif cfg.modality in ['video', 'audio']:
        model = Transformer(model_cfg, test_dataset)

    model.to(torch.device(cfg.device))
    model = torch.nn.DataParallel(model, cfg.device_ids)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Parameters: {param_num / 1000000} Mil.')
    model.load_state_dict(cap_model_cpt['model_state_dict'])

    # evaluation (1-by-1 word)
    metrics, duration = validation_1by1_loop(
        cfg, model, test_loader, teacher_forced_decoder, 0, None
    )
    print ('-' * 25)
    for metric, score in metrics.items():
        print ('| %s: %2.4f' % (metric, 100 * score))
    print ('-' * 25)
