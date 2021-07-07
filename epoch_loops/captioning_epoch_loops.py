import os
import json
from tqdm import tqdm
import torch
from time import time

from model.masking import mask
from evaluation.evaluate import AVSD_eval
from utilities.captioning_utils import HiddenPrints, get_lr


def greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    assert model.training is False, 'call model.eval first'

    with torch.no_grad():
        
        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            masks = make_masks(feature_stacks, trg, modality, pad_idx)
            preds = model(feature_stacks, trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg


def teacher_forced_decoder(model, batch, max_len, start_idx, end_idx, pad_idx, modality,
                           context_start_idx=None, context_end_idx=None, last_only=False):
    assert model.training is False, 'call model.eval first'
    feature_stacks = batch['feature_stacks']
    caption_idx = batch['caption_data'].caption
    start_pos, end_pos = get_context_positions(caption_idx, context_start_idx, context_end_idx)
    sources = []
    targets = []
    attweights = []
    with torch.no_grad():

        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')

        for t in range(start_pos.size(1)):
            # store source information
            max_src_context_len = int(torch.max(end_pos[:, t] - start_pos[:, t]))
            src = torch.full((B, max_src_context_len), end_idx, dtype=torch.long, device=device)
            for b, (s, e) in enumerate(zip(start_pos[:, t], end_pos[:, t])):
                if e >= 0:
                    src[b, :e-s] = caption_idx[b, s:e]
            sources.append(src)
            # prepare context used for teacher forcing
            max_context_len = int(torch.max(end_pos[:, t])) + 1
            trg = torch.full((B, max_context_len), pad_idx, dtype=torch.long, device=device)
            completeness_mask = torch.zeros(B, 1).byte().to(device)
            current_position = torch.zeros(B, dtype=torch.long, device=device)
            for b, e in enumerate(end_pos[:, t]):
                if e >= 0:
                    trg[b, :e+1] = caption_idx[b, :e+1]
                    current_position[b] = e
                else: # no more sentences
                    completeness_mask[b] = 1
            # greedy decoding
            out = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
            pad_idx_ = torch.full((B, 1), pad_idx, dtype=torch.long, device=device)
            end_idx_ = torch.full((B, 1), end_idx, dtype=torch.long, device=device)
            attw = None
            batch_indices = torch.arange(B, dtype=torch.long, device=device)
            while (out.size(-1) <= max_len) and (not completeness_mask.all()):
                masks = make_masks(feature_stacks, trg, modality, pad_idx)
                preds = model(feature_stacks, trg, masks)
                preds[:, :, 0] = float('-inf')  # suppress UNK
                next_word = torch.where(completeness_mask==0,
                                        preds[batch_indices, current_position].max(dim=-1)[1].unsqueeze(1),
                                        end_idx_)
                out = torch.cat([out, next_word], dim=-1)
                trg = torch.cat([trg, pad_idx_], dim=-1)
                aw = model.module.enc_attw()[batch_indices, current_position].unsqueeze(1)
                attw = torch.cat([attw, aw], dim=1) if attw is not None else aw
                current_position += 1
                trg[batch_indices, current_position] = next_word[:, 0]
                completeness_mask = completeness_mask | torch.eq(next_word, end_idx_).byte()
            targets.append(out)
            attweights.append(attw)
    return sources, targets, attweights


def save_model(cfg, epoch, model, optimizer, val_loss_value,
               val_metrics, trg_voc_size):
    
    dict_to_save = {
        'config': cfg,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss_value,
        'val_metrics': val_metrics,
        'trg_voc_size': trg_voc_size,
    }
    
    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    os.makedirs(cfg.model_checkpoint_path, exist_ok=True)
    
#     path_to_save = os.path.join(cfg.model_checkpoint_path, f'model_e{epoch}.pt')
    path_to_save = os.path.join(cfg.model_checkpoint_path, f'best_cap_model.pt')
    torch.save(dict_to_save, path_to_save)


def make_masks(feature_stacks, captions, modality, pad_idx):
    masks = {}

    if modality == 'video':
        if captions is None:
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
    elif modality == 'audio':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        else:
            masks['A_mask'], masks['C_mask'] = mask(feature_stacks['audio'][:, :, 0], captions, pad_idx)
    elif modality == 'audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
    elif modality == 'subs_audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
        masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        masks['S_mask'] = mask(feature_stacks['subs'], None, pad_idx)

    return masks


def get_context_positions(caption_idx, context_start_idx, context_end_idx):
    """ obtain context end positions based on context_start_idx and context_end_idx
    """
    B, L = caption_idx.size()
    cap = caption_idx.view(-1)
    positions = torch.arange(len(cap), device=cap.device)
    start_positions = positions[cap == context_start_idx]
    end_positions = positions[cap == context_end_idx]
    assert len(start_positions) == len(end_positions)
    start_pos_list = [[] for _ in range(B)]
    end_pos_list = [[] for _ in range(B)]
    for s, e in zip(start_positions.tolist(), end_positions.tolist()):
        start_pos_list[s // L].append(s % L)
        end_pos_list[e // L].append(e % L)
    max_npos = max([len(pl) for pl in start_pos_list])
    start_pos = torch.full((B, max_npos), -1, dtype=torch.long, device=cap.device)
    end_pos = torch.full((B, max_npos), -1, dtype=torch.long, device=cap.device)
    for b in range(B):
        start_pos[b, :len(start_pos_list[b])] = torch.tensor(start_pos_list[b], dtype=torch.long)
        end_pos[b, :len(end_pos_list[b])] = torch.tensor(end_pos_list[b], dtype=torch.long)
    return start_pos, end_pos


def get_context_masked_target(caption_idx, context_start_idx, context_end_idx, end_idx, pad_idx):
    """ replace token_ids between context_start_idx and context_end_idx with pad_idx, and
        also replace context_start_idx with end_idx unless it is in the beginning,
        e.g. in QA dialog '<s> Q: w1 w2 w3 A: w4 w5 Q: w6 w7 A: w8 w9 </s>' is converted to
        '- - - - - w4 w5 </s> - - - w8 w9 </s>',
        where 'Q:', 'A:', and '-' represent context_start, context_end, and pad tokens.
    """
    caption_idx_y = caption_idx[:, 1:]
    if context_start_idx is not None and context_end_idx is not None:
        L = caption_idx_y.size(1)
        cap = torch.clone(caption_idx_y).view(-1)
        positions = torch.arange(len(cap), device=cap.device)
        context_start_positions = positions[cap == context_start_idx]
        context_end_positions = positions[cap == context_end_idx]
        assert len(context_start_positions) == len(context_end_positions)
        for i in range(len(context_start_positions)):
            cap[context_start_positions[i]] = pad_idx if context_start_positions[i] % L == 0 else end_idx
            cap[context_start_positions[i] + 1 : context_end_positions[i] + 1] = pad_idx
        return cap.view(caption_idx_y.size())
    else:
        return caption_idx_y


def training_loop(cfg, model, loader, criterion, optimizer, epoch, TBoard):
    model.train()
    train_total_loss = 0
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.exp_name}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        caption_idx = batch['caption_data'].caption
        caption_idx_x = caption_idx[:, :-1]
        caption_idx_y = get_context_masked_target(caption_idx,
                                                loader.dataset.context_start_idx,
                                                loader.dataset.context_end_idx,
                                                loader.dataset.end_idx,
                                                loader.dataset.pad_idx)
        masks = make_masks(batch['feature_stacks'], caption_idx_x, cfg.modality, loader.dataset.pad_idx)
        pred = model(batch['feature_stacks'], caption_idx_x, masks)
        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        loss = criterion(pred, caption_idx_y) / n_tokens
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)
            

def validation_next_word_loop(cfg, model, loader, decoder, criterion, epoch, TBoard, exp_name):
    model.eval()
    val_total_loss = 0
    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    progress_bar_name = f'{cfg.exp_name}: {phase:<5} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        caption_idx = batch['caption_data'].caption
        caption_idx_x = caption_idx[:, :-1]
        caption_idx_y = get_context_masked_target(caption_idx,
                                                loader.dataset.context_start_idx,
                                                loader.dataset.context_end_idx,
                                                loader.dataset.end_idx,
                                                loader.dataset.pad_idx)
        masks = make_masks(batch['feature_stacks'], caption_idx_x, cfg.modality, loader.dataset.pad_idx)

        with torch.no_grad():
            pred = model(batch['feature_stacks'], caption_idx_x, masks)
            n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            loss = criterion(pred, caption_idx_y) / n_tokens
            val_total_loss += loss.item()
            
    val_total_loss_norm = val_total_loss / len(loader)
    if TBoard is not None:
        TBoard.add_scalar('debug/val_loss_epoch', val_total_loss_norm, epoch)

    return val_total_loss_norm

def validation_1by1_loop(cfg, model, loader, decoder, epoch, TBoard):
    start_timer = time()
    
    # init the dict with results and other technical info
    predictions = {
        'dialogs': [],
    }
    model.eval()
    loader.dataset.update_iterator()
    
    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    context_start_idx = loader.dataset.context_start_idx
    context_end_idx = loader.dataset.context_end_idx
    phase = loader.dataset.phase
    # feature_names = loader.dataset.feature_names
    
    reference_paths = cfg.reference_paths
    progress_bar_name = f'{cfg.exp_name}: {phase} 1by1 {epoch} @ {cfg.device}'
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        ### PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE
        ints_stack_list = decoder(
            model, batch, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality,
            context_start_idx=context_start_idx, context_end_idx=context_end_idx
        )
        input_lengths = torch.sum(mask(batch['feature_stacks']['rgb'][:, :, 0], None, pad_idx), dim=-1).cpu().view(-1)
        list_of_lists_with_filtered_sentences = [[] for _ in range(len(ints_stack_list[0][0]))]
        for ints_stack1, ints_stack2, attw_stack in zip(ints_stack_list[0], ints_stack_list[1], ints_stack_list[2]):
            ints_stack1 = ints_stack1.cpu().numpy()  # what happens here if I use only cpu?
            ints_stack2 = ints_stack2.cpu().numpy()  # what happens here if I use only cpu?
            attw_stack = attw_stack.cpu()
            # transform integers into strings
            list_of_lists_with_strings1 = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack1]
            list_of_lists_with_strings2 = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack2]
            ### FILTER PREDICTED TOKENS
            # initialize the list to fill it using indices instead of appending them

            for b, (strings1, strings2, attw) in enumerate(zip(list_of_lists_with_strings1, list_of_lists_with_strings2, attw_stack)):
                # remove starting token and everything after ending token
                if len(strings1) > 0:
                    strings1 = strings1[1:]  # skip Q:
                else:
                    continue  # no more turns
                if len(strings2) > 0:
                    strings2 = strings2[1:]  # skip <s>
                try:
                    first_entry_of_eos1 = strings1.index('</s>')
                    strings1 = strings1[:first_entry_of_eos1]
                except ValueError:
                    pass
                try:
                    first_entry_of_eos2 = strings2.index('</s>')
                    strings2 = strings2[:first_entry_of_eos2]
                except ValueError:
                    pass
                if len(strings1) == 0:
                    continue
                sentence1 = ' '.join(strings1)
                sentence2 = ' '.join(strings2)

                # find regions for reasoning with attention weights over visual feature frames
                # TODO: detection of multiple regions and audio features should be considered
                ilen = input_lengths[b]
                attw_mean = torch.mean(attw[:len(strings2)], dim=0)[:ilen]
                frame_indices = torch.arange(ilen, dtype=torch.float) / ilen  # relative frame positions
                frame_mean = float((frame_indices * attw_mean).sum())  # expected value of attended frame
                frame_std = float(((frame_indices - frame_mean) ** 2 * attw_mean).sum().sqrt())
                start_time = max(0.0, (frame_mean - cfg.region_std_coeff * frame_std))
                end_time = min(1.0, (frame_mean + cfg.region_std_coeff * frame_std))
                list_of_lists_with_filtered_sentences[b].append((sentence1, sentence2, (start_time, end_time)))

        ### ADDING RESULTS TO THE DICT WITH RESULTS
        for video_id, start, end, sents in zip(batch['video_ids'], batch['starts'], batch['ends'],
                                               list_of_lists_with_filtered_sentences):
            segment = []
            for sent in sents:
                start_time, end_time = sent[2]
                dur = end.item() - start.item()
                start_time = start_time * dur + start.item()
                end_time = end_time * dur + start.item()
                segment.append({
                    'question': sent[0],
                    'answer': sent[1],
                    'reason': [{'timestamp': [start_time, end_time], 'sentence': ''}]
                })
            predictions['dialogs'].append({'image_id': video_id, 'dialog': segment})

    if cfg.log_path is None:
        return None
    else:
        ## SAVING THE RESULTS IN A JSON FILE
        if cfg.procedure == 'train_cap':
            save_filename = f'captioning_results_{phase}_e{epoch}.json'
        else:
            save_filename = f'captioning_results_{phase}.json'
        submission_path = os.path.join(cfg.log_path, save_filename)

        # in case TBoard is not defined make logdir
        os.makedirs(cfg.log_path, exist_ok=True)

        # rename if already exists
        if os.path.exists(submission_path):
            root, ext = os.path.splitext(submission_path)
            n = 1
            while os.path.exists(submission_path):
                submission_path = f'{root}-{n}{ext}'
                n += 1

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf, indent=2)
        duration = time() - start_timer
        # blocks the printing
        with HiddenPrints():
            val_metrics = AVSD_eval(ground_truth_filenames=reference_paths,
                                    prediction_filename=submission_path,
                                    stopwords_filename=cfg.stopwords,
                                    last_only=cfg.last_only,
                                    verbose=False).evaluate()

        return val_metrics, duration
