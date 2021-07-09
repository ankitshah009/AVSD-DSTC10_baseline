import argparse
from pprint import pprint

from utilities.config_constructor import Config
from scripts.train_captioning_module import train_cap
from scripts.eval_captioning_module import eval_cap

def main(cfg):
    if cfg.procedure == 'train_cap':
        train_cap(cfg)
    elif cfg.procedure == 'eval_cap':
        eval_cap(cfg)
    else:
        raise NotImplementedError


def get_parser():
    parser = argparse.ArgumentParser(description='Run experiment')

    ## DATA
    # paths to the precalculated train meta files
    parser.add_argument('--train_meta_path', type=str, default='./data/dstc10_train.csv')
    parser.add_argument('--val_meta_path', type=str, default='./data/dstc10_val.csv')
    parser.add_argument('--test_meta_path', type=str, default='./data/dstc10_test.csv')
    parser.add_argument('--modality', type=str, default='audio_video',
                        choices=['audio', 'video', 'audio_video'],
                        help='modality to use. if audio_video both audio and video are used')
    parser.add_argument('--video_feature_name', type=str, default='i3d')
    parser.add_argument('--audio_feature_name', type=str, default='vggish')
    parser.add_argument('--video_features_path', type=str, 
                        default='./data/i3d_25fps_stack64step64_2stream_npy/')
    parser.add_argument('--audio_features_path', type=str, 
                        default='./data/vggish_npy/')
    parser.add_argument('--d_vid', type=int, default=1024, help='raw feature dimension')
    parser.add_argument('--d_aud', type=int, default=128, help='raw feature dimension')
    parser.add_argument('--word_emb_caps', default='glove.840B.300d', type=str, 
                        help='Embedding code name from torchtext.vocab.Vocab')
    parser.add_argument('--unfreeze_word_emb', dest='unfreeze_word_emb', action='store_true',
                        default=False, help='Whether to finetune the pre-trained text embeddings')
    parser.add_argument('--feature_timespan_in_fps', type=int, default=64,
                        help='how many fps the input features will temporally cover')
    parser.add_argument('--fps_at_extraction', type=int, default=25, 
                        help='how many fps were used at feature extraction')
    parser.add_argument('--audio_feature_timespan', type=float,
                        default=0.96, help='audio feature timespan')
    parser.add_argument('--train_json_path', type=str, default='./data/train.json')

    ## TRAINING
    parser.add_argument('--procedure', type=str, required=True, 
                        choices=['train_cap', 'eval_cap', 'evaluate'])
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0], help='separated by a whitespace')
    parser.add_argument('--start_token', type=str, default='<s>', help='starting token')
    parser.add_argument('--end_token', type=str, default='</s>', help='ending token')
    parser.add_argument('--pad_token', type=str, default='<blank>', help='padding token')
    parser.add_argument('--context_start_token', type=str, default='Q:', help='context start token')
    parser.add_argument('--context_end_token', type=str, default='A:', help='context end token')
    parser.add_argument('--max_len', type=int, default=20, help='maximum size of 1by1 prediction')
    parser.add_argument('--min_freq_caps', type=int, default=2,
                        help='a word should appear min_freq times in train dataset to be in the vocab')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='betas in adam')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps in adam')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, default='constant',
                        choices=['constant', 'reduce_on_plateau'], help='lr scheduler')
    parser.add_argument('--lr', type=float, default=5e-5, help='lr (if scheduler is constant)')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_patience', type=int, help='ReduceLROnPlateau arguments')
    parser.add_argument('--lr_reduce_factor', type=float,
                        help='ReduceLROnPlateau arguments, (use 0.2 for 1/5)')
    parser.add_argument('--B', type=int, default=12, help='batch size per device')
    parser.add_argument('--inf_B_coeff', type=int, default=2,
                        help='The batch size on inference will be inf_B_coeff times B arg')
    parser.add_argument('--epoch_num', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--one_by_one_starts_at', type=int, default=1,
                        help='# of epochs to skip before starting 1-by-1 validation (saves time)')
    parser.add_argument('--early_stop_after', type=int, default=30,
                        help='number of epochs to wait for best metric to change before stopping')
    parser.add_argument('--key-metric', type=str, default='Bleu_4',
                        choices=['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'IoU-1', 'IoU-2'],
                        help='number of epochs to wait for best metric to change before stopping')
    parser.add_argument(
        '--smoothing', type=float, default=0.7,
        help='smoothing coeff (= 0 cross ent loss, more -- stronger smoothing) must be in [0, 1]'
    )
    parser.add_argument('--grad_clip', type=float, help='max grad norm for gradients')
    parser.add_argument('--pretrained_prop_model_path', type=str, 
                        help='path to pre-trained cap model .pt')
    parser.add_argument('--pretrained_cap_model_path', type=str,
                        help='path to pre-trained cap model .pt')
    parser.add_argument('--region_std_coeff', default=1.0, type=float,
                        help='reasoning region is decided based on the most attended frame +/- std * coeff')
    parser.add_argument('--exp_name', type=str, default='avsd')
    parser.add_argument('--log_dir', type=str, default='./log/')

    ## EVALUATION
    parser.add_argument('--reference_paths', type=str, nargs='+',
                        default=['./data/val_set4DSTC10-AVSD+reason.json'],
                        help='reference paths for 1-by-1 validation')
    parser.add_argument('--stopwords', type=str, default=None,
                        help='use a file listing stop words')
    parser.add_argument('--last_only', action='store_true',
                        help='evaluate only the last turn')
    ## MODEL
    parser.add_argument('--model', type=str, default='av_transformer',
                        choices=['transformer', 'av_transformer'], help='caption model type')
    parser.add_argument('--dout_p', type=float, default=0.1, help='dropout probability: in [0, 1]')
    parser.add_argument('--N', type=int, default=2, help='number of layers in a model')
    parser.add_argument(
        '--d_model', type=int, default=1024,
        help='the internal space in the multi-headed attention (when input dims of Q, K, V differ)')
    parser.add_argument(
        '--d_model_video', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for video model'
    )
    parser.add_argument(
        '--d_model_audio', type=int,
        help='If use_linear_embedder is true, this is going to be the d_model size for audio model'
    )
    parser.add_argument(
        '--d_model_caps', type=int, default=300,
        help='hidden size of the crossmodal decoder (caption tokens are mapped into this dim)'
    )
    parser.add_argument(
        '--use_linear_embedder', dest='use_linear_embedder', action='store_true', default=False,
        help='Whether to include a dense layer between the raw features and input to the model'
    )
    parser.add_argument('--H', type=int, default=4, help='number of heads in multiheaded attention')
    parser.add_argument(
        '--d_ff_video', type=int, help='size of the internal layer of PositionwiseFeedForward')
    parser.add_argument(
        '--d_ff_audio', type=int, help='size of the internal layer of PositionwiseFeedForward')
    parser.add_argument(
        '--d_ff_caps', type=int, help='size of the internal layer of PositionwiseFeedForward')

    ## DEBUGGING
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='runs test() instead of main()')
    parser.add_argument('--dont_log', dest='to_log', action='store_false',
                        help='Prevent logging in the experiment.')

    parser.set_defaults(to_log=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pprint(vars(args))
    cfg = Config(args)

    if args.debug:
        # load your test to debug something using the same config as main() would
        # from tests import test_features_max_length
        # test_features_max_length(cfg)
        pass
    else:
        main(cfg)
