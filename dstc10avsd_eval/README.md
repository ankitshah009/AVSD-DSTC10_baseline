# Official evaluation package for DSTC10 Audio-Visual Scene-aware Dialog (AVSD) track

## Required programs/packages

- bash
- python3
- numpy
- pycocoevalcap  (https://github.com/salaniz/pycocoevalcap)

## Usage

run script `dstc10avsd_eval.sh` with your result `your_result.json` as

    $ ./dstc10avsd_eval.sh your_result.json

Here we assume that `your_result.json` was generated for the official test set: `test_set4DSTC10-AVSD_multiref+reason.json`, which is stored in `./data`.

## Example

    Reference: ./data/test_set4DSTC10-AVSD_multiref+reason.json
    Submission: sample/baseline.json
    PTBTokenizer tokenized 124033 tokens at 665117.78 tokens per second.
    PTBTokenizer tokenized 15141 tokens at 170529.33 tokens per second.
    {'testlen': 13172, 'reflen': 14065, 'guess': [13172, 11368, 9568, 7768], 'correct': [8057, 3796, 1874, 947]}
    ratio: 0.9365090650550347
    -------------------------
    | Bleu_1: 0.5716
    | Bleu_2: 0.4223
    | Bleu_3: 0.3196
    | Bleu_4: 0.2469
    | METEOR: 0.1909
    | ROUGE_L: 0.4386
    | CIDEr: 0.5657
    | IoU-1: 0.3614
    | IoU-2: 0.3798
    -------------------------
