# DSTC10 - Audio Visual Scene-aware Dialog (AVSD) - Baseline System

This repository provides a baseline system for the AVSD track of the 10-th Dialog System Technology Challenges (DSTC10).
The system employs an audio-visual Transformer with I3D visual features and VGGish audio features in the default setup.
The system outputs answers in response to questions with a dialog history and timings for evidence reasoing the answers as well. 

Details of our scheme are in the following paper.
https://arxiv.org/xxxxx

    @ixxxx{ankit@arxiv,
            title={Reasoning for Audio Visual Scene-Aware Dialog(AVSD) Track in DSTC10},
            author={Ankit Shah, Shijie Geng, Peng Gao, Anoop Cherian, Tim K. Marks, Chiori Hori},
            booktitle={Accepted to DSTC10},
            year={2021}
    }

## How to run the code:

   1. Obtain the package.
      - `git clone --recursive https://github.com/ankitshah009/AVSD-DSTC10_baseline/`
  
   2. Confirm if the following files exist in the downloaded repo
      - `data/train_set4DSTC10-AVSD.json` (official training set)
      - `data/valid_set4DSTC10-AVSD+reason.json` (official validation set)
      - `data/test_set4DSTC10-AVSD+reason.json` (official test set, not included in the package yet,
        however will be provided)
 
   3. Run `download_data.sh` to download the feature files in `./data/features` (or make sybolic links),
      where `video_feats` and `vggish` directories will be created. `wget` command is required. (If wget is not available, please see https://www.tecmint.com/install-wget-in-linux/)

   4. Run `setup.sh` to create a local conda environment (not mandatory. you can do it 
      manually based on the required packages specified in `./conda_env.yaml`)

   5. Run `run.sh` to train the audio-visual Transformer.<br>
      In order to run the code on multiple GPUs - add device_ids parameters to the main.py command.  
      Model files and generated sentences will be stored in `./log/XXXX/train_cap/`, where XXXX
      is an experiment name specified in `run.sh`.
      - `captioning_results_val_eYY.json`: results including generated sentences and
        reasoning regions for the validation set at epoch YY (YY > 30).
      - `best_cap_model.pt`: best model based on `Bleu_4` score
      - `events.out.tfevents..`: logfile for TensorBoard. You can check the progress with
        `tensorboard --logdir ./log` and your browser.

   6. Use `run_gen.sh` to generate answers and evaluate the performance using a trained model. The results will be stored
      in `./log/XXXX/eval_cap/`. <br>
      Note that `run_gen.sh` currently generates answers for the validation set. You can run `run_gen.sh test` for the test set if it is available.

   7. Use `eval.sh` to compute the quality of generated answers in json format

### Example output of `run_gen.sh`

    -------------------------
    | Bleu_1: 25.5674
    | Bleu_2: 16.4486
    | Bleu_3: 11.3103
    | Bleu_4: 8.1863
    | METEOR: 11.6901
    | ROUGE_L: 29.5971
    | CIDEr: 83.4498
    | IoU-1: 39.9970
    | IoU-2: 41.3343
    -------------------------

Note: IoU-1 and IoU-2 are computed based on the *Intersection over Union (IoU)* between ground-truth
and predicted reasoning regions, where IoU-1 measures the IoU for each pair of proposed regions while
IoU-2 measures the IoU between merged regions.


## Citation
Please cite the following papers if you use this package for publication:

### Attentional multimodal fusion for AVSD.
https://arxiv.org/abs/1806.08409

    @article{hori2018end,
      title={End-to-End Audio Visual Scene-Aware Dialog using Multimodal Attention-Based Video Features},
      author={Hori, Chiori and Alamri, Huda and Wang, Jue and Winchern, Gordon and Hori, Takaaki and Cherian, Anoop and Marks, Tim K and Cartillier, Vincent and Lopes, Raphael Gontijo and Das, Abhishek and others},
      journal={arXiv preprint arXiv:1806.08409},
      year={2018}
    } 

### Audio Visual Transformer-based AVSD@DSTC10 baseline system
https://arxiv.org/xxxxx

    @ixxxx{ankit@arxiv,
            title={Reasoning for Audio Visual Scene-Aware Dialog(AVSD) Track in DSTC10},
            author={Ankit Shah, Shijie Geng, Peng Gao, Anoop Cherian, Tim K. Marks, Chiori Hori},
            booktitle={Accepted to DSTC10},
            year={2021}
    }

## Acknowledgements
This system has been built upon the Bi-modal Transformer in https://github.com/v-iashin/BMT, and modified for AVSD.

    @InProceedings{BMT_Iashin_2020,
      title={A Better Use of Audio-Visual Cues: Dense Video Captioning with Bi-modal Transformer},
      author={Iashin, Vladimir and Rahtu, Esa},
      booktitle={British Machine Vision Conference (BMVC)},
      year={2020}
    }

