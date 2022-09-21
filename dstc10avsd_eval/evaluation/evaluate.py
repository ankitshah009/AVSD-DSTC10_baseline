import string

import argparse
import json
import sys
from collections import defaultdict
sys.path.insert(0, './evaluation/')

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from stopword_filter import StopwordFilter


class AVSD_eval(object):

    def __init__(self, ground_truth_filenames, prediction_filename,
                 stopwords_filename=None, last_only=False, verbose=False):
        # Check that the gt and submission files exist and load them
        if stopwords_filename:
            self.filter = StopwordFilter(stopwords_filename)
        else:
            self.filter = lambda x : x
        self.verbose = verbose
        self.ground_truths = self.import_ground_truths(ground_truth_filenames, last_only)
        self.prediction = self.import_prediction(prediction_filename, last_only)
        self.tokenizer = PTBTokenizer()
        # Set up scorers, if not verbose, we only use the one we're
        # evaluating on: METEOR
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

    def to_coco(self, kvs, keys):
        res = defaultdict(list)
        for k in keys:
            if k in kvs:
                caps = kvs[k]
                for c in caps:
                    res[k].append({'caption': c})
            else:
                res[k].append({'caption': ''})
        return res

    def import_prediction(self, prediction_filename, last_only=False):
        if self.verbose:
            print("| Loading submission...")
        submission = json.load(open(prediction_filename))
        results = defaultdict(list)
        intervals = defaultdict(list)
        for d in submission['dialogs']:
            vid = d['image_id']
            if last_only:
                t, turn = len(d['dialog']), d['dialog'][-1]
                answer = turn['answer'] if type(turn['answer'])==list else [turn['answer']]
                results['%s_res%03d' % (vid, t)].extend([self.filter(a) for a in answer])
                if 'reason' in turn:
                    for reason in turn['reason']:
                        intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
            else:
                for t, turn in enumerate(d['dialog'], 1):
                    answer = turn['answer'] if type(turn['answer'])==list else [turn['answer']]
                    results['%s_res%03d' % (vid, t)].extend([self.filter(a) for a in answer])
                    if 'reason' in turn:
                        for reason in turn['reason']:
                            intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
        return results, intervals

    def import_ground_truths(self, filenames, last_only=False):
        gts = defaultdict(list)
        intervals = defaultdict(list)
        for filename in filenames:
            gt = json.load(open(filename))
            for d in gt['dialogs']:
                vid = d['image_id']
                if last_only:
                    t, turn = len(d['dialog']), d['dialog'][-1]
                    answer = turn['answer'] if type(turn['answer'])==list else [turn['answer']]
                    gts['%s_res%03d' % (vid, t)].extend([self.filter(a) for a in answer])
                    if 'reason' in turn:
                        for reason in turn['reason']:
                            intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
                else:
                    for t, turn in enumerate(d['dialog'], 1):
                        answer = turn['answer'] if type(turn['answer'])==list else [turn['answer']]
                        gts['%s_res%03d' % (vid, t)].extend([self.filter(a) for a in answer])
                        if 'reason' in turn:
                            for reason in turn['reason']:
                                intervals['%s_res%03d' % (vid, t)].append(reason['timestamp'])
        if self.verbose:
            print ("| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(gts)))
        return gts, intervals

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def set_iou(self, intervals_1, intervals_2):
        def within_interval(intervals, t):
            return sum([interval[0] <= t < interval[1] for interval in intervals]) > 0
        tick = 0.1  # tick time [sec]
        intervals = intervals_1 + intervals_2
        lower = int(min([intrv[0] for intrv in intervals]) / tick)
        upper = int(max([intrv[1] for intrv in intervals]) / tick)
        intersection = 0
        union = 0
        for t in range(lower, upper):
            time = t * tick
            intersection += int(within_interval(intervals_1, time) & within_interval(intervals_2, time))
            union += int(within_interval(intervals_1, time) | within_interval(intervals_2, time))
        iou = float(intersection) / (union + 1e-8)
        return iou

    def average_iou1(self, ref, hypo):
        ious = []
        for key, intervals in ref.items():
            if key in hypo:
                for r_int in intervals:
                    ious.append(max([self.iou(r_int, h_int) for h_int in hypo[key]]))
            else:
                ious.extend([0.0] * len(intervals))
        if len(ious) > 0:
            return sum(ious) / len(ious)
        else:
            return 0.0

    def average_iou2(self, ref, hypo):
        ious = []
        for key, intervals in ref.items():
            if key in hypo:
                ious.append(self.set_iou(intervals, hypo[key]))
            else:
                ious.append(0.0)
        if len(ious) > 0:
            return sum(ious) / len(ious)
        else:
            return 0.0

    def evaluate(self):
        ref_sent, ref_int = self.ground_truths
        hypo_sent, hypo_int = self.prediction
        ref_coco = self.tokenizer.tokenize(self.to_coco(ref_sent, ref_sent.keys()))
        hypo_coco = self.tokenizer.tokenize(self.to_coco(hypo_sent, ref_sent.keys()))
        final_scores = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(ref_coco, hypo_coco)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        final_scores['IoU-1'] = self.average_iou1(ref_int, hypo_int)
        final_scores['IoU-2'] = self.average_iou2(ref_int, hypo_int)
        self.scores = final_scores
        return self.scores


def main(args):
    # Call coco eval
    evaluator = AVSD_eval(ground_truth_filenames=args.references,
                          prediction_filename=args.submission,
                          stopwords_filename=args.stopwords,
                          last_only=args.last_only,
                          verbose=args.verbose)
    evaluator.evaluate()
    # Output the results
    print ('-' * 25)
    for metric, score in evaluator.scores.items():
        print ('| %s: %2.4f' % (metric, score))
    print ('-' * 25)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='sample_submission.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, nargs='+', default=['data/val_1.json'],
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('-S', '--stopwords', type=str, default=None,
                        help='use a file listing stop words')
    parser.add_argument('-l', '--last_only', action='store_true',
                        help='evaluate only the last turn')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print intermediate steps.')
    args = parser.parse_args()

    main(args)
