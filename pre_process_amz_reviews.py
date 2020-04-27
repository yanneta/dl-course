
import argparse
import json
import numpy as np
import html
import spacy
import pickle

parser = argparse.ArgumentParser(description='Preprocess amazon reviews dataset using Spacy')
parser.add_argument('--input_file', default='', required=True, metavar='PATH')
parser.add_argument('--output_file', default='', required=True, metavar='PATH')
args = parser.parse_args()

def load_data(datafile):
    samples = [json.loads(line) for line in open(datafile).readlines()]
    data = {}
    data['review'] = [html.unescape(sample['reviewText']) for sample in samples]
    data['summary'] = [html.unescape(sample['summary']) for sample in samples]
    data['rating'] = np.array([sample['overall'] for sample in samples])
    return data

def get_clean_review(review, summ, rating):
    sample = {}
    # remove stop-words and whitespace tokens split paragraphs into sentences
    review_valid = [[tok for tok in sent if not tok.is_stop and tok.text.strip() != ''] for sent in review.sents]
    # remove empty sentences
    review_valid = [sent for sent in review_valid if not len(sent) == 0]
    sample['review'] = [[tok.text.lower() for tok in sent] for sent in review_valid]
    # remove stop-words and whitespace tokens
    summary_valid = [tok for tok in summ if not tok.is_stop and tok.text.strip() != '']
    sample['summary'] = [tok.text.lower() for tok in summary_valid]
    sample['rating'] = int(rating)
    return sample

def dump_dataset(raw_data, outfile, summary=True):
    with open(outfile, 'w') as outf:
        nlp = spacy.load('en_core_web_sm')
        review_docs = nlp.pipe(raw_data['review'])
        summ_docs = nlp.pipe(raw_data['summary'])
        n = len(raw_data['rating'])
        for review, summ, rating in zip(review_docs, summ_docs, raw_data['rating']):
            sample = get_clean_review(review, summ, rating)
            outf.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    print("Loading raw data from {}".format(args.input_file))
    data = load_data(args.input_file)
    print("Preprocessing data and writing to {}".format(args.output_file))
    dump_dataset(data, args.output_file)

