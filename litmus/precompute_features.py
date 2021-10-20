import os
import sys
import math
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import lang2vec.lang2vec as l2v

from config import SUPPORTED_MODELS, MODEL2LANGS

"""
    mBERT - Languages not covered
    Merged with another entry
    nn, pnb, azb

    Not in WALS
    vo, scn, sh, sr, sco, lmo, la, io, bpy, an
"""

"""
    XLMR - Languages not covered
    Romanized entry merged with original
    ta, te, ur, hi, bn

    2 entries - merged with original
    zh, my

    Not in WALS
    hr, sr, sa, mn, la, eo
"""

# Estimated based on sizes of wikidump 20200701
wikidump_sizes = [101389967, 86495335, 1126461327, 332797817, 223265632, 263585428, 67099152, 196524022, 31811929, 210474434, 202013601, 109628453, 47373068, 349183292, 39539484, 895839724, 1658060465, 48781035, 1961922268, 22514907, 835617128, 336451109, 1476680581, 17477811775, 207941275, 698919757, 4732727196, 254573131, 147473116, 5626005050, 377145828, 29300531, 12720745, 662585900, 162296644, 885350593, 43271701, 607530965, 26102678, 2988368384, 3145690946, 41561809, 77191210, 112313224, 32158891, 691806744, 130899417, 174032254, 37440082, 45436165, 156157024, 24296999, 234837091, 134065499, 54343825, 23564720, 34715073, 15417727, 734726142, 72369008, 760371152, 11966993, 1939734581, 1749522203, 88091722, 470628336, 4024668972, 261326883, 215136254, 3341233864, 23975080, 29778048, 1551056145, 54874876, 38293533, 161017276, 58100243, 106951236, 544125686, 1549537006, 139099825, 57416485, 725953837, 199104540, 70506029, 47652617, 12208974, 281461569]
assert len(MODEL2LANGS["mbert"]) == len(wikidump_sizes)

xlmr_pretraining_sizes = [1395864371.2, 858993459.2, 30064771072.0, 107374182.4, 6979321856.0, 4617089843.2, 61740154880.0, 9556302233.6, 107374182.4, 107374182.4, 10844792422.4, 17501991731.2, 858993459.2, 48962627174.4, 71511205478.4, 50358491545.6, 322981540659.2, 57230439219.2, 6549825126.4, 2147483648.0, 119829587558.4, 58304181043.2, 60988535603.2, 214748364.8, 536870912.0, 107374182.4, 3113851289.6, 2040109465.6, 322122547.2, 33930241638.4, 22226455756.8, 62706522521.6, 5905580032.0, 159235912499.2, 3435973836.8, 32427003084.8, 74410308403.2, 214748364.8, 9771050598.4, 6871947673.6, 1610612736.0, 3543348019.2, 58196806860.8, 429496729.6, 1288490188.8, 644245094.4, 14710262988.8, 9448928051.2, 214748364.8, 5153960755.2, 8160437862.4, 3006477107.2, 9126805504.0, 2147483648.0, 4080218931.2, 31460635443.2, 52613349376.0, 107374182.4, 644245094.4, 858993459.2, 47888885350.4, 751619276.8, 52720723558.4, 65927747993.6, 298500227072.0, 429496729.6, 3865470566.4, 24910810316.8, 11059540787.2, 429496729.6, 5798205849.6, 107374182.4, 12992276070.4, 1717986918.4, 13421772800.0, 5368709120.0, 76987288780.8, 3328599654.4, 22441204121.6, 429496729.6, 90838558310.4, 6657199308.8, 751619276.8, 147424752435.2, 107374182.4, 322122547.2, 68182605824.0]
assert len(MODEL2LANGS["xlmr"]) == len(xlmr_pretraining_sizes)


def calc_regression_feats(model, wals_features_path):
    # Excel sheet has binarized WALS features. First read these features
    sheet_name = {'mbert': 'mbert_features', 'xlmr': 'xlmr_features'}[model]
    df = pd.read_excel(wals_features_path, index_col=0, sheet_name=sheet_name)
    head = df.columns.values
    for i, x in enumerate(head):
        x = literal_eval(x)
        x = (x[0].split(' ')[0], x[1].split(' ')[0])
        head[i] = x

    # Use pretraining corpora size to determine the amount of training data present for each feature
    sizes_list = {'mbert': wikidump_sizes, 'xlmr': xlmr_pretraining_sizes}[model]
    feats = []
    for i, x in enumerate(df.columns.values):
        assert df.iloc[:, i].values.shape[0] == len(sizes_list)
        feats.append((x, np.sum(df.iloc[:, i].values * np.array(sizes_list))))
    feats = sorted(feats, key=lambda x: x[1], reverse=True)
    feats = [t[0] for t in feats]

    # Calculate underrep_feat metric as mrr over features that each language has
    mrrs = []
    for lang in df.index:
        lang_feats = df.loc[lang, :].values
        lang_mrr = []
        for i, feat in enumerate(df.columns.values):
            if lang_feats[i] == 1:
                rank = feats.index(feat) + 1
                lang_mrr.append(1 / rank)
        if len(lang_mrr) == 0:
            lang_mrr.append(0)
        mrrs.append((lang, np.mean(lang_mrr)))

    regression_feats = np.stack([[math.log10(t) for t in sizes_list], [t[1] for t in mrrs]])
    regression_feats = regression_feats.transpose(1, 0)
    # regression_feats = pd.DataFrame(regression_feats, index=MODEL2LANGS[model])

    return regression_feats


def read_tokenized_wiki(model, path_base):
    tokenized = {}

    # json files contain counts of each token in the wikipedia of each language
    path = {'mbert': f'{path_base}/mbert', 'xlmr': f'{path_base}/xlmr'}[model]
    for f in os.listdir(path):
        lang = f.split('.')[0]
        tokenized[lang] = json.load(open(os.path.join(path, f)))

    # Ignore types which occur less than 5 times
    for lang in tokenized:
        to_del = []
        for key in tokenized[lang]:
            if tokenized[lang][key] <= 5:
                to_del.append(key)
        for key in to_del:
            del tokenized[lang][key]

    return tokenized


def type_overlap(tokenized, lang1, lang2):
    keys1 = set(tokenized[lang1].keys())
    keys2 = set(tokenized[lang2].keys())
    intersection = keys1 & keys2
    union = keys1 | keys2
    return len(intersection) / len(union)


def lang_to_l2v_index(lang):
    if lang in l2v.LETTER_CODES:
        return l2v.LETTER_CODES[lang]
    elif lang in l2v.LANGUAGES:
        return lang
    else:
        raise Exception("Language not found in lang2vec")


def precompute_main(args):

    print("Precomputing type-overlaps (on wiki data)...")
    type_overlaps = {
        model: {
            lang1: {lang2: type_overlap(tokenized, lang1, lang2) for lang2 in tokenized}
                for lang1 in tqdm(tokenized)
        }
             for model in tqdm(SUPPORTED_MODELS)
                 for tokenized in [read_tokenized_wiki(model, args.token_data_path)]
    }

    print("Precomputing syntactic distances...")

    syntactic_distances = {
        model: l2v.syntactic_distance([lang_to_l2v_index(t) for t in MODEL2LANGS[model]]).tolist()
            for model in tqdm(SUPPORTED_MODELS)
    }

    print("Precomputing feats for regression: pretraining data size and underrepresented features metric...")
    regression_feats = {
        model: calc_regression_feats(model, args.wals_features_path).tolist()
            for model in tqdm(SUPPORTED_MODELS)
    }

    # Write out all features into a unified json
    features = {
        "type_overlap": type_overlaps,
        "syntactic_distance": syntactic_distances,
        "regression_feats": regression_feats
    }

    print("Dumping features to JSON...")
    with open(args.output_file, "w") as fout:
        json.dump(features, fout, indent=4)


def parse_args(args):
    parser = argparse.ArgumentParser("LITMUS Pre-compute features tool")
    parser.add_argument("--token_data_path", type=str, default="./data/token_data", help="location of token_data folder. (default: ./data/token_data)")
    parser.add_argument("--wals_features_path", type=str, default="./data/wals_data.xlsx", help="location of xlsx file with processed WALS features. (default: ./wals_data.xlsx)")
    parser.add_argument("--output_file", type=str, default='./data/precomputed_features.json', help="file to output generated features to")
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    precompute_main(args)