import pandas as pd
import numpy as np
from collections import Counter
import sys

# wals_codes of the languages. Must match the first column in raw_wals_data.csv
langs_wals = ['afr', 'amh', 'arg', 'ass', 'aze', 'bsk', 'blr', 'bul', 'ben', 'ctl', 'krd', 'cze', 'wec', 'dsh', 'ger', 'grk', 'eng', 'spa', 'est', 'bsq', 'prs', 'fin', 'fre', 'iri', 'glc', 'guj', 'heb', 'hin', 'hun', 'arm', 'ind', 'ice', 'ita', 'jpn', 'geo', 'kaz', 'khm', 'knd', 'kor', 'kgz', 'lao', 'lit', 'lat', 'mcd', 'mym', 'mhi', 'mly', 'mlt', 'brm', 'nep', 'dut', 'nor', 'oya', 'pan', 'pol', 'psh', 'por', 'rom', 'rus', 'sdh', 'snh', 'svk', 'slo', 'alb', 'swe', 'swa', 'tml', 'tel', 'taj', 'tha', 'tag', 'tur', 'tvo', 'uyg', 'ukr', 'urd', 'uzb', 'vie', 'ydd', 'mnd']

# iso codes of the languages. This is what the user is expected to enter and must be used in litmus.py
langs_iso = ['af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'ca', 'ckb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'kk', 'km', 'kn', 'ko', 'ky', 'lo', 'lt', 'lv', 'mk', 'ml', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sq', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'yi', 'zh']

wals_file_path = sys.argv[1]
output_file_path = sys.argv[2]

with open(wals_file_path) as f:
    lines = f.read().strip().split('\n')
lines = [line.split(',') for line in lines]

df = pd.read_csv(wals_file_path, sep=',')
all_lines = df.values
feat_start_index = 29
feat_end_index = 202

lines = np.stack([line for line in all_lines if line[0] in langs_wals])

features = []
feature_names = list(df)[feat_start_index:feat_end_index]
for i, col in enumerate(range(feat_start_index, feat_end_index)):
    vals = [x for x in lines[:, col] if x == x]
    keys = Counter(vals).keys()
    features.extend([(feature_names[i], x) for x in keys])

feature_values = []
array = langs_wals
for i in range(len(array)):
    lang = array[i]
    feature_values.append([])
    for j in range(len(lines)):
        if lines[j][0] == lang:
            break
    idx = j
    col = feat_start_index
    k = 0
    for j in range(len(features)):
        if feature_names[k] != features[j][0]:
            k += 1
            col += 1
        if lines[idx][col] != lines[idx][col]:
            # check for nan
            feature_values[i].append(0)
        else:
            if lines[idx][col] == features[j][1]:
                feature_values[i].append(1)
            else:
                feature_values[i].append(0)
feature_values = np.stack(feature_values)

pd.DataFrame(feature_values, columns=features, index=langs_iso).to_csv(output_file_path)
