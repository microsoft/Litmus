import argparse
import os
import json
import glob
import pandas as pd
import pdb
import sys

sys.path.append("../../Litmus")
from litmus.litmus_mixing import parse_args, litmus_main

PRECOMPUTED_FEATURE_FILE = "../../Litmus/litmus/data/precomputed_features.json"

pred2train_files = {
    "TyDiQA_XLMR_ID.json": "TyDiQA_XLMR_ID.json",
    "TyDiQA_XLMR_OOD.json": "TyDiQA_XLMR_OOD.json",
    "TyDiQA_TULRv6Large_ID.json": "TyDiQA_TULRv6Large_ID.json",
    "TyDiQA_TULRv6Large_OOD.json": "TyDiQA_TULRv6Large_OOD.json",
    "UDPOS_XLMR.json": "UDPOS_XLMR.json",
    "UDPOS_XLMR_surprise_langs_same_configs.json": "UDPOS_XLMR.json",
    "UDPOS_XLMR_surprise_langs_diff_configs.json": "UDPOS_XLMR.json",
    "WikiANN_XLMR.json": "WikiANN_XLMR.json",
    "WikiANN_XLMR_surprise_langs_same_configs.json": "WikiANN_XLMR.json",
    "WikiANN_XLMR_surprise_langs_diff_configs.json": "WikiANN_XLMR.json",
    "XNLI_XLMR.json": "XNLI_XLMR.json",
    "XNLI_XLMR_surprise_langs_same_configs.json": "XNLI_XLMR.json",
    "XNLI_XLMR_surprise_langs_diff_configs.json": "XNLI_XLMR.json",
    "XNLI_TULRv6Large.json": "XNLI_TULRv6Large.json",
    "XNLI_TULRv6Large_surprise_langs_same_configs.json": "XNLI_TULRv6Large.json",
    "XNLI_TULRv6Large_surprise_langs_diff_configs.json": "XNLI_TULRv6Large.json",
}


def predict_on_file(pred_filename, data_dir, save_dir, train_filename, model):

    with open(pred_filename) as fp:
        to_pred_data = json.load(fp)
    data_size_str = ""

    for i in range(len(to_pred_data)):
        train_config = to_pred_data[i]["train_config"]
        data_size_str += ",".join(
            [f"{lang}:{size}" for lang, size in train_config.items()]
        )
        if i != len(to_pred_data) - 1:
            data_size_str += ";"

    tgt_langs = list(to_pred_data[0]["eval_results"].keys())

    cmd = f"{model} --scores_file {train_filename} --error_method split --pivot_features all --mode pred"
    cmd += f" --data_sizes {data_size_str}"
    cmd += f" --heatmap_targets {','.join(tgt_langs)} --precomputed_features {PRECOMPUTED_FEATURE_FILE}"
    config = parse_args(cmd.split())
    ret_val = litmus_main(config)
    tgt_perf_preds = ret_val["user-config-perfs"]["tgt_perfs"]
    pred_dict = []
    for i in range(len(to_pred_data)):
        pred_dict.append(
            {
                "train_config": to_pred_data[i]["train_config"],
                "eval_results": {
                    lang: str(pred) for lang, pred in zip(tgt_langs, tgt_perf_preds[i])
                },
            }
        )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = f"{save_dir}/{pred_filename.split('/')[-1]}"
    with open(save_filename, "w") as fp:
        json.dump(obj=pred_dict, fp=fp, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="xlmr", choices=["xlmr", "tulrv6"])
    args = parser.parse_args()

    model_str = "XLMR" if args.model == "xlmr" else "TULRv6Large"

    pred_files = glob.glob(f"{args.data_dir}/test_release/*_{model_str}*.json")
    # train_files = glob.glob(f"{args.data_dir}/train/*_XLMR*.json")
    for pred_file in pred_files:
        print(f"Predicting on {pred_file}")
        pred_filename = pred_file.split("/")[-1]
        train_filename = pred2train_files[pred_filename]
        train_file = f"{args.data_dir}/train/{train_filename}"
        predict_on_file(
            pred_file, args.data_dir, f"{args.out_dir}/preds/", train_file, args.model
        )


if __name__ == "__main__":
    main()
