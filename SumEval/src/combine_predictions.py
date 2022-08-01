from __future__ import annotations

import itertools
import json
import os
import argparse
import sys
from typing import Iterable

filenames = {
    "sumeval_test": [
        "TyDiQA_TULRv6Large_ID.json",
        "TyDiQA_TULRv6Large_OOD.json",
        "TyDiQA_XLMR_ID.json",
        "TyDiQA_XLMR_OOD.json",
        "UDPOS_XLMR.json",
        "WikiANN_XLMR.json",
        "XNLI_TULRv6Large.json",
        "XNLI_XLMR.json",
    ],
    "sumeval_surprise": [
        "UDPOS_XLMR_surprise_langs_diff_configs.json",
        "UDPOS_XLMR_surprise_langs_same_configs.json",
        "WikiANN_XLMR_surprise_langs_diff_configs.json",
        "WikiANN_XLMR_surprise_langs_same_configs.json",
        "XNLI_TULRv6Large_surprise_langs_diff_configs.json",
        "XNLI_TULRv6Large_surprise_langs_same_configs.json",
        "XNLI_XLMR_surprise_langs_diff_configs.json",
        "XNLI_XLMR_surprise_langs_same_configs.json",
    ],
}


def parse_to_combined_json(
    json_dir: str,
    input_filename: str,
    value_name: str = "predicted_value",
) -> Iterable[dict[str, float | str]]:
    """
    Parse the original file to a combined json format.

    Inputs:
        - json_dir: the directory with the json files
        - input_filename: the name of the input JSON file from which to get the examples
        - value_name: the field name for the output value

    Returns:
        - an iterator over the output examples
    """
    full_path = os.path.join(json_dir, input_filename)
    if not os.path.exists(full_path):
        raise ValueError(f"could not find file {full_path}")
    if not input_filename.endswith(".json"):
        raise ValueError(f"{input_filename} does not end with .json")
    # Get features for the data setting
    features: dict[str, float | str] = {}
    features["overall_setting"] = input_filename[:-5]
    overall_cols = features["overall_setting"].split("_")
    features["dataset_name"] = overall_cols[0]
    features["model_name"] = overall_cols[1]
    features["sub_setting"] = (
        "_".join(overall_cols[2:]) if len(overall_cols) > 2 else "default"
    )
    # Enumerate the configurations
    with open(full_path, "r") as in_handle:
        for config_id, config in enumerate(json.load(in_handle)):
            if any([x not in config for x in ("train_config", "eval_results")]):
                raise ValueError(f"Missing train_config or eval_results in {config}")
            features["data_setting"] = f'{features["overall_setting"]}_{config_id}'
            for language, value in config["eval_results"].items():
                features["target_lang_data_size"] = config["train_config"].get(
                    language, 0
                )
                features["target_lang"] = language
                try:
                    value = float(value)
                except:
                    pass
                features[value_name] = value
                yield dict(features)


def verify_template(split_name: str, pred_data: list[dict], template_dir: str) -> None:
    """
    Verify that the extracted data matches the template and throw an exception if not.

    Inputs:
        - split_name: The name of the split
        - pred_data: The parsed data from the prediction
        - template_dir: The directory that the template files are in
    """
    with open(os.path.join(template_dir, f"{split_name}.json"), "r") as template_handle:
        template_data = json.load(template_handle)
        if len(template_data) != len(pred_data):
            raise ValueError(
                "len(template_data) != len(pred_data) -> "
                f"{len(template_data)} != {len(pred_data)}"
            )
        for line_id, (my_template, my_pred) in enumerate(zip(template_data, pred_data)):
            for pred_key, pred_value in my_pred:
                if pred_key != "prediction" and my_template[pred_key] != pred_value:
                    raise ValueError(
                        f"line {line_id} mismatch with template"
                        f"\n{my_template}\n{my_pred}"
                    )


def main():
    global filenames

    parser = argparse.ArgumentParser(
        "Consolidates separate JSON files into a single one for upload"
    )
    parser.add_argument(
        "--pred_dir",
        default="Baselines/preds",
        type=str,
        help="The directory with the predictions",
    )
    parser.add_argument(
        "--template_dir",
        default=None,
        type=str,
        required=False,
        help="A directory of template files",
    )
    parser.add_argument(
        "--out_dir",
        default="Baselines/combined_preds",
        type=str,
        help="Path of directory to output consolidated JSON",
    )
    parser.add_argument(
        "--value_name",
        default="predicted_value",
        type=str,
        help="The name of the value field in the output json",
    )
    parser.add_argument(
        "--split",
        default="all",
        type=str,
        choices=["sumeval_test", "sumeval_surprise", "all"],
        help="Test split for which to combine predictions",
    )
    args = parser.parse_args()

    if args.split != "all":
        filenames = {args.split: filenames[args.split]}

    # Read in the data
    pred_data = {}
    for split, split_files in filenames.items():
        pred_data[split] = list(
            itertools.chain.from_iterable(
                [
                    parse_to_combined_json(args.pred_dir, x, value_name=args.value_name)
                    for x in split_files
                ]
            )
        )

    # Output the data split by split
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    for split_name, pred_val in pred_data.items():
        # Verify with the template if exists
        if args.template_dir is not None:
            verify_template(split_name, pred_val, args.template_dir)
        out_path = os.path.join(args.out_dir, f"{split_name}.json")
        print(f"* Writing {out_path}", file=sys.stderr)
        with open(out_path, "w") as out_handle:
            json.dump({"examples": pred_val}, out_handle)

    print(f"Success!", file=sys.stderr)


if __name__ == "__main__":
    main()
