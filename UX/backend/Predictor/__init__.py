import logging
import tempfile
import os
import json
import base64
import numpy as np

import azure.functions as func
from ..common import jsonResponse, dumpBinaryToFile

from litmus.litmus_mixing import parse_args, litmus_main
from litmus.config import SUPPORTED_MODELS, SUPPORTED_MODES, MODEL2LANGS

def trim_spaces(input_string):
    input_list = input_string.split(",")
    processed_list = [item.strip(' ') for item in input_list]
    return ",".join(processed_list)

def process_targets(targets):
    input_list = targets.split(",")
    processed_list = [item.strip(' ') for item in input_list]
    target_langs = processed_list[:20]
    return ",".join(target_langs)

def get_py_flt(value):
    if isinstance(value, np.floating):
        return value.item()
    return value

"""
    Parse request inputs
"""
def readInputs(req):

    train_algo = req.params.get('training_algorithm')
    predictive_mode = req.params.get('predictive_mode');
    pivot_info = req.params.get('pivots_and_sizes', '') 
    pivots_and_sizes = json.loads(pivot_info)
    suggestions_targets = req.params.get('suggestions_targets', '')
    suggestions_budget = req.params.get('suggestions_budget', '10000')
    suggestions_augmentable = req.params.get('suggestions_augmentable','')
    suggestions_pivots_row = req.params.get('suggestions_pivots_row','')
    suggestions_objective = req.params.get('suggestions_objective')
    suggestions_lang_spec_budget = req.params.get('suggestions_lang_spec_budget','')
    suggestions_weights = req.params.get('suggestions_weights','')
    suggestions_minperf = req.params.get('suggestions_minperf','0')
    suggestions_minlangperf = req.params.get('suggestions_minlangperf','')

    predictive_mode = int(predictive_mode)
    suggestions_augmentable=trim_spaces(suggestions_augmentable)
    suggestions_targets=process_targets(suggestions_targets)

    # Convert inferencing train-configs to reqd format
    parseConfig = lambda s: ",".join(["%s:%s" % (lang.strip(), size.strip()[:-1]) for el in s.split(",") for lang, size in [el.strip().split("(")]])
    datasizes = ";".join(parseConfig(config) for config in pivots_and_sizes)

    # Read in model and data
    model, data, trainFormat = None, None, None

    # Priority-1: Check if pretrained exists
    pretrained = req.params.get('pretrained')
    if pretrained not in [None, "", "custom"] and pretrained in ["data_xlmr_xnli","data_xlmr_udpos","data_xlmr_wikiann"]:
        _, model, _ = pretrained.split("_")
        data = bytes("", 'utf-8')
        trainFormat = "matrix"

    # Priority-2: Use custom model + train-data
    if model == None or data == None:
        
        model = req.params.get('model')
        if not model or model not in SUPPORTED_MODELS:
            return jsonResponse({"error": "Could not find valid model in request!"}, status_code=400)

        data = req.get_body()
        if not data:
            return jsonResponse({"error": "Could not find valid training data in request!"}, status_code=400)

        trainFormat = "decomposed"

    # Build temp working directory and dump data
    tmpdir = tempfile.TemporaryDirectory()
    trainfile = dumpBinaryToFile(tmpdir.name, "train_data.tsv", data)

    return model, train_algo, trainfile, trainFormat, tmpdir, datasizes, suggestions_targets, pretrained, suggestions_budget, suggestions_augmentable, suggestions_pivots_row, pretrained, suggestions_objective, suggestions_lang_spec_budget, suggestions_weights, suggestions_minperf, suggestions_minlangperf, predictive_mode 


"""
    Main request handler
"""
def main(req: func.HttpRequest) -> func.HttpResponse:

    # Parse inputs
    res = readInputs(req)
    if type(res) == func.HttpResponse:
        return res
    model, train_algo, trainfile, trainFormat, tmpdir, datasizes, suggestions_targets, pretrained, suggestions_budget, suggestions_augmentable, suggestions_pivots_row, pretrained, suggestions_objective, suggestions_lang_spec_budget, suggestions_weights, suggestions_minperf, suggestions_minlangperf, predictive_mode = res

    train_format = "csv" if pretrained == "custom" else "json"

    try:
        # Basic params
        cli_args = [
            model, 
            "--common_scaling",
            "--training_algorithm", "xgboost",
            "--error_method", "split",
            "--output_dir", tmpdir.name,
            "--data_sizes", datasizes,
            "--train_format", train_format,
            "--use_all_langs",
            "--suggestions_targets", suggestions_targets,
        ]

        # Model-training/loading param
        if pretrained != "custom":  cli_args += ["--load_state", "Predictor/" + pretrained + ".pkl"]
        else:                       cli_args += ["--scores_file", tmpdir.name+"/train_data.tsv"]

        # Mode param
        if predictive_mode == 1:    cli_args += ["--mode", "heatmap"]
        else:                       cli_args += ["--mode", "heatmap", "suggestions"]
        
        # Suggestions-specific params
        if predictive_mode != 1:
            cli_args += [
                "--suggestions_budget", suggestions_budget,
                "--suggestions_pivots", suggestions_pivots_row,
                "--suggestions_augmentable", suggestions_augmentable,
                "--suggestions_langbudget", suggestions_lang_spec_budget,
                "--suggestions_weights", suggestions_weights,
                "--suggestions_minperf", suggestions_minperf,
                "--suggestions_objective", suggestions_objective,
                "--suggestions_minlangperf", suggestions_minlangperf
            ]

        # Invoke predictor tool
        args = parse_args(cli_args)
        res = litmus_main(args)
    
    except Exception as e:
        return jsonResponse({"error": str(e)}, status_code=400)

    if not res or len(res.keys()) <= 1:
        return jsonResponse({"error": "Some error occurred!"}, status_code=400)

    # Binarize heatmap to embed in json
    if "heatmapFile" in res:
        with open(res["heatmapFile"], "rb") as fin:
            img = fin.read()
        res["heatmap"] = base64.b64encode(img).decode("utf-8")

    # Cleanup temp files
    tmpdir.cleanup()

    del_keys=["langs_pivot", "langs_target", "baseline_error"]
    for k in del_keys:
        res.pop(k, None)

    # Float-ify
    if "user-config-perfs" in res:
        for ind,v in enumerate(res["user-config-perfs"]["best-tgt-perfs"]):
            value_ = list(v)
            value_[1]=get_py_flt(value_[1])
            res["user-config-perfs"]["best-tgt-perfs"][ind] = tuple(value_)

    if "suggestions" in res:
        res["suggestions"].pop("search-stats", None)
        res["suggestions"].pop("ablation", None)
        res["suggestions"]["search-perfs"].pop("max-perf", None)
        res["suggestions"]["search-perfs"]["baseline-perf"] = get_py_flt(res["suggestions"]["search-perfs"]["baseline-perf"])
        res["suggestions"]["search-perfs"]["augmented-perf"] = get_py_flt(res["suggestions"]["search-perfs"]["augmented-perf"])
        res["suggestions"]["search-perfs"]["equal-aug-perf"] = get_py_flt(res["suggestions"]["search-perfs"]["equal-aug-perf"])

        target_args = ["before-aug","after-aug","equal-aug"]
        for key in target_args:
            for k,v in res["suggestions"]["lang-perfs"][key].items():
                res["suggestions"]["lang-perfs"][key][k] = get_py_flt(v)

    return jsonResponse(res)