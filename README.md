# LITMUS Predictor

[![license](https://img.shields.io/badge/Demo-youtu.be/20dR8oKc9j0-critical?style=&logo=youtube)](https://www.youtube.com/watch?v=20dR8oKc9j0)
[![license](https://img.shields.io/badge/Predictor-microsoft.github.io/Litmus-informational?style=&logo=microsoft-azure)](https://microsoft.github.io/Litmus/)
[![license](https://img.shields.io/badge/LITMUS-microsoft.com/research/project--litmus-green?style=&logo=microsoft)](https://www.microsoft.com/en-us/research/project/project-litmus/)


LITMUS Predictor provides support for simulating performance in ~100 languages given training observations of the desired task-model. Each training observation specifies the finetuning-datasize + test-performance in different languages.

Further, the tool provides support for constructing a data-collection strategy to maximize performance in desired targets subject to different constraints.

## Installation
```
pip install -U pip
pip install -r requirements.txt
```

## Usage
`litmus/litmus_mixing.py` contains the implementation of the LITMUS Predictor which can be trained on observations of different task-model trainings.

```
usage: LITMUS Tool [-h] [--scores_file SCORES_FILE]
                   [--train_format {json,csv}] [--save_state SAVE_STATE]
                   [--load_state LOAD_STATE]
                   [--precomputed_features PRECOMPUTED_FEATURES]
                   [--pivot_features {none,all,data_only}] [--use_all_langs]
                   [--common_scaling] [--training_algorithm {xgboost,mlp}]
                   [--error_method {LOO,LOTO,split,kfold,manual_split}]
                   [--data_sizes DATA_SIZES] [--mode MODE [MODE ...]]
                   [--output_dir OUTPUT_DIR]
                   [--heatmap_targets HEATMAP_TARGETS]
                   [--suggestions_budget SUGGESTIONS_BUDGET]
                   [--suggestions_langbudget SUGGESTIONS_LANGBUDGET]
                   [--suggestions_targets SUGGESTIONS_TARGETS]
                   [--suggestions_weights SUGGESTIONS_WEIGHTS]
                   [--suggestions_pivots SUGGESTIONS_PIVOTS]
                   [--suggestions_augmentable SUGGESTIONS_AUGMENTABLE]
                   [--suggestions_grid {exponential,linear}]
                   [--suggestions_objective {avg,min}]
                   [--suggestions_minperf SUGGESTIONS_MINPERF]
                   [--suggestions_minlangperf SUGGESTIONS_MINLANGPERF]
                   [--suggestions_verbose]
                   {mbert,xlmr}

positional arguments:
  {mbert,xlmr}          name of model to use

optional arguments:
  -h, --help            show this help message and exit
  --scores_file SCORES_FILE
                        path of json file containing scores to train on
  --train_format {json,csv}
                        Format of the training data
  --save_state SAVE_STATE
                        Save state of training of model to pickle file
  --load_state LOAD_STATE
                        Load trained model from pickle file
  --precomputed_features PRECOMPUTED_FEATURES
                        Path to precomputed-features file.
  --pivot_features {none,all,data_only}
                        What features based on pivot langs to use
  --use_all_langs       Add features based on all langs the tool supports
                        (Needed for transfer)
  --common_scaling      Common min max scaling params that are pvt
                        dependent(data size, type overlap, distance)
  --training_algorithm {xgboost,mlp}
                        which regressor to use
  --error_method {LOO,LOTO,split,kfold,manual_split}

  --data_sizes DATA_SIZES
                        Pivot data-size configs (semi-colon separated configs,
                        each config itself being comma-separated key-value
                        pairs)

  --mode MODE [MODE ...]
                        Output modes (comma-separated). Choose from following:
                        {heatmap, suggestions}.
  --output_dir OUTPUT_DIR
                        Overrride output directory
  --heatmap_targets HEATMAP_TARGETS
                        Targets for heatmap. Overrides suggestions_targets
                        (which is used by deafult)

  --suggestions_budget SUGGESTIONS_BUDGET
                        Budget for finding suggestions of which languages to
                        add data for (0 to disable)
  --suggestions_langbudget SUGGESTIONS_LANGBUDGET
                        Language-specific budget for finding suggestions
                        (overrrides suggestions_budget for these langs, comma-
                        separated list of key:value pairs)
  --suggestions_targets SUGGESTIONS_TARGETS
                        Targets being considered (comma-separated)
  --suggestions_weights SUGGESTIONS_WEIGHTS
                        Target weights for avg perf objective (comma-separated
                        list of key:value pairs, default wt=1)
  --suggestions_pivots SUGGESTIONS_PIVOTS
                        Index of desired row in data_sizes
  --suggestions_augmentable SUGGESTIONS_AUGMENTABLE
                        Set of augmentable languages (comma-separated)
  --suggestions_grid {exponential,linear}
                        Search space grid to use for suggestions
  --suggestions_objective {avg,min}
                        Objective function to be used for finding suggestions
  --suggestions_minperf SUGGESTIONS_MINPERF
                        Minimum acceptable average performance across tgts
  --suggestions_minlangperf SUGGESTIONS_MINLANGPERF
                        Minimum acceptable performance for given tgts (comma-
                        separated list of key:value pairs)
  --suggestions_verbose
                        Verbose logging of search
```

## Examples

### From shell
```
python3 litmus_mixing.py xlmr --scores_file training_observations.json --common_scaling --error_method split --mode heatmap --data_sizes "en:1000,hi:1000;en:1000,ar:1000" --use_all_langs --heatmap_targets en,fr,de,hi,ar,ru
```

### From external scripts
```python
from litmus import litmus_mixing

data_file = "" # Location of train data file
args = litmus_mixing.parse_args([
    "xlmr", data_file,
    "--common_scaling",
    "--error_method", "kfold",
    "--training_algorithm", "xgboost"
])
res = litmus_mixing.litmus_main(args)
```

## WebApp
`frontend/` contains the code for hosting the tool as a webapp using Azure Functions. `frontend/WebUx` implements the client-side as a static website which interacts with a Azure Functions backend which internally runs the `litmus/litmus_mixing.py` script.

### Instructions to self-host

1. Create an Azure Functions resource on Azure.
2. Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/) and [Functions Core Tools](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=linux%2Ccsharp%2Cportal%2Cbash%2Ckeda)
3. `cd` into the `frontend/` directory and deploy to azure functions using `func azure functionapp publish <function-resource-name>`.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.
