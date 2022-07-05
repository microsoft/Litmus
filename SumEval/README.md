# SumEval

Helper Code and Datasets for the 1st Workshop on Scaling Up Multilingual Evaluation (SUMEval).

## Submission Instruction
The test files containing the training configurations and languages for which the predictions are to be made are located in [`data/test_release/`](data/test_release/)

We have three types of test files (in most cases) for every dataset-model pair which are:
- Test sets containing new configurations but same languages as seen in the training data. These are often denoted without any suffix, for eg: [`data/test_release/XNLI_XLMR.json`](data/test_release/XNLI_XLMR.json)
- Test sets containing new languages aka surprise languages but same configurations as the ones seen during training. These are denoted by the suffix '_surprise_langs_same_configs', for eg: [`data/test_release/XNLI_XLMR_surprise_langs_same_configs.json`](data/test_release/XNLI_XLMR_surprise_langs_same_configs.json)
- Test sets containing surprise languages as well as new configurations. These are denoted by the suffix '_surprise_langs_diff_configs', for eg: [`data/test_release/XNLI_XLMR_surprise_langs_diff_configs.json`](data/test_release/XNLI_XLMR_surprise_langs_diff_configs.json)

All the test files are of the following format:

```json
[
  {
    "train_config": {
      "<train_lang_1>": "<Size(train_lang_1)>",
      .,
      .,
      .,
      "<train_lang_n>": "<Size(train_lang_n)>",
    },
    "eval_results" : {
      "<eval_lang_1>" : "x",
      .,
      .,
      .,
      "<eval_lang_m>" : "x",
    }
  
  }

]
```

We ask the participants to predict the `"x"` values in these files by training predictor models on the training data, and replacing `"x"` with the predicted values in these files. For instance one can generate the predictions using the LITMUS predictor baseline by running:

```bash
python -m src.sum_eval_predict --data_dir ./data --out_dir ./Baselines
```

This will generate predictions for each file in the `./Baselines/preds` directory.

Once the predictions are generated they can be combined together to be compatible with [Explainaboard](https://explainaboard.inspiredco.ai/) by running:

```bash
python src/combine_predictions.py --pred_dir Baselines/preds --out_dir Baselines/combined_pred --value_name predicted_value
```

This will generate two files namely `Baselines/combined_pred/sumeval_test.json` and `Baselines/combined_pred/sumeval_surprise.json` which can be uploaded to Explainaboard for evaluation. The former will combine predictions for the test files not involving any surpise languages while the latter as the name suggests involve combining predictions for test data with surprise languages (for both same and diff versions)

### Explainaboard Submission instructions

The two files generate above should be uploaded to explainaboard (as seperate submissions) by following the steps below:
1. Visit the Explainaboard [link](https://explainaboard.inspiredco.ai/) and signup/login
2. Go to "Systems" and click "New"
3. Under Task select "tabular-regression" and select "sumeval2022" as dataset
4. Select "test" as split while submitting `sumeval_test.json` and "surprise" for sumeval_surprise.json`
5. Select "RMSE" and "AbsoluteError" as the metrics
6. Upload `sumeval_test.json` or `sumeval_surprise.json` (according to the split selected in step 4) by clicking on "Select File"
7. For your final submissions that you want to be considered, uncheck "Make it private?" 

This shall upload your submissions on the explainaboard which should appear [here](https://explainaboard.inspiredco.ai/leaderboards?dataset=sumeval2022). The AbsoluteError and RMSE columns should reveal the average errors across all the test sets. For a fine-grained analaysis click on the "Analysis" button.

Contact Kabir (t-kabirahuja@microsoft.com) if there are any questions.
