import argparse
import os
import pandas as pd
from litmus.litmus_mixing import parse_args, litmus_main
PRECOMPUTED_FEATURE_FILE = "litmus/data/precomputed_features.json"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required = True)
    parser.add_argument("--out_dir", required = True)
    parser.add_argument("--eval_type", default = "LOLO", type = str)
    parser.add_argument("--model", default = "xlmr", choices = ["xlmr", "tulrv6"])
    args = parser.parse_args() 

    errors = {}
    model_str = "XLMR" if args.model == "xlmr" else "TULRv6Large"
    #Obtain Baselines Errors for TyDiQA (In Distribution)
    print(f"{args.model} --scores_file {args.data_dir}/TyDiQA_{model_str}_ID.json --error_method {args.eval_type} --pivot_features all --precomputed_features {PRECOMPUTED_FEATURE_FILE}")
    config = parse_args(f"{args.model} --scores_file {args.data_dir}/TyDiQA_{model_str}_ID.json --error_method {args.eval_type} --pivot_features all --precomputed_features {PRECOMPUTED_FEATURE_FILE}".split())
    tydiqa_id_ret_val = litmus_main(config)
    errors["TyDiQA-ID"] = {
        "litmus_avg_error" : 100 * tydiqa_id_ret_val["error"],
        "litmus_std_error" : 100 * tydiqa_id_ret_val["std_error"],
        "averaging_baseline_avg_error" : 100 * tydiqa_id_ret_val["baseline_error"],
        "averaging_baseline_std_error" : 100 * tydiqa_id_ret_val["baseline_std_error"]
    }

    #Obtain Baselines Errors for TyDiQA (Out of Distribution)
    print(f"{args.model} --scores_file {args.data_dir}/TyDiQA_{model_str}_OOD.json --error_method {args.eval_type} --pivot_features all --precomputed_features {PRECOMPUTED_FEATURE_FILE}")
    config = parse_args(f"{args.model} --scores_file {args.data_dir}/TyDiQA_{model_str}_OOD.json --error_method {args.eval_type} --pivot_features all --precomputed_features {PRECOMPUTED_FEATURE_FILE}".split())
    tydiqa_ood_ret_val = litmus_main(config)
    errors["TyDiQA-OOD"] = {
        "litmus_avg_error" : 100 * tydiqa_ood_ret_val["error"],
        "litmus_std_error" : 100 * tydiqa_ood_ret_val["std_error"],
        "averaging_baseline_avg_error" : 100 * tydiqa_ood_ret_val["baseline_error"],
        "averaging_baseline_std_error" : 100 * tydiqa_ood_ret_val["baseline_std_error"]
    }


    # Obtain Errors for XNLI, UDPOS and TyDiQA
    datasets = ["XNLI", "UDPOS", "WikiANN"] if args.model == "XLMR" else ["XNLI"]
    for dataset in datasets:
        print(f"{args.model} --scores_file {args.data_dir}/{dataset}_{model_str}.json --error_method {args.eval_type} --pivot_features all --precomputed_features {PRECOMPUTED_FEATURE_FILE}")
        config = parse_args(f"{args.model} --scores_file {args.data_dir}/{dataset}_{model_str}.json --error_method {args.eval_type} --pivot_features all --precomputed_features {PRECOMPUTED_FEATURE_FILE}".split())
        ret_val = litmus_main(config)
        errors[dataset] = {
            "litmus_avg_error" : 100 * ret_val["error"],
            "litmus_std_error" : 100 * ret_val["std_error"],
            "averaging_baseline_avg_error" : 100 * ret_val["baseline_error"],
            "averaging_baseline_std_error" : 100 * ret_val["baseline_std_error"]
        }
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    errors_df = pd.DataFrame(errors).transpose()
    errors_df.to_csv(f"{args.out_dir}/baseline_errors_{args.model}_{args.eval_type}.csv")

if __name__ == "__main__":
    main()