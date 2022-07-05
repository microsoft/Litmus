import argparse
import os
import pandas as pd
from litmus.litmus_mixing import parse_args, litmus_main


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required = True)
    parser.add_argument("--out_dir", required = True)
    parser.add_argument("--eval_type", default = "LOLO", type = str)
    args = parser.parse_args() 

    errors = {}

    #Obtain Baselines Errors for TyDiQA (In Distribution)
    print(f"xlmr --scores_file {args.data_dir}/TyDiQA_XLMR_ID.json --error_method {args.eval_type} --pivot_features all")
    config = parse_args(f"xlmr --scores_file {args.data_dir}/TyDiQA_XLMR_ID.json --error_method {args.eval_type} --pivot_features all".split())
    tydiqa_id_ret_val = litmus_main(config)
    errors["TyDiQA-ID"] = {
        "litmus_avg_error" : 100 * tydiqa_id_ret_val["error"],
        "litmus_std_error" : 100 * tydiqa_id_ret_val["std_error"],
        "averaging_baseline_avg_error" : 100 * tydiqa_id_ret_val["baseline_error"],
        "averaging_baseline_std_error" : 100 * tydiqa_id_ret_val["baseline_std_error"]
    }

    #Obtain Baselines Errors for TyDiQA (Out of Distribution)
    print(f"xlmr --scores_file {args.data_dir}/TyDiQA_XLMR_OOD.json --error_method {args.eval_type} --pivot_features all")
    config = parse_args(f"xlmr --scores_file {args.data_dir}/TyDiQA_XLMR_OOD.json --error_method {args.eval_type} --pivot_features all".split())
    tydiqa_ood_ret_val = litmus_main(config)
    errors["TyDiQA-OOD"] = {
        "litmus_avg_error" : 100 * tydiqa_ood_ret_val["error"],
        "litmus_std_error" : 100 * tydiqa_ood_ret_val["std_error"],
        "averaging_baseline_avg_error" : 100 * tydiqa_ood_ret_val["baseline_error"],
        "averaging_baseline_std_error" : 100 * tydiqa_ood_ret_val["baseline_std_error"]
    }


    # Obtain Errors for XNLI, UDPOS and TyDiQA
    for dataset in ["XNLI", "UDPOS", "WikiANN"]:
        print(f"xlmr --scores_file {args.data_dir}/{dataset}_XLMR.json --error_method {args.eval_type} --pivot_features all")
        config = parse_args(f"xlmr --scores_file {args.data_dir}/{dataset}_XLMR.json --error_method {args.eval_type} --pivot_features all".split())
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
    errors_df.to_csv(f"{args.out_dir}/baseline_errors_xlmr_{args.eval_type}.csv")

if __name__ == "__main__":
    main()