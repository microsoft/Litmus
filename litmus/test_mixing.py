import pytest
from litmus import litmus_mixing

def test_suggestions():

	args = litmus_mixing.parse_args([
	    "xlmr", 
	    "--load_state", "frontend/Predictivev2/data_xlmr_udpos.pkl",
	    "--common_scaling",
	    "--error_method", "kfold",
	    "--training_algorithm", "xgboost",
	    "--mode", "suggestions",
	    "--data_sizes", "ar:5000,ru:5000",
	    "--suggestions_targets", "en,fr,de,hi,ta,zh,ja",
	    "--suggestions_budget", "10000"
	])
	res = litmus_mixing.litmus_main(args)

	assert ("suggestions" in res)
	assert (len(res["suggestions"]["augments"]) > 0)