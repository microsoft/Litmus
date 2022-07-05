import os
import re
import sys
import time
import json
import pkgutil
import argparse
import collections
from tqdm import tqdm
import pickle
import copy
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.facecolor"] = "white"

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.neural_network
import pdb

# Stopgap to enable use as module & standalone script
if __name__ == "__main__":
    from config import (
        SUPPORTED_MODELS,
        SUPPORTED_MODES,
        SUPPORTED_DATA_FORMATS,
        MODEL2LANGS,
        LANG2INDEX,
    )

    PRECOMPUTED_FEATURE_FILE = "data/precomputed_features.json"
else:
    from litmus.config import (
        SUPPORTED_MODELS,
        SUPPORTED_MODES,
        SUPPORTED_DATA_FORMATS,
        MODEL2LANGS,
        LANG2INDEX,
    )

    PRECOMPUTED_FEATURE_FILE = ""


class Featurizer:
    def __init__(self, model, precomputed_feature_file, pivot_features):
        self.langs_list = MODEL2LANGS[model]
        self.pivot_features = pivot_features

        if precomputed_feature_file != "":
            with open(precomputed_feature_file) as fin:
                precomputed = json.load(fin)
        else:
            pc_data = pkgutil.get_data(__name__, "data/precomputed_features.min.json")
            precomputed = json.loads(pc_data.decode("utf-8"))

        self.precomputed_type_overlaps = precomputed["type_overlap"][model]
        self.precomputed_syntactic_distance = precomputed["syntactic_distance"][model]
        self.regression_feats = pd.DataFrame(
            precomputed["regression_feats"][model], index=self.langs_list
        )

    def featurize(self, langs_target, langs_pivot, task_sizes):
        syntactic_distance = [
            [
                self.precomputed_syntactic_distance[self.langs_list.index(pivot)][
                    self.langs_list.index(lang)
                ]
                for lang in langs_target
            ]
            for pivot in langs_pivot
        ]
        type_overlaps = [
            [self.precomputed_type_overlaps[pivot][lang] for lang in langs_target]
            for pivot in langs_pivot
        ]

        pivot_fts = [
            self.regression_feats.loc[langs_target].values,
        ]
        if self.pivot_features == "data_only":
            for i in range(len(langs_pivot)):
                if task_sizes[i] == 0:
                    syntactic_distance[i] = [0 for _ in range(len(langs_target))]
                    type_overlaps[i] = [0 for _ in range(len(langs_target))]
        if self.pivot_features != "none":
            pivot_fts.append(np.array(type_overlaps).transpose())
            pivot_fts.append(np.array(syntactic_distance).transpose())
        task_size = [task_sizes for i in range(len(langs_target))]
        pivot_fts.append(np.array(task_size).transpose(0, 1))

        return np.concatenate(pivot_fts, axis=1)


def pretty_print(list_to_print, sep=" "):
    list_to_print = [
        [
            str(x) if not isinstance(x, (np.floating, float)) else "{0:0.4f}".format(x)
            for x in t
        ]
        for t in list_to_print
    ]
    n_cols = len(list_to_print[0])
    max_l = [max(len(t[i]) for t in list_to_print) for i in range(n_cols)]
    format_string = ["{:<" + str(m) + "}" for m in max_l[:-1]]
    format_string.append("{}")
    format_string = sep.join(format_string)
    for i in range(len(list_to_print)):
        print(format_string.format(*list_to_print[i]))


class CustomMinMaxScaler(sklearn.preprocessing.MinMaxScaler):
    def _handle_zeros_in_scale(scale, copy=True):
        # if we are fitting on 1D arrays, scale might be a scalar
        if np.isscalar(scale):
            if scale == 0.0:
                scale = 1.0
            return scale
        elif isinstance(scale, np.ndarray):
            if copy:
                # New array to avoid side-effects
                scale = scale.copy()
            scale[scale == 0.0] = 1.0
            return scale

    def __init__(self, common_feats=None, **kwargs):
        super().__init__(**kwargs)
        self.common_feats = common_feats

    def fit(self, X, y=None):
        first_pass = not hasattr(self, "n_samples_seen_")
        if not first_pass:
            return self
        else:
            return super().fit(X, y)

    def partial_fit(self, X, y=None):
        super().partial_fit(X, y)

        if self.common_feats:
            for group in self.common_feats:
                group = np.array(group)
                self.data_min_[group] = np.min(self.data_min_[group])
                self.data_max_[group] = np.max(self.data_max_[group])
                self.data_range_[group] = self.data_max_[group] - self.data_min_[group]

                self.scale_[group] = (
                    self.feature_range[1] - self.feature_range[0]
                ) / CustomMinMaxScaler._handle_zeros_in_scale(self.data_range_[group])
                self.min_[group] = (
                    self.feature_range[0] - self.data_min_[group] * self.scale_[group]
                )

        return self


def regression(X, Y, common_feats, training_algorithm, load_model, model):
    fit_kwargs = {}
    if model or load_model:
        if load_model:
            with open(load_model, "rb") as f:
                model = pickle.load(f)
        elif model:
            model = copy.deepcopy(model)
        if training_algorithm == "mlp":
            model.named_steps["regressor"].warm_start = True
        elif training_algorithm == "xgboost":
            fit_kwargs["regressor__xgb_model"] = model.named_steps[
                "regressor"
            ].get_booster()
    else:
        if training_algorithm == "mlp":
            model = sklearn.pipeline.Pipeline(
                [
                    (
                        "scaler",
                        CustomMinMaxScaler(common_feats=common_feats, clip=True),
                    ),
                    ("regressor", sklearn.neural_network.MLPRegressor((50, 50))),
                ]
            )
        elif training_algorithm == "xgboost":
            model = sklearn.pipeline.Pipeline(
                [
                    (
                        "scaler",
                        CustomMinMaxScaler(common_feats=common_feats, clip=True),
                    ),
                    (
                        "regressor",
                        xgb.XGBRegressor(
                            objective="reg:squarederror",
                            learning_rate=0.1,
                            n_estimators=100,
                            max_depth=10,
                        ),
                    ),
                ]
            )
    if X.shape[0] > 0:
        model.fit(X, Y, **fit_kwargs)
    return model


def prepare_data(args):

    model = args.model_name
    langs_list = MODEL2LANGS[model]
    featurizer = Featurizer(model, args.precomputed_features, args.pivot_features)

    if args.use_all_langs:
        all_langs = set(MODEL2LANGS["mbert"]) | set(MODEL2LANGS["xlmr"])
    else:
        all_langs = set(langs_list)

    X_array, Y_array = [], []
    examples = []
    if args.train_format == "json":
        with open(args.scores_file) as f:
            scores = json.load(f)

        for entry in scores:
            train_langs = entry["train_config"].keys()
            entry["train_config"].update({k: 0 for k in all_langs - train_langs})

            train_config = collections.OrderedDict(
                sorted(entry["train_config"].items())
            )
            eval_results = collections.OrderedDict(
                sorted(entry["eval_results"].items())
            )
            pivots_to_delete = [x for x in train_config if x not in langs_list]
            targets_to_delete = [x for x in eval_results if x not in langs_list]
            [train_config.pop(key) for key in pivots_to_delete]
            [eval_results.pop(key) for key in targets_to_delete]

            X_array.append(
                featurizer.featurize(
                    list(eval_results.keys()),
                    list(train_config.keys()),
                    list(train_config.values()),
                )
            )
            Y_array.append(list(eval_results.values()))
            examples.extend([[train_config, t, l] for l, t in eval_results.items()])

        langs_pivot = list(train_config.keys())
        langs_target = list(eval_results.keys())

    elif args.train_format == "csv":
        df = pd.read_csv(args.scores_file)
        langs_target = set()
        for i, row in df.iterrows():
            if row["target_lang"] not in langs_list:
                continue
            train_langs = row["train_langs"].split(",")
            data_sizes = [float(x) for x in row["train_data_sizes"].split(",")]
            values = {k: v for k, v in zip(train_langs, data_sizes)}
            values.update({k: 0 for k in all_langs - set(train_langs)})
            train_config = collections.OrderedDict(sorted(values.items()))
            pivots_to_delete = [x for x in train_config if x not in langs_list]
            [train_config.pop(key) for key in pivots_to_delete]

            langs_target.add(row["target_lang"])

            X_array.append(
                featurizer.featurize(
                    [row["target_lang"]],
                    list(train_config.keys()),
                    list(train_config.values()),
                )
            )
            Y_array.append(row["score"])

        langs_target = list(langs_target)
        langs_pivot = list(train_config.keys())

    # Reshape datasets
    X = np.concatenate(X_array)
    Y = np.array(Y_array).reshape(-1)

    # Establish feature-set
    feat_names = ["Data Size", "Well rep features"]
    common_feats = []
    if args.pivot_features != "none":
        feat_names += ["Type Overlap {}".format(lang) for lang in langs_pivot] + [
            "Syntactic Distance {}".format(lang) for lang in langs_pivot
        ]
        if args.common_scaling:
            start = 2
            mid = start + len(langs_pivot)
            end = mid + len(langs_pivot)
            common_feats.append([_ for _ in range(start, mid)])
            common_feats.append([_ for _ in range(mid, end)])
    feat_names += ["Task data size {}".format(lang) for lang in langs_pivot]
    if args.common_scaling:
        start = 2 + (2 * len(langs_pivot) if args.pivot_features != "none" else 0)
        end = start + len(langs_pivot)
        common_feats.append([_ for _ in range(start, end)])

    return (
        model,
        langs_list,
        feat_names,
        featurizer,
        common_feats,
        X,
        Y,
        langs_pivot,
        langs_target,
        examples,
    )


"""
    Parse user-specified pivot-size configurations
"""


def prepare_inf_data(args, langs_list):
    data_sizes = args.data_sizes
    if isinstance(data_sizes, str) and data_sizes != "":
        sizes = re.sub(r"\s+", "", data_sizes)
        data_sizes = []
        for row in sizes.split(";"):
            sizes_map = {
                lang: int(size)
                for el in row.split(",")
                for lang, size in [el.split(":")]
            }
            data_sizes += [[sizes_map.get(lang, 0) for lang in langs_list]]
        data_sizes = np.array(data_sizes)

    tgt_langs = (
        args.heatmap_targets if args.heatmap_targets else args.suggestions_targets
    )
    tgt_langs = re.sub(r"\s+", "", tgt_langs).split(",")
    # pdb.set_trace()

    return data_sizes, tgt_langs


def train_model(
    X, Y, feat_names, common_feats, langs_pivot, langs_target, examples, args
):

    error_method = args.error_method
    bprint = args.print
    train_indices = args.train_indices
    test_indices = args.test_indices

    if args.model and args.load_model:
        raise ValueError("Cannot specify model load_model. Only specify one")

    model = regression(
        X, Y, common_feats, args.training_algorithm, args.load_model, args.model
    )

    avg_error, std_error = None, None
    num_targets = len(langs_target)
    num_pivots = len(langs_pivot)
    train_function = regression
    examples_indices = None
    predictions = None

    if error_method == "split":
        # Split examples into train and test split to compute error on test split alone
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            X, Y, random_state=42
        )
        res1 = train_function(
            X_train,
            Y_train,
            common_feats,
            args.training_algorithm,
            args.load_model,
            args.model,
        )
        Y_pred = res1.predict(X_test)
        errors = abs(Y_test - Y_pred)
        baseline_errors = abs(np.mean(Y_train) - Y_test)

    elif error_method == "kfold":
        # Use 5 fold CV to compute error over all examples
        kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
        errors = []
        predictions = []
        examples_indices = []
        baseline_errors = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            res1 = train_function(
                X_train,
                Y_train,
                common_feats,
                args.training_algorithm,
                args.load_model,
                args.model,
            )
            Y_pred = res1.predict(X_test)
            error = abs((Y_pred - Y_test))
            errors.extend(error)
            examples_indices.extend(test_index)
            predictions.extend(Y_pred)
            baseline_errors.extend(abs(np.mean(Y_train) - Y_test))

    elif error_method == "manual_split":
        # Use train and test splits supplied in args
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]
        res1 = train_function(
            X_train,
            Y_train,
            common_feats,
            args.training_algorithm,
            args.load_model,
            args.model,
        )
        Y_pred = res1.predict(X_test)
        errors = abs(Y_test - Y_pred)
        baseline_errors = abs(np.mean(Y_train) - Y_test)

    elif error_method == "LOTO":
        # Leave one target out scenario, predict for the left out column of elements and compute errors
        errors = []
        baseline_errors = []
        for t in tqdm(range(num_targets)):
            indices_to_delete = [_ * num_targets + t for _ in range(num_pivots)]
            X_reg = np.delete(X, indices_to_delete, axis=0)
            Y_reg = np.delete(Y, indices_to_delete, axis=0)
            Y_gold = Y[indices_to_delete]
            res1 = regression(
                X_reg,
                Y_reg,
                common_feats,
                args.training_algorithm,
                args.load_model,
                args.model,
            )
            Y_pred = res1.predict(X[indices_to_delete])

            error = abs((Y_gold - Y_pred))
            errors.extend(error)
            baseline_errors.extend(abs(np.mean(Y_reg) - Y_gold))

    elif error_method == "LOO":
        # Leave one out scenario, predict for the left out element and compute errors
        errors = []
        baseline_errors = []
        for _ in tqdm(range(X.shape[0])):
            X_reg = np.delete(X, _, axis=0)
            Y_reg = np.delete(Y, _, axis=0)
            Y_gold = Y[_]
            res1 = regression(
                X_reg,
                Y_reg,
                common_feats,
                args.training_algorithm,
                args.load_model,
                args.model,
            )
            Y_pred = res1.predict(X[_].reshape(1, -1))[0]

            error = abs((Y_gold - Y_pred))
            errors.append(error)
            baseline_errors.append(abs(np.mean(Y_reg) - Y_gold))

    elif error_method == "LOCO":
        # Leave one configuration out scenario
        errors = []
        predictions = []
        examples_indices = []
        baseline_errors = []
        num_tgts = len(langs_target)
        for i in tqdm(range(0, len(examples), num_tgts)):
            test_index = list(range(i, i + num_tgts))
            train_index = list(range(i)) + list(range(i + num_tgts, len(examples)))
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            res1 = train_function(
                X_train,
                Y_train,
                common_feats,
                args.training_algorithm,
                args.load_model,
                args.model,
            )
            Y_pred = res1.predict(X_test)
            error = abs((Y_pred - Y_test))
            errors.extend(error)
            examples_indices.extend(test_index)
            predictions.extend(Y_pred)
            baseline_errors.extend(abs(np.mean(Y_train) - Y_test))

    elif error_method == "LOLO":
        # Leave one language out scenario, remove all instances of a language (in train_config as well as results) and move to test.
        errors = []
        predictions = []
        examples_indices = []
        baseline_errors = []
        for lang in tqdm(langs_target):
            train_index, test_index = [], []
            for i, example in enumerate(examples):
                # If language is present neither as a pivot or target
                if lang != example[-1] and example[0][lang] == 0:
                    train_index.append(i)

                # If language is present as a target put it in test
                elif lang == example[-1]:
                    test_index.append(i)

            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            res1 = train_function(
                X_train,
                Y_train,
                common_feats,
                args.training_algorithm,
                args.load_model,
                args.model,
            )
            Y_pred = res1.predict(X_test)
            error = abs((Y_pred - Y_test))
            errors.extend(error)
            examples_indices.extend(test_index)
            predictions.extend(Y_pred)
            baseline_errors.extend(abs(np.mean(Y_train) - Y_test))

    avg_error, std_error = np.mean(errors), np.std(errors)
    baseline_error, baseline_std_error = np.mean(baseline_errors), np.std(
        baseline_errors
    )
    if bprint:
        print("Avg Pred Error: {0:0.6f}".format(avg_error))
        print("Std Pred Error: {0:0.6f}".format(std_error))

        print("Baseline Error: {0:0.6f}".format(baseline_error))
        print("Std Baseline Error: {0:0.6f}".format(baseline_std_error))

    if args.save_model:
        with open(args.save_model, "wb") as f:
            pickle.dump(model, f)

    return (
        model,
        (avg_error, std_error),
        (baseline_error, baseline_std_error),
        errors,
        examples_indices,
        predictions,
    )


def build_acc_matrix(langs_pivot, langs_target, featurizer, model, data_sizes):
    X = np.concatenate(
        [
            featurizer.featurize(langs_target, langs_pivot, data_sizes[_])
            for _ in range(len(data_sizes))
        ]
    )
    Y = model.predict(X)
    Y = Y.reshape(len(data_sizes), len(langs_target))
    return Y


"""
    Enforcing language-sparsity constraint on search space
"""


def find_suggestions(
    args,
    model,
    featurizer,
    langs_list,
    budget,
    lang_budgets,
    targets,
    pivots,
    augmentable,
    weights,
    data_sizes,
    is_exp_grid,
    objective,
    min_perf,
    min_lang_perf,
):

    # Helpers
    rmSpaces = lambda s: re.sub(r"\s+", "", s)
    parseLangData = lambda s, t: {
        kv.split(":")[0]: t(kv.split(":")[1])
        for kv in rmSpaces(s).split(",")
        if kv != ""
    }
    printInfo = lambda s: print(s) if args.suggestions_verbose else None

    # Parse configuration
    targets, augmentable = rmSpaces(targets), rmSpaces(augmentable)
    targets = set(targets.split(","))
    augmentable = set(augmentable.split(",")) if augmentable != "" else targets
    weights = parseLangData(weights, int)
    lang_budgets = parseLangData(lang_budgets, int)
    min_perf = float(min_perf) if min_perf.strip() != "" else 0
    min_lang_perf = parseLangData(min_lang_perf, float)
    assert len(targets) > 0 and len(pivots) > 0 and len(augmentable) > 0

    langs_list = sorted(langs_list)
    langs_pvts = [lang for lang in langs_list]
    langs_tgts = [lang for lang in langs_list if lang in targets]
    langs_wts = [weights.get(lang, 1) for lang in langs_tgts]
    orig_sizes = tuple(pivots)

    # Helpers
    TgtPerfs = lambda sizes: model.predict(
        featurizer.featurize(langs_tgts, langs_pvts, sizes)
    )
    if objective == "avg":
        AvgPerf = lambda sizes: np.average(TgtPerfs(sizes), weights=langs_wts)
    else:
        AvgPerf = lambda sizes: np.amin(TgtPerfs(sizes))
    SimpleAugSet = lambda sizes: [
        (lang, sizes[idx]) for idx, lang in enumerate(langs_pvts) if sizes[idx] > 0
    ]
    baseline_perf = AvgPerf(np.array(orig_sizes))
    grid_linear_step = [
        0.1 * lang_budgets.get(lang, budget) for idx, lang in enumerate(langs_pvts)
    ]

    class SearchNode:
        ctr_eval = 0

        def __init__(self, sizes):
            self.sizes = sizes
            self.is_terminal = sum(sizes) <= budget
            self.score = None
            self.perf = None
            self.tgt_perfs = None

        def Eval(self):
            if self.score == None:
                SearchNode.ctr_eval += 1
                self.score = self.Perf()
            return self.score

        def Perf(self):
            if self.perf == None:
                self.tgt_perfs = TgtPerfs(np.array(orig_sizes) + np.array(self.sizes))
                self.perf = AvgPerf(np.array(orig_sizes) + np.array(self.sizes))
            return self.perf

        def Augment(self, lidx):
            if self.sizes[lidx] == 0:
                return None
            if is_exp_grid:
                new_sizes = tuple(
                    [
                        el // 2 if idx == lidx else el
                        for idx, el in enumerate(self.sizes)
                    ]
                )
            else:
                new_sizes = tuple(
                    [
                        el - grid_linear_step[idx] if idx == lidx else el
                        for idx, el in enumerate(self.sizes)
                    ]
                )
            return SearchNode(new_sizes)

        def Expand(self):
            if self.is_terminal:
                return [self]

            frontier = [
                augNode
                for idx, lang in enumerate(langs_pvts)
                if lang in augmentable
                for augNode in [self.Augment(idx)]
                if augNode != None
            ]
            if not len(frontier):
                self.is_terminal = True
                return [self]
            else:
                return frontier

        def __hash__(self):
            return hash((self.sizes, self.is_terminal))

        def __eq__(self, other):
            return (
                other != None
                and self.sizes == other.sizes
                and self.is_terminal == other.is_terminal
            )

        def Print(self):
            print(tuple(SimpleAugSet(self.sizes)), self.score, self.is_terminal)

    # Search params
    BEAM_WIDTH = 5
    PRUNE_LANG_QLTY_THRESHOLD = 0.02

    t0 = time.time()

    # Actual search
    # Using list representation for beam, sorting beam is cheap as beam-width is small :P
    start_size = budget // 2 if is_exp_grid else budget
    theoretic_max_case = tuple(
        [
            lang_budgets.get(lang, start_size) if lang in augmentable else 0
            for idx, lang in enumerate(langs_pvts)
        ]
    )

    beam = [SearchNode(theoretic_max_case)]
    while any(not node.is_terminal for node in beam):
        beam = [f for node in beam for f in node.Expand()]
        beam = list(set(beam))

        # batch mode scoring
        inps = np.concatenate(
            [
                featurizer.featurize(
                    langs_tgts, langs_pvts, np.array(orig_sizes) + np.array(node.sizes)
                )
                for node in beam
            ]
        )
        outs = model.predict(inps).reshape(len(beam), len(langs_tgts))
        if objective == "avg":
            scores = np.average(outs, axis=1, weights=langs_wts)
        else:
            scores = np.amin(outs, axis=1)

        for idx, score in enumerate(list(scores)):
            beam[idx].score = score
            beam[idx].tgt_perfs = outs[idx, :]
        SearchNode.ctr_eval += len(beam)

        # Apply constraints on beam candidates
        beam = [
            node
            for node in beam
            if node.Eval() >= min_perf
            and all(
                lang_perf >= min_lang_perf[lang]
                for lang, lang_perf in zip(langs_tgts, node.tgt_perfs)
                if lang in min_lang_perf
            )
        ]

        # Retain top-K candidates
        beam = sorted(beam, key=lambda x: x.Eval(), reverse=True)
        beam = beam[:BEAM_WIDTH]
        if args.suggestions_verbose:
            print("Beam:")
            [node.Print() for node in beam]

    if len(beam) == 0:
        best = None
    else:
        # Cleanup best solution:
        # Remove lowest aug langs iteratively as long as drop in gains is less than 5%
        best = beam[0]
        best_gains = best.Perf() - baseline_perf
        sorted_aug_langs = sorted(
            [(langs_pvts[idx], size, idx) for idx, size in enumerate(best.sizes)],
            key=lambda x: x[1],
        )
        for aug_lang, aug_size, aug_idx in sorted_aug_langs:
            while best.sizes[aug_idx] > 0:
                curr = best.Augment(aug_idx)
                curr_gains = curr.Perf() - baseline_perf
                printInfo(
                    "Attempting reduction of lang (%s, %d, %d) with gains delta (%.4f - %.4f)..."
                    % (aug_lang, curr.sizes[aug_idx], aug_idx, curr_gains, best_gains)
                )
                if (
                    curr_gains > best_gains
                    or 1.0 - curr_gains / best_gains < PRUNE_LANG_QLTY_THRESHOLD
                ):
                    printInfo("Reduced lang %s..." % aug_lang)
                    best = curr
                else:
                    break

    t1 = time.time()

    equal_aug_sizes = np.array(orig_sizes) + np.array(
        [
            (budget // len(augmentable)) if lang in augmentable else 0
            for lang in langs_pvts
        ]
    )
    return {
        "search-stats": {
            "num_nodes_searched": SearchNode.ctr_eval,
            "time-taken": t1 - t0,
            "budget": budget,
            "used_budget": sum(best.sizes) if best != None else 0,
        },
        "search-perfs": {
            "baseline-perf": AvgPerf(orig_sizes),
            "augmented-perf": best.Perf() if best != None else 0,
            "max-perf": AvgPerf(np.array(orig_sizes) + np.array(theoretic_max_case)),
            "equal-aug-perf": AvgPerf(equal_aug_sizes),
        },
        "lang-perfs": {
            "before-aug": {
                tgt: score for tgt, score in zip(langs_tgts, TgtPerfs(orig_sizes))
            },
            "after-aug": {
                tgt: score
                for tgt, score in zip(
                    langs_tgts, TgtPerfs(np.array(orig_sizes) + np.array(best.sizes))
                )
            }
            if best != None
            else {},
            "equal-aug": {
                tgt: score for tgt, score in zip(langs_tgts, TgtPerfs(equal_aug_sizes))
            },
        },
        "augments": SimpleAugSet(best.sizes) if best != None else {},
        "ablation": {
            langs_pvts[idx]: best.Augment(idx).Perf() - best.Perf()
            for idx, size in enumerate(best.sizes)
            if size > 0
        }
        if best != None
        else {},
    }


"""
    Returns predicted performance in targets for diff pivot data_sizes
"""


def get_user_config_perfs(model, featurizer, langs_list, targets, data_sizes):

    # Parse params
    # pdb.set_trace()
    targets = set(targets)
    langs_tgts = [lang for lang in langs_list if lang in targets]
    langs_pvts = langs_list

    # Helpers
    TgtPerfs = lambda sizes: model.predict(
        featurizer.featurize(langs_tgts, langs_pvts, sizes)
    )
    AvgPerf = lambda sizes: np.average(TgtPerfs(sizes))
    SimpleAugSet = lambda sizes: [
        (lang, sizes[idx]) for idx, lang in enumerate(langs_pvts) if sizes[idx] > 0
    ]

    # Add tgt-score-distrib for best performing user-specified config
    best_user_config, best_user_config_idx, _ = max(
        [
            (user_config, row_idx, AvgPerf(user_config))
            for row_idx, user_config in enumerate(data_sizes.tolist())
        ],
        key=lambda x: x[2],
    )
    best_user_config_perfs = TgtPerfs(best_user_config)
    tgt_perfs = [
        TgtPerfs(user_config) for row_idx, user_config in enumerate(data_sizes.tolist())
    ]

    return {
        "best-config-idx": best_user_config_idx,
        "best-config": SimpleAugSet(best_user_config),
        "best-tgt-perfs": list(zip(langs_tgts, best_user_config_perfs)),
        "tgt_perfs": tgt_perfs,
    }


"""
    Plots heatmap of predicted performance in langs_tgts for given data_sizes
"""


def plot_perf_heatmap(
    langs_list, langs_pivot, langs_tgts, data_sizes, model, featurizer, output_dir
):
    langs_tgts = [t for t in langs_tgts if t in langs_list]
    Y = build_acc_matrix(langs_pivot, langs_tgts, featurizer, model, data_sizes)

    print("vsize", data_sizes.shape[0])
    fig = plt.figure(figsize=(6, 3), dpi=300)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)

    ax = sns.heatmap(
        Y,
        cmap="RdYlGn",
        cbar=True,
        cbar_kws={"orientation": "horizontal"},
        yticklabels=[
            "Config-%d" % (ConfigIdx + 1) for ConfigIdx in range(len(data_sizes))
        ],
        xticklabels=langs_tgts,
    )
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    pltfile = "{}.png".format(int(time.time()))
    if output_dir:
        pltfile = os.path.join(output_dir, pltfile)
    plt.savefig(pltfile, bbox_inches="tight")

    return pltfile, Y


def litmus_main(args):

    """
    Prepare predictive model
    """
    assert args.load_state or args.scores_file
    if args.load_state:
        # Load trained model + metadata
        with open(args.load_state, "rb") as pkl_f:
            (
                model,
                featurizer,
                langs_list,
                langs_pivot,
                langs_target,
                avg_error,
                baseline_error,
            ) = pickle.load(pkl_f)
    else:
        # Prepare data
        (
            model,
            langs_list,
            feat_names,
            featurizer,
            common_feats,
            X,
            Y,
            langs_pivot,
            langs_target,
            examples,
        ) = prepare_data(args)
        # Train prediction model
        (
            model,
            (avg_error, std_error),
            (baseline_error, baseline_std),
            errors,
            examples_indices,
            predictions,
        ) = train_model(
            X, Y, feat_names, common_feats, langs_pivot, langs_target, examples, args
        )

    if args.save_state:
        with open(args.save_state, "wb") as pkl_f:
            pickle.dump(
                [
                    model,
                    featurizer,
                    langs_list,
                    langs_pivot,
                    langs_target,
                    avg_error,
                    baseline_error,
                ],
                pkl_f,
            )

    ret_val = {
        "error": avg_error,  # Average Error Computed by specified Method
        "std_error": std_error,  # Standard Deviation of Predictor's Error
        "langs_pivot": langs_pivot,  # Pivot Languages
        "langs_target": langs_target,  # Target Languages
        "baseline_error": baseline_error,  # Error when using Mean Baseline Method instead of training model
        "baseline_std_error": baseline_std,  # Std of Error of mean baseline
    }

    """
        Inference using trained model
    """
    if args.mode:

        data_sizes, langs_tgts = prepare_inf_data(args, langs_list)

        # Set basline perfs for all target modes
        ret_val["user-config-perfs"] = get_user_config_perfs(
            model, featurizer, langs_list, langs_tgts, data_sizes
        )
        pprint(ret_val["user-config-perfs"])

        if "heatmap" in args.mode:
            pltfile, Y = plot_perf_heatmap(
                langs_list,
                langs_pivot,
                langs_tgts,
                data_sizes,
                model,
                featurizer,
                args.output_dir,
            )

            ret_val["heatmapFile"] = pltfile
            ret_val["acc_matrix"] = {"index": langs_list, "matrix": Y.tolist()}

        if "suggestions" in args.mode:

            # Get baseline row for pivot sizes
            pivot_row_idx = (
                int(args.suggestions_pivots)
                if args.suggestions_pivots != ""
                else ret_val["user-config-perfs"]["best-config-idx"]
            )
            pivot_sizes = data_sizes[pivot_row_idx]

            ret_val["suggestions"] = find_suggestions(
                args,
                model,
                featurizer,
                langs_list,
                args.suggestions_budget,
                args.suggestions_langbudget,
                args.suggestions_targets,
                pivot_sizes,
                args.suggestions_augmentable,
                args.suggestions_weights,
                data_sizes,
                args.suggestions_grid == "exponential",
                args.suggestions_objective,
                args.suggestions_minperf,
                args.suggestions_minlangperf,
            )
            ret_val["suggestions"]["suggestions_row"] = pivot_row_idx
            pprint(ret_val["suggestions"])

    return ret_val


def parse_args(args):
    parser = argparse.ArgumentParser("LITMUS Tool")

    # Options for loading training data / model
    parser.add_argument(
        "model_name",
        type=str,
        default="xlmr",
        help="name of model to use",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--scores_file",
        default=None,
        type=str,
        help="path of json file containing scores to train on",
    )
    parser.add_argument(
        "--train_format",
        default="json",
        help="Format of the training data",
        choices=["json", "csv"],
    )
    parser.add_argument(
        "--save_state",
        default=None,
        type=str,
        help="Save state of training of model to pickle file",
    )
    parser.add_argument(
        "--load_state",
        default=None,
        type=str,
        help="Load trained model from pickle file",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="Test file to generate prediction. If `None` then no predictions are made",
    )
    # Feature options
    parser.add_argument(
        "--precomputed_features",
        type=str,
        default=PRECOMPUTED_FEATURE_FILE,
        help="Path to precomputed-features file.",
    )
    parser.add_argument(
        "--pivot_features",
        type=str,
        default="none",
        choices=["none", "all", "data_only"],
        help="What features based on pivot langs to use",
    )
    parser.add_argument(
        "--use_all_langs",
        action="store_true",
        help="Add features based on all langs the tool supports (Needed for transfer)",
    )
    parser.add_argument(
        "--common_scaling",
        action="store_true",
        help="Common min max scaling params that are pvt dependent(data size, type overlap, distance)",
    )

    # Model training options
    parser.add_argument(
        "--training_algorithm",
        type=str,
        default="xgboost",
        help="which regressor to use",
        choices=["xgboost", "mlp"],
    )
    parser.add_argument(
        "--error_method",
        type=str,
        default="split",
        choices=["LOO", "LOTO", "LOLO", "LOCO", "split", "kfold", "manual_split"],
    )
    parser.add_argument(
        "--dont_print",
        action="store_false",
        dest="print",
        help="disable any form of printing",
    )

    # Experimental Options
    parser.add_argument(
        "--train_indices",
        default=None,
        help="train indices if error_method is manual_split",
    )
    parser.add_argument(
        "--test_indices",
        default=None,
        help="test indices if error_method is manual_split",
    )
    parser.add_argument(
        "--model", default=None, help="Model to load"
    )  # hack - for calling from external script using actual model
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Load neural network model from saved file",
    )
    parser.add_argument(
        "--save_model", type=str, default=None, help="Save neural network model to file"
    )

    # Options for inferencing
    parser.add_argument(
        "--data_sizes",
        default="",
        help="Pivot data-size configs (semi-colon separated configs, each config itself being comma-separated key-value pairs)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        nargs="+",
        help="Output modes (comma-separated). Choose from following: {heatmap, suggestions}.",
    )

    # Options for heatmap comparison of multiple user-configs
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Overrride output directory"
    )
    parser.add_argument(
        "--heatmap_targets",
        type=str,
        default=None,
        help="Targets for heatmap. Overrides suggestions_targets (which is used by deafult)",
    )

    # Options for suggestions finding
    parser.add_argument(
        "--suggestions_budget",
        type=int,
        default=0,
        help="Budget for finding suggestions of which languages to add data for (0 to disable)",
    )
    parser.add_argument(
        "--suggestions_langbudget",
        type=str,
        default="",
        help="Language-specific budget for finding suggestions (overrrides suggestions_budget for these langs, comma-separated list of key:value pairs)",
    )
    parser.add_argument(
        "--suggestions_targets",
        type=str,
        default="",
        help="Targets being considered (comma-separated)",
    )
    parser.add_argument(
        "--suggestions_weights",
        type=str,
        default="",
        help="Target weights for avg perf objective (comma-separated list of key:value pairs, default wt=1)",
    )
    parser.add_argument(
        "--suggestions_pivots",
        type=str,
        default="",
        help="Index of desired row in data_sizes",
    )
    parser.add_argument(
        "--suggestions_augmentable",
        type=str,
        default="",
        help="Set of augmentable languages (comma-separated)",
    )
    parser.add_argument(
        "--suggestions_grid",
        type=str,
        default="exponential",
        choices=["exponential", "linear"],
        help="Search space grid to use for suggestions",
    )
    parser.add_argument(
        "--suggestions_objective",
        type=str,
        default="avg",
        help="Objective function to be used for finding suggestions",
        choices=["avg", "min"],
    )
    parser.add_argument(
        "--suggestions_minperf",
        type=str,
        default="",
        help="Minimum acceptable average performance across tgts",
    )
    parser.add_argument(
        "--suggestions_minlangperf",
        type=str,
        default="",
        help="Minimum acceptable performance for given tgts (comma-separated list of key:value pairs)",
    )
    parser.add_argument(
        "--suggestions_verbose", action="store_true", help="Verbose logging of search"
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    litmus_main(args)
