"""Alternative held-out architecture families, used by the LLM runs (gpu/run_ablation_phase.py, env FOLD) and the
CPU baselines (fold_tabular.py). The main split (config.py) holds out "other".

cnn is not held out (59% of all runs, 73% of the training split). When another family is held out, "other" (the main
test family) moves into training. Holding out segmentation removes image segmentation from training, as the main split
already does for text generation; object detection is never in training, since detector is a validation family otherwise.
"""
FOLDS = {
    "tf": dict(TRAIN_ARCH_FAMILIES=frozenset({"cnn", "segmentation", "other"}),
               VAL_ARCH_FAMILIES=frozenset({"detector", "rnn"}), TEST_ARCH_FAMILIES=frozenset({"transformer"})),
    "rnn": dict(TRAIN_ARCH_FAMILIES=frozenset({"cnn", "segmentation", "other"}),
                VAL_ARCH_FAMILIES=frozenset({"detector", "transformer"}), TEST_ARCH_FAMILIES=frozenset({"rnn"})),
    "det": dict(TRAIN_ARCH_FAMILIES=frozenset({"cnn", "segmentation", "other"}),
                VAL_ARCH_FAMILIES=frozenset({"rnn", "transformer"}), TEST_ARCH_FAMILIES=frozenset({"detector"})),
    # mirrors the main split with "other" and segmentation swapped
    "seg": dict(TRAIN_ARCH_FAMILIES=frozenset({"cnn", "other"}),
                VAL_ARCH_FAMILIES=frozenset({"detector", "rnn", "transformer"}), TEST_ARCH_FAMILIES=frozenset({"segmentation"})),
}
