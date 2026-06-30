import random
from typing import Dict, List

import numpy as np
import tensorflow as tf
from mlxtend.evaluate import mcnemar, mcnemar_table
from scipy.stats import wilcoxon

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_phase1: np.ndarray,
    y_pred_final: np.ndarray,
) -> Dict[str, float]:
    contingency = mcnemar_table(y_target=y_true, y_model1=y_pred_phase1, y_model2=y_pred_final)
    chi2, p_value = mcnemar(ary=contingency, corrected=True)
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "contingency_table": contingency.tolist(),
    }


def _paired_wilcoxon(
    first: List[float],
    second: List[float],
    apply_bonferroni: bool = False,
) -> Dict[str, float]:
    min_len = min(len(first), len(second))
    if min_len == 0:
        return {"statistic": float("nan"), "p_value": float("nan")}
    statistic, p_value = wilcoxon(np.array(first[:min_len]), np.array(second[:min_len]))
    corrected = min(p_value * 3, 1.0) if apply_bonferroni else p_value
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "corrected_p_value": float(corrected),
    }


def run_iou_wilcoxon_tests(localization_results: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    return {
        "Grad-CAM_vs_Grad-CAM++": _paired_wilcoxon(
            localization_results["Grad-CAM"]["ious"],
            localization_results["Grad-CAM++"]["ious"],
            apply_bonferroni=True,
        ),
        "Grad-CAM_vs_Integrated Gradients": _paired_wilcoxon(
            localization_results["Grad-CAM"]["ious"],
            localization_results["Integrated Gradients"]["ious"],
            apply_bonferroni=True,
        ),
        "Grad-CAM++_vs_Integrated Gradients": _paired_wilcoxon(
            localization_results["Grad-CAM++"]["ious"],
            localization_results["Integrated Gradients"]["ious"],
            apply_bonferroni=True,
        ),
    }


def run_stability_wilcoxon_tests(stability_results: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    return {
        "Grad-CAM_vs_Grad-CAM++": _paired_wilcoxon(
            stability_results["Grad-CAM"]["ssim_scores"],
            stability_results["Grad-CAM++"]["ssim_scores"],
        ),
        "Grad-CAM_vs_Integrated Gradients": _paired_wilcoxon(
            stability_results["Grad-CAM"]["ssim_scores"],
            stability_results["Integrated Gradients"]["ssim_scores"],
        ),
        "Grad-CAM++_vs_Integrated Gradients": _paired_wilcoxon(
            stability_results["Grad-CAM++"]["ssim_scores"],
            stability_results["Integrated Gradients"]["ssim_scores"],
        ),
    }
