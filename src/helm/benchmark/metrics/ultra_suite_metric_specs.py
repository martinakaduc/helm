from typing import List
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_multiple_choice_classification_metric_specs,
)


def get_audio_classification_metric_specs() -> List[MetricSpec]:
    return get_multiple_choice_classification_metric_specs() + get_basic_metric_specs(
        ["exact_match", "quasi_exact_match"]
    )


def get_ultra_suite_asr_classification_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.ultra_suite_asr_classification_metrics.UltraSuiteASRMetric", args={}
        )
    ]
