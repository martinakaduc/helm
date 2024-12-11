"""Run spec functions for DecodingTrust.

DecodingTrust aims at providing a thorough assessment of trustworthiness in language models.

Website: https://decodingtrust.github.io/
Paper: https://arxiv.org/abs/2306.11698"""

from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT
from helm.benchmark.adaptation.common_adapter_specs import (
    get_completion_adapter_spec,
    get_few_shot_instruct_adapter_spec,
    get_generation_adapter_spec,
    get_instruct_adapter_spec,
    get_multiple_choice_adapter_spec,
    get_language_modeling_adapter_spec,
    get_ranking_binary_adapter_spec,
    get_summarization_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs,
    get_bias_metric_specs,
    get_classification_metric_specs,
    get_copyright_metric_specs,
    get_disinformation_metric_specs,
    get_exact_match_metric_specs,
    get_f1_metric_specs,
    get_generative_harms_metric_specs,
    get_language_modeling_metric_specs,
    get_numeracy_metric_specs,
    get_open_ended_generation_metric_specs,
    get_summarization_metric_specs,
    get_basic_generation_metric_specs,
    get_basic_reference_metric_specs,
    get_generic_metric_specs,
)
from helm.benchmark.metrics.common_metric_specs import get_exact_match_metric_specs, get_generative_harms_metric_specs
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


def get_privacy_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.decodingtrust_privacy_metrics.PrivacyMetric", args={})]


def get_stereotype_bias_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.decodingtrust_stereotype_bias_metrics.StereotypeMetric", args={})
    ]


def get_fairness_metric_specs() -> List[MetricSpec]:
    return [MetricSpec(class_name="helm.benchmark.metrics.decodingtrust_fairness_metrics.FairnessMetric", args={})]


def get_ood_knowledge_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(class_name="helm.benchmark.metrics.decodingtrust_ood_knowledge_metrics.OODKnowledgeMetric", args={}),
    ]

@run_spec_function("villm_question_answering")
def get_villm_question_answering_spec(dataset: str = "mlqa", device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.villm_qa_scenario."
        "ViLLMQAScenario",
        args={"dataset_name": dataset},
    )
    
    adapter_spec = get_generation_adapter_spec(
        instructions="Generate a response given a patient's questions and concerns.",
        input_noun="Patient",
        output_noun="Doctor",
        max_tokens=128,
    )

    return RunSpec(
        name=f"villm_question_answering:dataset={dataset},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_open_ended_generation_metric_specs() + get_generative_harms_metric_specs(),
        groups=["villm_question_answering", "question_answering"],
    )
    
@run_spec_function("villm_summarization")
def get_villm_summarization_spec(dataset: str = "vietnews", temperature: float = 0.3, device: str = "cpu") -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.villm_summarization_scenario."
        "ViLLMSumScenario",
        args={"dataset_name": dataset, "sampling_min_length": 50, "sampling_max_length": 150, "doc_max_length": 512},
    )

    adapter_spec = get_summarization_adapter_spec(
        num_sents=3,
        max_tokens=128,  # From Zhang et al. 2020 (https://arxiv.org/pdf/1912.08777.pdf)
        temperature=temperature,  # From Wu et al. 2021 (https://arxiv.org/pdf/2109.10862.pdf)
    )

    return RunSpec(
        name=f"villm_summarization:dataset={dataset},temperature={temperature},device={device}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=get_summarization_metric_specs({"task": "summarization_vietnews", "device": device})
        + get_generative_harms_metric_specs(),
        groups=["villm_summarization", "summarization"],
    )
    
