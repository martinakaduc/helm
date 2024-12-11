import os
from typing import Dict, List
import json

from helm.common.general import ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    PassageQuestionInput
    Input,
    Output,
)


class ViLLMQAScenario(Scenario):
    """
    ViLLM consists of 10 evalutation scenarios for Vietnamese language models.
    """

    name = "villm_question_answering"
    description = "Benchmark on question-answering for Vietnamese Language Models"
    tags = ["question_answering"]

    def __init__(
        self,
        dataset_name: str,
    ):
        """
        Initializes summarization scenario.
        Args:
            dataset_name: String identifier for dataset. Currently
                          supported options ["Xsum", "cnn-dm"].
            sampling_min_length: Int indicating minimum length for training
                                 documents. Training examples smaller than
                                 sampling_min_length will be filtered out.
                                 Useful for preventing the adapter from sampling
                                 really small documents.
            sampling_max_length: Int indicating maximum length for training
                                 documents. Training examples larger than
                                 sampling_max_length will be filtered out.
                                 Useful for preventing the adapter from
                                 sampling really large documents.
            doc_max_length: Int indicating the maximum length to truncate
                            documents. Documents in all splits will be
                            truncated to doc_max_length tokens.
                            NOTE: Currently uses whitespace tokenization.
        """
        super().__init__()
        if dataset_name not in ["mlqa", "xquad_xtreme"]:
            raise Exception(f"Unknown dataset_name: {dataset_name}")
        self.dataset_name = dataset_name
    
    def get_instances(self, output_path: str) -> List[Instance]:
        if self.dataset_name == "mlqa":
            dataset_repo = "facebook/mlqa"
            context_key = "context"
            query_key = "question"
            answer_key = "answers"
            subset = "mlqa.vi.vi"
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        elif self.dataset_name == "xquad_xtreme":
            dataset_repo = "juletxara/xquad_xtreme"
            context_key = "context"
            query_key = "question"
            answer_key = "answers"
            subset = "vi"
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        else:
            raise ValueError("The specified dataset is not supported")
        
        dataset = load_dataset(dataset_repo, subset=subset)
        instances: List[Instance] = []

        for split_name, split in splits.items():
            for example in dataset[split_name]:
                passage: str = example[context_key]
                question: str = example[query_key]
                answer: str = example[answer_key]
                    
                instances.append(
                    Instance(
                        input=PassageQuestionInput(passage=passage, question=question, separator=""),
                        references=[Reference(Output(text=answer), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
