from typing import List, Optional
from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class ViLLMSumScenario(Scenario):
    """
    ViLLM consists of 10 evalutation scenarios for Vietnamese language models.
    """

    name = "villm_summarization"
    description = "Benchmark on summarization for Vietnamese Language Models"
    tags = ["summarization"]

    def __init__(
        self,
        dataset_name: str,
        sampling_min_length: Optional[int] = None,
        sampling_max_length: Optional[int] = None,
        doc_max_length: Optional[int] = None,
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
        if dataset_name not in ["vietnews", "wiki_lingua"]:
            raise Exception(f"Unknown dataset_name: {dataset_name}")
        self.dataset_name = dataset_name
        self.sampling_min_length = sampling_min_length
        self.sampling_max_length = sampling_max_length
        self.doc_max_length = doc_max_length

    def _clean_and_truncate(self, text: str, max_length: Optional[int] = None) -> str:
        text = text.replace("\n", " ")
        return " ".join(text.split()[:max_length])
    
    def get_instances(self, output_path: str) -> List[Instance]:
        if self.dataset_name == "vietnews":
            dataset_repo = "Yuhthe/vietnews"
            article_key = "article"
            summary_key = "abstract"
            subset = None
            splits = {"train": TRAIN_SPLIT, "validation": "validation", "test": TEST_SPLIT}
        elif self.dataset_name == "wiki_lingua":
            dataset_repo = "GEM/wiki_lingua"
            article_key = "source"
            summary_key = "target"
            subset = "vi"
            splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}
        else:
            raise ValueError("The specified dataset is not supported")
        
        dataset = load_dataset(dataset_repo, subset=subset)
        instances: List[Instance] = []

        for split_name, split in splits.items():
            for example in dataset[split_name]:
                article: str = self._clean_and_truncate(example[article_key], self.doc_max_length)
                summary: str = self._clean_and_truncate(example[summary_key])

                if split_name == "train":
                    art_len = len(article.split())
                    if self.sampling_max_length and art_len > self.sampling_max_length:
                        continue
                    if self.sampling_min_length and art_len < self.sampling_min_length:
                        continue
                    
                instances.append(
                    Instance(
                        input=Input(text=article),
                        references=[Reference(Output(text=summary), tags=[CORRECT_TAG])],
                        split=split,
                    )
                )

        return instances
