import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from metrics.fluency.xcomet import CometFluency
from metrics.similarity import SimilarityConfig, SimilarityMeasurement
from metrics.toxicity import ToxicityConfig, ToxicityMeasurement
from utils import RequiredColumns, read_dataframes

REFERENCE_COLUMN = "neutral_sentence_reference"
EVALUATION_PATH = Path(__file__).resolve()
DATASET_TEXTDETOXEVAL_PATH =  Path(EVALUATION_PATH.parent.parent, "dataset/dataset_textdetoxeval.csv")
DATASET_RUSSE2022_PATH =  Path(EVALUATION_PATH.parent.parent, "dataset/dataset_russe2022.csv")


def main() -> dict[Any, Any]:
    parser = argparse.ArgumentParser(
        description="Calculate text similarity between original and rewritten texts."
    )
    parser.add_argument(
        "--dataset_name",
        type=Literal["textdetox", "russe"],
        default="textdetox",
        help="name of the tested dataset",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing in similarity and toxicity models",
    )
    parser.add_argument(
        "--fluency_batch_size",
        type=int,
        default=512,
        help="Batch size for processing in fluency models",
    )
    parser.add_argument(
        "--efficient",
        type=bool,
        default=False,
        help="Use efficient similarity calculation",
    )

    args = parser.parse_args()

    # Read input files
    if args.dataset_name == "textdetox":
        dataset_path = DATASET_TEXTDETOXEVAL_PATH
    elif args.dataset_name == "russe":
        dataset_path = DATASET_RUSSE2022_PATH
    else:
        raise NotImplementedError(f"{args.dataset_name=}")
    submission_df_merged = read_dataframes(dataset_path)

    original_texts = submission_df_merged[RequiredColumns.TOXIC_SENTENCE].tolist()
    rewritten_texts = submission_df_merged[RequiredColumns.NEUTRAL_SENTENCE].tolist()
    reference_texts = submission_df_merged[REFERENCE_COLUMN].tolist()

    # Configure and run similarity measurement
    sim_config = SimilarityConfig(
        batch_size=args.batch_size,
        efficient_version=args.efficient,
        device=args.device,
    )
    similarity_measurer = SimilarityMeasurement(sim_config)
    sim_scores = similarity_measurer.evaluate_similarity(
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        reference_texts=reference_texts,
    )

    # Configure and run toxicity measurement
    tox_config = ToxicityConfig(
        batch_size=args.batch_size,
        device=args.device,
    )
    toxicity_measurer = ToxicityMeasurement(tox_config)
    tox_scores = toxicity_measurer.compare_toxicity(
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        reference_texts=reference_texts,
    )

    # Configure and run fluency measurement
    fluency_measurer = CometFluency()

    comet_input: list[dict[str, str]] = []
    for original_sent, rewritten_sent, reference_sent in zip(
        original_texts, rewritten_texts, reference_texts
    ):
        comet_input.append(
            {"src": original_sent, "mt": rewritten_sent, "ref": reference_sent}
        )

    fluency_scores = fluency_measurer.get_scores(
        input_data=comet_input, batch_size=args.fluency_batch_size
    )

    # Get Final Metric
    J = np.array(sim_scores) * np.array(tox_scores) * np.array(fluency_scores)
    submission_df_merged["J"] = J
    submission_df_merged["STA"] = tox_scores
    submission_df_merged["SIM"] = sim_scores
    submission_df_merged["XCOMET"] = fluency_scores
    results = submission_df_merged.groupby("lang").agg(
        {"STA": "mean", "SIM": "mean", "XCOMET": "mean", "J": "mean"}
    )
    print(results.reset_index().to_markdown())
    return results.reset_index().to_dict(orient="records")


if __name__ == "__main__":
    main()
