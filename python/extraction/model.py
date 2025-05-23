import argparse
import difflib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import dotenv
from langchain import hub
from langchain_community.callbacks.manager import get_openai_callback
from utils import Paper
from langchain.prompts import ChatPromptTemplate
from prompt import (
   SYSTEM_PROMPT_LIST_CELL_TYPES,
   FEW_SHOT_LIST_CELL_TYPES,
   SYSTEM_PROMPT_IS_CELL_TYPE,
   FEW_SHOT_IS_CELL_TYPE,
   SYSTEM_PROMPT_EXTRACT_CELL_TYPE_DETAILS,
   FEW_SHOT_EXTRACT_CELL_TYPE_DETAILS,
   SYSTEM_PROMPT_REMOVE_DUPLICATES,
   FEW_SHOT_REMOVE_DUPLICATES,
)

dotenv.load_dotenv()


def find_paragraph(
    cell_type: str, paper: Paper, additional_paragraphs: List[Tuple[str, int]] = None
) -> List[Tuple[str, int, str]]:
    """Find the paragraphs in the paper that contain the cell type.

    Args:
        cell_type: The cell type to search for.
        paper: The paper to search in.
        additional_paragraphs: Optional list of (section, index) tuples to include.

    Returns:
        A list of tuples, each containing the section type, the index of the paragraph, and the paragraph text.
    """

    def strict_search_fn(text: str):
        regex_pattern = r"(^|[^\w])" + re.escape(cell_type.lower()) + r"([^\w]|$)"
        return bool(re.search(regex_pattern, text.lower()))

    # Use a set to track unique (section, idx) pairs
    found_pairs = set()
    found_paragraphs = []

    for section_name, paragraphs in paper.sections.items():
        for idx, paragraph in enumerate(paragraphs):
            if strict_search_fn(paragraph):
                if (section_name, idx) not in found_pairs:
                    found_pairs.add((section_name, idx))
                    found_paragraphs.append((section_name, idx, paragraph))

    # Combine with additional paragraphs if provided
    if additional_paragraphs:
        for section, idx in additional_paragraphs:
            if section in paper.sections and idx < len(paper.sections[section]):
                if (section, idx) not in found_pairs:
                    found_pairs.add((section, idx))
                    paragraph = paper.sections[section][idx]
                    found_paragraphs.append((section, idx, paragraph))

    return found_paragraphs


@dataclass
class CellTypeInfo:
    """Stores information about a cell type found in a paper."""

    cell_type: str
    explanation: str = ""
    factoids: List[str] = field(default_factory=list)
    sources: Set[Tuple[str, int]] = field(
        default_factory=set
    )  # Set of (section_name, idx)
    pmid: str = ""
    is_valid: bool = False

    @property
    def evidence(self) -> List[str]:
        """Returns formatted evidence strings like 'RESULTS 26'."""
        return [f"{section} {idx}" for section, idx in sorted(self.sources)]

    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "cell_type": self.cell_type,
            "explanation": self.explanation,
            "factoids": self.factoids,
            "metadata": {"pmid": self.pmid, "evidence": self.evidence},
        }


class ExtractionModel:
    def __init__(self, model_type: str = "gpt-4o"):
        self.model_type = model_type
        self.logger = logging.getLogger("ExtractionModel")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

        self.load_prompts()

    # def load_prompts(self):
    #     prompt_names = ["list_all", "check", "extract", "deduplicate"]
    #     prompts = {}
    #     for prompt_name in prompt_names:
    #         prompts[prompt_name] = hub.pull(f"nfi_{prompt_name}")
    #     self.prompts = prompts

    def load_prompts(self):
        list_all_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_LIST_CELL_TYPES + "\n\n" + FEW_SHOT_LIST_CELL_TYPES),
            ("user", "{input}")
        ])
        
        check_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_IS_CELL_TYPE + "\n\n" + FEW_SHOT_IS_CELL_TYPE),
            ("user", "Related paragraphs: {related_paragraphs}\nCandidate cell type: {cell_type}")
        ])
        
        extract_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_EXTRACT_CELL_TYPE_DETAILS + "\n\n" + FEW_SHOT_EXTRACT_CELL_TYPE_DETAILS),
            ("user", "List all factoids about {cell_type} given the \"Related paragraphs\". For each factoid, start with \"{cell_type}\" as the subject.\nContext:\n{related_paragraphs}\nClosest cell types: {closest_cell_types}")
        ])
        
        deduplicate_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_REMOVE_DUPLICATES + "\n\n" + FEW_SHOT_REMOVE_DUPLICATES),
            ("user", "{cell_types}")
        ])
        
        self.prompts = {
            "list_all": list_all_template,
            "check": check_template,
            "extract": extract_template,
            "deduplicate": deduplicate_template
        }

    def get_model(self, json_mode: bool = False):
        """Get the LLM model based on the model type and response format.

        Args:
            json_mode (bool): If True, configures the model to return JSON formatted responses.

        Returns:
            Union[ChatOpenAI, ChatGoogleGenerativeAI]: Configured language model instance.

        Raises:
            ValueError: If model_type is not supported.
        """
        model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        default_config = {"temperature": 0, "frequency_penalty": 0.1}
        if self.model_type in ["gpt-4o", "gpt-4o-mini"]:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=self.model_type,
                model_kwargs=model_kwargs,
                **default_config,
            )
        elif self.model_type in ["o3-mini"]:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=self.model_type,
                model_kwargs=model_kwargs,
                reasoning_effort="low",
            )  # No default config for o3-mini
        elif self.model_type in ["gemini-1.5-flash", "gemini-1.5-pro"]:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=self.model_type,
                model_kwargs=model_kwargs,
                **default_config,
            )
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        return model

    def llm_call(
        self,
        task: str,
        single_input: Dict[str, Any],
    ) -> str:
        model = self.get_model()
        prompt = self.prompts[task]
        model = prompt | model
        return model.invoke(single_input).content

    def llm_call_batch(
        self,
        task: str,
        batch_inputs: List[Dict[str, Any]],
    ) -> List[str]:
        model = self.get_model()
        prompt = self.prompts[task]
        model = prompt | model
        batch_outputs = model.batch(batch_inputs)
        return [r.content for r in batch_outputs]

    def list_candidate_cell_types_batch(self, paper: Paper) -> List[CellTypeInfo]:
        """Extract candidate cell types from paper paragraphs using batch processing.

        Processes each paragraph of the paper to identify potential cell types, aiming for high recall.
        Tracks the source location (section and paragraph index) for each identified cell type.

        Args:
            paper (Paper): Paper object containing sections and paragraphs to analyze.

        Returns:
            List[CellTypeInfo]: List of CellTypeInfo objects for each unique candidate cell type,
                including source locations but without validation or detailed information.
        """
        # Build list of prompts for each paragraph
        batch_inputs = []
        paragraph_sources = []  # Track source info for each prompt
        for section_name, paragraphs in paper.sections.items():
            for idx, paragraph in enumerate(paragraphs):
                batch_inputs.append({"input": paragraph})
                paragraph_sources.append((section_name, idx))

        # Process all paragraphs in batch
        raw_responses = self.llm_call_batch(
            task="list_all",
            batch_inputs=batch_inputs,
        )

        # Track cell types and their sources
        cell_type_map: Dict[str, CellTypeInfo] = {}

        # Process responses
        for raw_response, (section_name, idx) in zip(raw_responses, paragraph_sources):
            cell_types = [ct.strip() for ct in raw_response.split("\n") if ct.strip()]

            # Update sources for each cell type
            for cell_type in cell_types:
                if cell_type == "None":  # Skip if no cell type is found.
                    continue
                if cell_type not in cell_type_map:
                    cell_type_map[cell_type] = CellTypeInfo(
                        cell_type=cell_type, pmid=paper.pmid
                    )
                cell_type_map[cell_type].sources.add((section_name, idx))

            self.logger.info(f"{section_name}: {cell_types}")

        # Log results
        cell_types = sorted(cell_type_map.keys())
        self.logger.info(
            f"Found {len(cell_types)} candidate cell types: {', '.join(cell_types)}."
        )

        # Log source information
        for info in cell_type_map.values():
            self.logger.info(
                f"Sources for {info.cell_type}: {', '.join(info.evidence)}"
            )

        return list(cell_type_map.values())

    def check_cell_type_batch(
        self,
        cell_type_infos: List[CellTypeInfo],
        paper: Paper,
    ) -> List[CellTypeInfo]:
        """Validate whether candidate cell types are actually neuronal cell types.

        Processes multiple cell types in batch, examining relevant paragraphs for each candidate
        to determine if it represents a valid neuronal cell type.

        Args:
            cell_type_infos (List[CellTypeInfo]): List of candidate cell types to validate.
            paper (Paper): Paper object containing the source text.

        Returns:
            List[CellTypeInfo]: Filtered list of CellTypeInfo objects that were validated as
                neuronal cell types, with updated is_valid and explanation fields.
        """
        # Build prompts for each cell type
        batched_inputs = []
        cell_type_infos_w_paragraphs = []
        for info in cell_type_infos:
            retrieved_paragraphs = find_paragraph(info.cell_type, paper, info.sources)
            if not retrieved_paragraphs:
                continue
            cell_type_infos_w_paragraphs.append(info)
            paragraphs_str = "\n".join(
                [paragraph for _, _, paragraph in retrieved_paragraphs]
            )
            batched_inputs.append(
                {
                    "related_paragraphs": paragraphs_str,
                    "cell_type": info.cell_type,
                }
            )
        if not batched_inputs:
            return []

        batched_outputs = self.llm_call_batch(
            task="check",
            batch_inputs=batched_inputs,
        )

        for info, raw_response in zip(cell_type_infos_w_paragraphs, batched_outputs):
            response_lines = raw_response.strip().split("\n")
            assert response_lines[0].lower().strip() in ["true", "false"]
            info.is_valid = response_lines[0].lower().strip() == "true"
            info.explanation = "\t".join(response_lines[1:])
            self.logger.info(
                f"{info.is_valid} ({response_lines[0].strip()}) {info.cell_type}: {info.explanation}"
            )

        return [info for info in cell_type_infos if info.is_valid]

    def extract_cell_type_details_batch(
        self, cell_type_infos: List[CellTypeInfo], paper: Paper
    ) -> List[CellTypeInfo]:
        """Extract detailed information about validated cell types in batch.

        For each valid cell type, analyzes relevant paragraphs to extract factoids and
        properties while avoiding confusion with similar cell types.

        Args:
            cell_type_infos (List[CellTypeInfo]): List of validated cell types to analyze.
            paper (Paper): Paper object containing the source text.

        Returns:
            List[CellTypeInfo]: Updated CellTypeInfo objects with extracted factoids and details.
        """
        batched_inputs = []

        for i, info in enumerate(cell_type_infos):
            retrieved_paragraphs = find_paragraph(info.cell_type, paper, info.sources)

            paragraphs_str = "\n".join(
                [paragraph for _, _, paragraph in retrieved_paragraphs]
            )

            # Get closest cell types
            cell_types_wo_cell_type = [
                ct.cell_type for ct in cell_type_infos if ct.cell_type != info.cell_type
            ]
            closest_cell_type = difflib.get_close_matches(
                info.cell_type, cell_types_wo_cell_type, n=5
            )
            closest_cell_type_str = ", ".join(closest_cell_type)

            batched_inputs.append(
                {
                    "cell_type": info.cell_type,
                    "related_paragraphs": paragraphs_str,
                    "closest_cell_types": closest_cell_type_str,
                }
            )

        if not batched_inputs:
            return []

        # Process all prompts in batch
        raw_responses = self.llm_call_batch(
            task="extract",
            batch_inputs=batched_inputs,
        )

        # Update cell type infos with factoids
        for cell_type_info, raw_response in zip(cell_type_infos, raw_responses):
            cell_type_info.factoids = raw_response.strip().split("\n")
            self.logger.info(
                f"Extracted details for {cell_type_info.cell_type}: {cell_type_info.to_dict()}"
            )

        return cell_type_infos

    def remove_duplicates(
        self, cell_type_infos: List[CellTypeInfo]
    ) -> List[CellTypeInfo]:
        """Remove semantically duplicate cell types from the list.

        Uses LLM to identify and remove cell types that refer to the same biological entity,
        even if they have slightly different names or representations.

        Args:
            cell_type_infos (List[CellTypeInfo]): List of cell types to deduplicate.

        Returns:
            List[CellTypeInfo]: Deduplicated list of CellTypeInfo objects.

        Raises:
            ValueError: If LLM response includes cell types not in the original list.
        """
        cell_types = [info.cell_type for info in cell_type_infos]
        cell_types = list(sorted(cell_types))
        user_prompt = "\n".join(cell_types)
        raw_response = self.llm_call(
            task="deduplicate",
            single_input={"cell_types": user_prompt},
        )
        unique_cell_types = raw_response.strip().split("\n")

        # Validate response
        valid_cell_types = []
        for cell_type in unique_cell_types:
            cell_type = cell_type.strip()
            if cell_type not in cell_types:
                self.logger.warning(f"Cell type {cell_type} not found in the list.")
                continue
            valid_cell_types.append(cell_type)

        # Return CellTypeInfo objects for unique cell types
        return [info for info in cell_type_infos if info.cell_type in valid_cell_types]

    def extract_cell_types(self, paper: Paper) -> List[dict]:
        """Extract and process neuronal cell types from a scientific paper.

        The extraction process follows these steps:
        1. Identify candidate cell types from the paper text
        2. Validate each candidate is a neuronal cell type
        3. Remove duplicate cell types
        4. Extract detailed information for valid cell types

        Args:
            paper: Paper object containing the text to analyze

        Returns:
            List of dictionaries containing details for each valid cell type
        """
        # Get initial list of candidate cell types
        candidates = self.list_candidate_cell_types_batch(paper)
        if not candidates:
            return []

        # Filter to only valid neuronal cell types
        valid_cell_types = self.check_cell_type_batch(candidates, paper)

        # Remove any duplicate cell types
        unique_cell_types = self.remove_duplicates(valid_cell_types)
        if not unique_cell_types:
            return []

        # Extract detailed information for each unique cell type
        detailed_cell_types = self.extract_cell_type_details_batch(
            unique_cell_types, paper
        )
        return [info.to_dict() for info in detailed_cell_types]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pmid",
        type=int,
        default=32619476,
        help="The PMID of the paper.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="The model to use for extraction.",
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    model = ExtractionModel(model_type=args.model)
    paper = Paper.from_pmid(args.pmid)

    with get_openai_callback() as cb:
        cell_types = model.extract_cell_types(paper)
    # Save the cell types to a JSON file.
    output_file = os.path.join(
        project_root,
        "assets",
        "cell_types",
        args.model.replace("-", "_"),
        f"{args.pmid}.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(cell_types, f, indent=4, ensure_ascii=False)
    print(cb)
