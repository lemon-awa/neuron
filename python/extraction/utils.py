import dataclasses
import json
import os
from typing import Dict, List

import requests

BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"


def request_fulltext_raw(pmid: str) -> dict:
    """Request full text from PMC OA API or load from local file if available"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, "full_texts", f"{pmid}.json")

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    url = BASE_URL.format(pmid)
    response = requests.get(url)
    response.raise_for_status()
    content = response.json()[0]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(content, f)

    return content


def request_fulltext(pmid: str) -> dict:
    """Request full text from PMC OA API and process it by sections"""
    content = request_fulltext_raw(pmid)
    document = content["documents"]
    assert len(document) == 1
    document = document[0]
    passages = document["passages"]

    records = []
    for passage in passages:
        _type = passage["infons"]["type"]
        _section_type = passage["infons"]["section_type"]
        if _type == "paragraph" or _type == "abstract":
            text = passage["text"]
            records.append((_section_type, text))
        elif _section_type == "TITLE":
            text = passage["text"]
            records.append((_section_type, text))

    # Merge paragraphs with the same section_type into a list
    merged = {}
    for section_type, text in records:
        if section_type not in merged:
            merged[section_type] = []
        merged[section_type].append(text)
    return merged


def get_paper_texts(pmid):
    # Request full text.
    fulltext_dict = request_fulltext(pmid)
    title = fulltext_dict.get("TITLE", ["None"])[0]
    abstract = fulltext_dict.get("ABSTRACT", ["None"])[0]

    texts = []
    texts.append(f"[Title] {title}")
    texts.append(f"[Abstract] {abstract}")
    # Filter out paragraphs that contain cell type names.
    for i, p in enumerate(fulltext_dict.get("INTRO", [])):
        texts.append(f"[INTRO {i}]: {p}")
    for i, p in enumerate(fulltext_dict.get("RESULTS", [])):
        texts.append(f"[RESULTS {i}]: {p}")

    return "\n\n".join(texts)


def get_evidence(pmid, index):
    paper = Paper.from_pmid(pmid)
    sec, idx = index.split(" ")
    idx = int(idx)
    return paper.sections[sec][idx]


class CellTypeDetector:
    def __init__(self, fuzzy_cell_types_path: str = "fuzzy_cell_types.json"):
        with open(fuzzy_cell_types_path, "r") as f:
            self.fuzzy_cell_types = json.load(f)
            self.fuzzy_cell_types = {x.lower() for x in self.fuzzy_cell_types}

    def is_cell_type_in_text(self, text, verbose=False):
        for cell_type in self.fuzzy_cell_types:
            if cell_type in text.lower().split():
                if verbose:
                    print(cell_type)
                return True
        return False

    def __call__(self, text):
        return self.is_cell_type_in_text(text)


@dataclasses.dataclass
class Paper:
    """A class to store the details of a paper"""

    title: str
    abstract: str
    sections: Dict[str, List[str]]
    pmid: str

    def to_str(self) -> str:
        """Convert the paper to a string."""
        paper_str = f"Title: {self.title}\n"
        paper_str += f"Abstract: {self.abstract}\n"
        for section_name, paragraphs in self.sections.items():
            for idx, paragraph in enumerate(paragraphs):
                paper_str += f"{section_name} {idx+1}: {paragraph}\n"
        return paper_str

    @classmethod
    def from_pmid(cls, pmid: str) -> "Paper":
        """Get the paper from PubMed."""
        paper_dict = request_fulltext(pmid)
        title = paper_dict.get("TITLE", ["None"])[0]
        abstract = paper_dict.get("ABSTRACT", ["None"])[0]
        sections = {
            "INTRO": paper_dict.get("INTRO", []),
            "RESULTS": paper_dict.get("RESULTS", []),
        }
        return cls(title, abstract, sections, pmid)
