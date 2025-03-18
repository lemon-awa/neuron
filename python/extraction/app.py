import streamlit as st
import json
import os
import argparse
from pathlib import Path

from utils import Paper, get_evidence

# Add default cell types directory path
DEFAULT_CELL_TYPES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cell_types"
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cell Type Navigation App")
    parser.add_argument(
        "--cell-types-dir",
        type=str,
        default=DEFAULT_CELL_TYPES_DIR,
        help="Directory containing cell type JSON files",
    )
    return parser.parse_args()


def load_cell_types(pmid, cell_types_dir):
    """Load cell type data from JSON file based on PMID"""
    try:
        file_path = Path(cell_types_dir) / f"{pmid}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def main():
    args = parse_args()
    st.set_page_config(layout="wide")

    with st.sidebar:
        st.title("Cell Type Navigation")
        # Get list of PMIDs from cell_types directory
        cell_types_dir = Path(args.cell_types_dir)
        pmid_files = [f.stem for f in cell_types_dir.glob("*.json")]
        pmid_files = [
            pmid
            for pmid in pmid_files
            if os.path.getsize(cell_types_dir / f"{pmid}.json") > 4
        ]
        pmid = st.selectbox("Select PMID:", options=pmid_files)

    st.title("Cell Type Extraction Results")

    if pmid:
        data = load_cell_types(pmid, cell_types_dir)
        paper = Paper.from_pmid(pmid)

        if data is None:
            st.error(f"No extraction results found for PMID: {pmid}")
        else:
            st.success(f"Found {len(data)} cell types")

            with st.sidebar:
                st.subheader("Cell Types")
                selected_cell = st.radio(
                    "Select cell type:",
                    options=[cell_info["cell_type"] for cell_info in data],
                    key="cell_type_radio",
                )

            for cell_info in data:
                if cell_info["cell_type"] == selected_cell:
                    st.header(cell_info["cell_type"])

                    st.subheader("Factoids")
                    for factoid in cell_info["factoids"]:
                        factoid_text = factoid.replace(
                            selected_cell, f":blue[**{selected_cell}**]"
                        )
                        st.markdown(f"* {factoid_text}")
                    st.subheader("Evidence")
                    for evidence in cell_info["metadata"]["evidence"]:
                        evidence_text = get_evidence(pmid, evidence)
                        evidence_text = evidence_text.replace(
                            selected_cell, f":blue[**{selected_cell}**]"
                        )
                        st.markdown(f"**{paper.title}**")
                        st.markdown(f"> {evidence}: {evidence_text}")

                    st.subheader("Explanation")
                    st.markdown(f"*{cell_info['explanation']}*")

                    st.divider()


if __name__ == "__main__":
    main()
