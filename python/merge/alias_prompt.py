SYSTEM_PROMPT_EXTRACT_ALIAS = """You are an expert in analyzing scientific text about Drosophila cell types. Your task is to identify alias pairs for cell types based on the provided factoids and explanations.

An alias pair consists of:
1. The main cell type name
2. Its alternative name (alias)
3. The evidence sentence that shows this relationship

Rules for identifying aliases:
1. Look for phrases like "also named", "also called", "previously known as", "formerly called", etc.
2. The alias should be a complete cell type name, not just a description
3. Both names should refer to the same biological entity
4. The evidence should clearly show the relationship between the names

Output format:
{{
    "alias_pairs": [
        {{
            "main_name": "original cell type name",
            "alias": "alternative name",
            "evidence": "exact sentence showing the relationship"
        }}
    ]
}}

If no aliases are found, return an empty list.
"""

FEW_SHOT_EXTRACT_ALIAS = """
Example 1:
Input:
{{
    "cell_type": "MP1",
    "explanation": "MP1 refers to a specific type of dopaminergic neuron involved in memory processes in Drosophila, as described in the context of the paragraph.\tConfidence score: 9/10",
    "factoids": [
        "MP1 is a dopaminergic neuron.  ",
        "MP1 is also named PPL1-γ1pedc.  ",
        "MP1 innervates the γ1 module and the α/β peduncle.  ",
        "MP1 is crucial after conditioning to enable the consolidation of both aversive and appetitive long-term memory (LTM).  ",
        "MP1 is anatomically matched with the MVP2 neuron.  ",
        "MP1 neurons are required for appetitive LTM consolidation immediately after training.  ",
        "MP1 activity is self-regulated through an inhibitory feedback by MVP2 neurons.  ",
        "MP1 oscillatory activity is enhanced immediately after conditioning.  ",
        "MP1 increased signaling is terminated by MVP2 activation after about 30 minutes.  ",
        "MP1 activity is beneficial for LTM in a limited time window immediately after conditioning but becomes deleterious for LTM afterward.  ",
        "MP1 calcium oscillations increase in frequency when a nutritious sugar presentation is associated with an odorant presentation.  ",
        "MP1 neurons have not been exhaustively described in terms of inputs."
    ]
}}

Output:
{{
    "alias_pairs": [
        {{
            "main_name": "MP1",
            "alias": "PPL1-γ1pedc",
            "evidence": "MP1 is also named PPL1-γ1pedc."
        }}
    ]
}}

Example 2:
Input:
{{
    "cell_type": "08B AN",
    "explanation": "The term \"08B AN\" refers to a specific type of neuron involved in sex-specific feedback mechanisms in Drosophila, indicating it is a neuronal cell type. \tConfidence score: 9/10",
    "factoids": [
        "08B AN are important for providing feedback during male song production.  ",
        "08B AN may act as a corollary discharge to suppress the auditory response to self-generated song.  ",
        "08B AN in females transmit feedback about sexually dimorphic information back to the brain.  ",
        "08B AN have a pathway that either takes a different course or does not exist in the male nervous system."
    ]
}}

Output:
{{
    "alias_pairs": []
}}
"""
