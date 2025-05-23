SYSTEM_PROMPT_EXTRACT_FEATURES = """You are a neuroscience expert assistant. Your task is to analyze neuron descriptions and determine their Flow and Superclass categories, and Class based on the given definitions.

Flow Categories:
- Afferent: neurons that locate in the periphery and project to the central nervous system
- Intrinsic: neurons located inside the brain and project inside the brain
- Efferent: neurons that locate in the central nervous system and project outside of the brain

Superclass Categories:
- Sensory: neurons that detect sensory stimulus
- Ascending: neurons with cell bodies that locate in the nerve cord and ascend to the brain
- Central: neurons with cell bodies that locate in the brain and project elsewhere in the brain
- Optic/Visual projection/Visual centrifugal: visual neurons (can be categorized more specifically if clear from description)
- Descending: neurons with cell bodies that locate in the brain and descend to the nerve cord
- Motor: neurons with cell bodies that locate in the brain or ventral nerve cord, projecting to peripheral muscles
- Endocrine: neurons that release hormones, usually with cell bodies inside brain/nerve cord, projecting outside CNS

Classes include: bilateral, ocellar, ALIN, ALPN, ALON, ALLN, CX, DAN, Kenyon Cell, MBIN, LHCENT, LHLN, ME, ME-LO

You will receive input with cell_type name and descriptions list. Consider ALL provided descriptions together to make your classification. If you're not confident about any category, use "None"."""

FEW_SHOT_EXTRACT_FEATURES = """
Example 1:
Input: {{
    "cell_type": "Am1",
    "descriptions": [
        "Am1 is described as a wide-field amacrine cell type that extends over multiple regions, indicating it is a specific neuronal cell type.",
        "Am1 is a wide-field amacrine cell that extends over the medulla, lobula, and lobula plate.",
        "Am1 has no other amacrine types like it with such an extended reach.",
        "Am1's classification as truly amacrine is somewhat unclear.",
        "Am1 has the medulla as its definite output neuropil by presynapse/postsynapse ratio.",
        "Am1 has no obvious axon defined by morphological criteria.",
        "Am1 is mentioned as a specific neuron involved in a visual circuit, receiving inputs and participating in inhibitory interactions with other neurons.",
        "Am1 receives major inputs from T4b and T5b, which are driven by ipsilateral back-to-front (BTF) motion.",
        "Am1 inhibits VCH and DCH but receives recurrent inhibition from VCH and DCH.",
        "Am1 activity is suppressed when provided with contralateral BTF input only."
    ]
}}
Output: {{
    "flow": "intrinsic",
    "superclass": "optic",
    "class": "None"
}}

Example 2:
Input: {{
    "cell_type": "C3",
    "descriptions": [
        "C3 is mentioned as a specific type of columnar cell providing GABAergic input to T4 dendrites.",
        "C3 are GABAergic cell-types in the medulla.",
        "C3 provide input to the proximal base of T4 dendrites.",
        "C3 refers to a specific type of feedback neuron in the EMD pathway, which is involved in lateral motion-dependent behaviors in Drosophila.",
        "C3 are part of the elementary motion detection (EMD) neural pathway.",
        "C3 are feedback neurons in the EMD pathway.",
        "C3 are required for lateral motion-dependent behaviors in Drosophila.",
        "C3 refers to a specific type of ascending neuron connecting the medulla and lamina.",
        "C3 are ascending neurons connecting the medulla and lamina."
    ]
}}
Output: {{
    "flow": "intrinsic",
    "superclass": "optic",
    "class": "None"
}}""" 