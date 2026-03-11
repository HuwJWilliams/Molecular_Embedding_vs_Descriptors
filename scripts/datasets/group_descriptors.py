"""
Grouping functions for molecular descriptors (RDKit, Mordred, etc.)
Each function returns a dictionary of descriptor groups.
"""

def getRDKitGroups(prefix="_rdkit"):
    """Return descriptor groups for RDKit descriptors."""

    def p(name):
        return f"{name}{prefix}"

    return {

        # --- Size & Mass ---
        "size_mass": [
            p("MolWt"), p("HeavyAtomMolWt"), p("ExactMolWt"),
            p("NumValenceElectrons"), p("HeavyAtomCount"),
            p("FractionCSP3"), p("NumHeteroatoms")
        ],

        # --- Electronic (scalar descriptors only) ---
        "electronic_charges": [
            p("MaxPartialCharge"), p("MinPartialCharge"),
            p("MaxAbsPartialCharge"), p("MinAbsPartialCharge")
        ],

        "electronic_estate_indices": [
            p("MaxEStateIndex"), p("MinEStateIndex"),
            p("MaxAbsEStateIndex"), p("MinAbsEStateIndex")
        ],

        # --- All VSA descriptors grouped ---
        "vsa_peoe": [p(f"PEOE_VSA{i}") for i in range(1, 15)],
        "vsa_estate": [p(f"EState_VSA{i}") for i in range(1, 11)],
        "vsa_vsaestate": [p(f"VSA_EState{i}") for i in range(1, 11)],
        "vsa_logp": [p(f"SlogP_VSA{i}") for i in range(1, 13)],
        "vsa_mr": [p(f"SMR_VSA{i}") for i in range(1, 11)],

        # --- Lipophilicity ---
        "lipophilicity_basic": [
            p("MolLogP"), p("TPSA"), p("MolMR")
        ],

        # --- Topological indices ---
        "topological_chi": [
            p("Chi0"), p("Chi0n"), p("Chi0v"),
            p("Chi1"), p("Chi1n"), p("Chi1v"),
            p("Chi2n"), p("Chi2v"),
            p("Chi3n"), p("Chi3v"),
            p("Chi4n"), p("Chi4v")
        ],

        "topological_shape": [
            p("HallKierAlpha"), p("Kappa1"),
            p("Kappa2"), p("Kappa3")
        ],

        "topological_complexity": [
            p("BalabanJ"), p("BertzCT"),
            p("Ipc"), p("AvgIpc")
        ],

        # --- BCUT descriptors ---
        "bcut": [
            p("BCUT2D_MWHI"), p("BCUT2D_MWLOW"),
            p("BCUT2D_CHGHI"), p("BCUT2D_CHGLO"),
            p("BCUT2D_LOGPHI"), p("BCUT2D_LOGPLOW"),
            p("BCUT2D_MRHI"), p("BCUT2D_MRLOW")
        ],

        # --- Shape/Surface ---
        "shape_surface": [
            p("LabuteASA"), p("Phi")
        ],

        # --- Rings ---
        "rings_aromatic": [
            p("NumAromaticCarbocycles"),
            p("NumAromaticHeterocycles"),
            p("NumAromaticRings")
        ],

        "rings_non_aromatic": [
            p("RingCount"),
            p("NumAliphaticCarbocycles"),
            p("NumAliphaticHeterocycles"),
            p("NumAliphaticRings"),
            p("NumAmideBonds"),
            p("NumBridgeheadAtoms"),
            p("NumSaturatedCarbocycles"),
            p("NumSaturatedHeterocycles"),
            p("NumSaturatedRings")
        ],

        # --- Hydrogen / Rotatable bonds ---
        "hydrogen_rotatable": [
            p("NumHAcceptors"), p("NumHDonors"),
            p("NHOHCount"), p("NOCount"),
            p("NumRotatableBonds")
        ],

        # --- Stereo ---
        "stereo": [
            p("NumAtomStereoCenters"),
            p("NumUnspecifiedAtomStereoCenters")
        ],

        # --- Fingerprints ---
        "fingerprints": [
            p("FpDensityMorgan1"),
            p("FpDensityMorgan2"),
            p("FpDensityMorgan3")
        ],

        # --- Drug-likeness ---
        "druglikeness": [
            p("qed")
        ]
    }

# %%
def getMordredGroups(prefix="_mordred"):
    """Return descriptor groups for Mordred descriptors."""

    from mordred import Calculator, descriptors

    groups = {}

    calc = Calculator(descriptors, ignore_3D=True)

    for desc in calc.descriptors:
        module = desc.__module__.split(".")[-1]
        name = str(desc).split(".")[-1]

        # apply prefix
        name = f"{name}{prefix}"

        groups.setdefault(module, []).append(name)

    return groups


def getGroups(source):
    """Return descriptor groups depending on source."""
    if source.lower() == "rdkit":
        return getRDKitGroups()
    elif source.lower() == "mordred":
        return getMordredGroups()
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'rdkit' or 'mordred'.")

getGroups(source="mordred")
# %%
