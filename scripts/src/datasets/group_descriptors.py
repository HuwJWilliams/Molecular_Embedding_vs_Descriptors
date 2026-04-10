"""
Grouping functions for molecular descriptors (RDKit, Mordred, etc.)
Each function returns a dictionary of descriptor groups.
"""
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors

def getRDKitGroups(prefix="_rdkit"):
    """Return RDKit descriptor groups aligned to Mordred descriptor categories."""

    def p(name):
        return f"{name}{prefix}"
    
    def get_fragment_names():
        return[
            f"{name}{prefix}"
            for name, _ in Descriptors.descList
            if name.startswith("fr_")
        ]

    return {

        # --- Constitutional / Weight ---
        "Constitutional": [
            p("MolWt"), p("HeavyAtomMolWt"), p("ExactMolWt"),
            p("NumValenceElectrons"), p("HeavyAtomCount"),
            p("FractionCSP3"), p("NumHeteroatoms")
        ],

        "Weight": [
            p("MolWt"), p("ExactMolWt"), p("HeavyAtomMolWt")
        ],

        # --- Electronic / EState ---
        "EState": [
            p("MaxEStateIndex"), p("MinEStateIndex"),
            p("MaxAbsEStateIndex"), p("MinAbsEStateIndex"),
            *[p(f"EState_VSA{i}") for i in range(1, 11)],
            *[p(f"VSA_EState{i}") for i in range(1, 11)],
        ],

        "TopologicalCharge": [
            p("MaxPartialCharge"), p("MinPartialCharge"),
            p("MaxAbsPartialCharge"), p("MinAbsPartialCharge")
        ],

        # --- VSA / MOE-type ---
        "MoeType": [
            *[p(f"PEOE_VSA{i}") for i in range(1, 15)],
            *[p(f"SlogP_VSA{i}") for i in range(1, 13)],
            *[p(f"SMR_VSA{i}") for i in range(1, 11)],
        ],

        # --- Physicochemical ---
        "SLogP": [
            p("MolLogP"),
            *[p(f"SlogP_VSA{i}") for i in range(1, 13)]
        ],

        "TopoPSA": [
            p("TPSA")
        ],

        "Polarizability": [
            p("MolMR"),
            *[p(f"SMR_VSA{i}") for i in range(1, 11)]
        ],

        # --- Topological indices ---
        "Chi": [
            p("Chi0"), p("Chi0n"), p("Chi0v"),
            p("Chi1"), p("Chi1n"), p("Chi1v"),
            p("Chi2n"), p("Chi2v"),
            p("Chi3n"), p("Chi3v"),
            p("Chi4n"), p("Chi4v")
        ],

        "KappaShapeIndex": [
            p("HallKierAlpha"), p("Kappa1"),
            p("Kappa2"), p("Kappa3")
        ],

        "BalabanJ": [
            p("BalabanJ")
        ],

        "BertzCT": [
            p("BertzCT")
        ],

        "InformationContent": [
            p("Ipc"), p("AvgIpc")
        ],

        # --- BCUT ---
        "BCUT": [
            p("BCUT2D_MWHI"), p("BCUT2D_MWLOW"),
            p("BCUT2D_CHGHI"), p("BCUT2D_CHGLO"),
            p("BCUT2D_LOGPHI"), p("BCUT2D_LOGPLOW"),
            p("BCUT2D_MRHI"), p("BCUT2D_MRLOW")
        ],

        # --- Surface / geometry ---
        "CPSA": [
            p("LabuteASA")  # closest proxy
        ],

        "Geometrical": [
            p("Phi")
        ],

        # --- Rings ---
        "RingCount": [
            p("RingCount"),
            p("NumAromaticCarbocycles"),
            p("NumAromaticHeterocycles"),
            p("NumAromaticRings"),
            p("NumAliphaticCarbocycles"),
            p("NumAliphaticHeterocycles"),
            p("NumAliphaticRings"),
            p("NumSaturatedCarbocycles"),
            p("NumSaturatedHeterocycles"),
            p("NumSaturatedRings"),
            p("NumBridgeheadAtoms")
        ],

        "Aromatic": [
            p("NumAromaticCarbocycles"),
            p("NumAromaticHeterocycles"),
            p("NumAromaticRings")
        ],

        # --- Bonds / flexibility ---
        "RotatableBond": [
            p("NumRotatableBonds")
        ],

        "HydrogenBond": [
            p("NumHAcceptors"), p("NumHDonors"),
            p("NHOHCount"), p("NOCount")
        ],

        "BondCount": [
            p("NumAmideBonds")
        ],

        # --- Stereo ---
        "Stereochemistry": [
            p("NumAtomStereoCenters"),
            p("NumUnspecifiedAtomStereoCenters")
        ],

        # --- Drug-likeness ---
        "Lipinski": [
            p("NumHAcceptors"), p("NumHDonors"),
            p("MolLogP"), p("MolWt")
        ],

        "Druglikeness": [
            p("qed")
        ],

        # --- Fragment-based ---
        "FragmentComplexity":
            get_fragment_names(),

        # --- Morgan fingerprint density (non-Mordred but useful) ---
        "FingerprintDensity": [
            p("FpDensityMorgan1"),
            p("FpDensityMorgan2"),
            p("FpDensityMorgan3")
        ]
    }

# %%
def getMordredGroups(prefix="_mordred"):
    """Return descriptor groups for Mordred descriptors."""

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

# %%
