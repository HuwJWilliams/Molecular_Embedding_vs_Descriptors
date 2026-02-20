"""
Grouping functions for molecular descriptors (RDKit, Mordred, etc.)
Each function returns a dictionary of descriptor groups.
"""

def getRDKitGroups():
    """Return descriptor groups for RDKit descriptors."""
    return {
        # --- Size & Mass ---
        "size_mass": [
            "MolWt","HeavyAtomMolWt","ExactMolWt","NumValenceElectrons",
            "HeavyAtomCount","FractionCSP3","NumHeteroatoms"
        ],

        # --- Electronic (scalar descriptors only) ---
        "electronic_charges": [
            "MaxPartialCharge","MinPartialCharge",
            "MaxAbsPartialCharge","MinAbsPartialCharge"
        ],
        "electronic_estate_indices": [
            "MaxEStateIndex","MinEStateIndex",
            "MaxAbsEStateIndex","MinAbsEStateIndex"
        ],

        # --- All VSA descriptors grouped ---
        "vsa_peoe": [f"PEOE_VSA{i}" for i in range(1,15)],
        "vsa_estate": [f"EState_VSA{i}" for i in range(1,11)],
        "vsa_vsaestate": [f"VSA_EState{i}" for i in range(1,11)],
        "vsa_logp": [f"SlogP_VSA{i}" for i in range(1,13)],
        "vsa_mr": [f"SMR_VSA{i}" for i in range(1,11)],

        # --- Lipophilicity (non-VSA ones) ---
        "lipophilicity_basic": ["MolLogP","TPSA","MolMR"],

        # --- Topological indices ---
        "topological_chi": [
            "Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v",
            "Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v"
        ],
        "topological_shape": ["HallKierAlpha","Kappa1","Kappa2","Kappa3"],
        "topological_complexity": ["BalabanJ","BertzCT","Ipc","AvgIpc"],

        # --- BCUT descriptors ---
        "bcut": [
            "BCUT2D_MWHI","BCUT2D_MWLOW","BCUT2D_CHGHI","BCUT2D_CHGLO",
            "BCUT2D_LOGPHI","BCUT2D_LOGPLOW","BCUT2D_MRHI","BCUT2D_MRLOW"
        ],

        # --- Shape/Surface ---
        "shape_surface": ["LabuteASA","Phi"],

        # --- Rings (split aromatic vs non-aromatic) ---
        "rings_aromatic": [
            "NumAromaticCarbocycles","NumAromaticHeterocycles","NumAromaticRings"
        ],
        "rings_non_aromatic": [
            "RingCount","NumAliphaticCarbocycles","NumAliphaticHeterocycles","NumAliphaticRings",
            "NumAmideBonds","NumBridgeheadAtoms","NumSaturatedCarbocycles","NumSaturatedHeterocycles","NumSaturatedRings"
        ],

        # --- Hydrogen / Rotatable bonds ---
        "hydrogen_rotatable": [
            "NumHAcceptors","NumHDonors","NHOHCount","NOCount","NumRotatableBonds"
        ],

        # --- Stereo ---
        "stereo": ["NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters"],

        # --- Fingerprints ---
        "fingerprints": ["FpDensityMorgan1","FpDensityMorgan2","FpDensityMorgan3"],

        # --- Drug-likeness ---
        "druglikeness": ["qed"]
    }

# %%
def getMordredGroups():
    """Return descriptor groups for Mordred descriptors."""
    from mordred import Calculator, descriptors

    groups = {}

    calc = Calculator(descriptors, ignore_3D=True)
    ls = calc.descriptors

    for desc in ls:
        module = desc.__module__
        module = module.split(".")[-1]
        desc = str(desc).split(".")[-1]
    
        if module not in groups.keys():
            groups[module] = []
        groups.setdefault(module, []).append(desc)

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
