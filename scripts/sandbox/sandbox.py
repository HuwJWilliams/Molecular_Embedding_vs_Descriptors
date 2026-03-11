import pandas as pd
import sys
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent / "path"))
from get_paths import getPaths

from transformers import AutoConfig

# config = AutoConfig.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
# print(config.max_position_embeddings)


# config = AutoConfig.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
# print(config.max_position_embeddings)

paths=getPaths()

print(paths["prediction_output_dirs"]["embedding_and_descriptor_cross_predictions"])