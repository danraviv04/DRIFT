"""Config-driven generation of AlphaFold 3 input JSONs for mutation ensembles.

Replaces the hand-edited ``generate_af3_ensembles.py``: a complex is described
once in ``af3_jobs/configs/<NAME>.json`` (chains, ligands, seeds, mutation
source) and the CLI turns that description into one AF3 job JSON per variant.

    python -m af3_jobs generate PIK3CA

See ``af3_jobs/README.md``.
"""

from af3_jobs.spec import JobSpec, ChainSpec, MutationSourceSpec, load_spec
from af3_jobs.build import build_jobs, write_jobs

__all__ = [
    "JobSpec",
    "ChainSpec",
    "MutationSourceSpec",
    "load_spec",
    "build_jobs",
    "write_jobs",
]
