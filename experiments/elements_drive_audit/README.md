# experiments/elements_drive_audit/

Is everything that should be on `$DRIFT_ARCHIVE_ROOT` there, intact, and what it claims to be?
Two scripts, both read-only on the drive (macOS mounts it read-only anyway).

```
PY=python
$PY experiments/elements_drive_audit/validate_batch.py --zip muts_jsons/<batch>.zip --drive-dir <folder>
$PY experiments/elements_drive_audit/inventory.py
```

**`validate_batch.py`**: one af3_jobs batch against the inputs that generated it, reading
every archive once, end to end. Per archive: the gzip CRC over the whole stream; the job
JSON's name, chains (protein, DNA, ligand) and seeds equal to the input; one model per
seed x sample; **every** model CIF folding exactly the input chains (from the CA / C1'
atoms); `_SUCCESS` present. Then each variant against the batch's own folded WT: exactly
one difference, at the coded position. Plus md5 and top-model duplicates. Resumable:
results append to `results/<folder>_archives.jsonl`, and a rerun skips finished archives
(`--fresh` starts over). The drive reads at 20-50 MB/s, so an 850 MB MutS_alpha archive
takes 17-45 s; run at most one drive-heavy job at a time.

**`inventory.py`**: metadata only, every gene. Every archive the pipeline's
`structure_file_index` names is listed, opens, is gzip or plain tar, and is not under
half its gene's median size; every local pLDDT CSV maps to one of them.

## Results, 2026-10-12

| batch | archives | every check |
| --- | --- | --- |
| MSH6_GT (`MutS_alpha_MSH6_GT_Mutations.zip`) | 148 | 148/148, variant-vs-WT 147/147 |
| PIK3CA (`PIK3CA_Mutations.zip`) | 92 | 92/92, variant-vs-WT 91/91 |

Both: 125 seed models + 1 top model per archive, one run folder each, no duplicate
archives or models, no `has_clash` model. MSH6_GT median pTM 0.75, ipTM 0.76; PIK3CA
0.93, 0.91.

Inventory: PTEN 264, VHL 131, KRAS (4A) 57, MutS_alpha 152, PIK3CA 92, FOXA1 822 and
TP53 1,176 archives all present and openable, and every local pLDDT CSV has its archive.
Not problems:
- `PTEN_WT_seeds26-200` is the WT-null pool and `P53_WT_with_ZN_and_DNA_again` a second
  WT run; neither is a cohort artifact.
- `FOXA1_WT.tar.gz` is an uncompressed tar under a `.gz` name; `tarfile` "r:*" reads it.
- 12 TP53 archives carry a rerun timestamp (`P53_D57E_with_ZN_and_DNA_20251204_115232`).

One real gap, in the cache rather than on the drive: **POLE P356S** is complete on the
drive (126 CIFs in `POLE_folders/POLE_P356S`) and has its pLDDT CSV, but it is absent
from POLE's `structure_file_index` and so from the coordinate cache (the "POLE P356S has
no coordinates" note in `coordinate_sd/`). A POLE cache rebuild would pick it up.

Not yet on the drive, still running on the cluster: KRAS4B (980 jobs) and TP53 seeds
21-25. Copy them over on the lab Linux box, then run `validate_batch.py` on KRAS4B.
