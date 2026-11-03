import json
import zipfile
import copy
import os
import pandas as pd
import re

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_AA_TO_CODONS: dict[str, list[str]] = {}
for _codon, _aa in CODON_TABLE.items():
    if _aa != "*":
        _AA_TO_CODONS.setdefault(_aa, []).append(_codon)


def min_nt_distance(aa1: str, aa2: str) -> int:
    """Minimum single-nucleotide substitutions to go from any codon of aa1 to any codon of aa2."""
    c1s = _AA_TO_CODONS.get(aa1, [])
    c2s = _AA_TO_CODONS.get(aa2, [])
    if not c1s or not c2s:
        return 3
    return min(sum(a != b for a, b in zip(c1, c2)) for c1 in c1s for c2 in c2s)


THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}

# The 10 verified MSH2 mutations
# mutations_done = {
#     "V3L": ("V", 3, "L"),
#     "R106K": ("R", 106, "K"),
#     "E198G": ("E", 198, "G"),
#     "A272V": ("A", 272, "V"),
#     "Q629R": ("Q", 629, "R"),
#     "V161D": ("V", 161, "D"),
#     "C199R": ("C", 199, "R"),
#     "G338E": ("G", 338, "E"),
#     "M453K": ("M", 453, "K"),
#     "C697F": ("C", 697, "F"),
# }

# KRAS variants ALREADY generated in the 4A isoform (the 43 in $DRIFT_ARCHIVE_ROOT/
# KRAS_archive, WT excluded). The loop below skips these, so a KRAS run only fills
# the 4A gap rather than re-running structures we already have.
# --- KRAS run (kept for reference) ---
# mutations_done = {
#     'A146P', 'A146V', 'A18V', 'A59S', 'A59T', 'F156I', 'F156S', 'F156V', 'G12A', 'G12C',
#     'G12D', 'G12R', 'G12S', 'G12V', 'G13C', 'G13D', 'G13R', 'G13V', 'G60R', 'G60S',
#     'G60V', 'I36M', 'K117N', 'K147E', 'K147R', 'K5E', 'K5N', 'L19F', 'M72L', 'N116H',
#     'P34L', 'P34R', 'Q22E', 'Q22K', 'Q22R', 'Q61H', 'Q61K', 'Q61L', 'Q61R', 'R68S',
#     'T58I', 'V14I', 'Y71H',
# }

# PTEN targeted re-run: nothing is 'already done' for the 14 forced variants.
mutations_done: set[str] = set()

# MUT_NAME = "KRAS"
MUT_NAME = "PTEN"

# Suffix for the output zip only (not for the job names inside it).
ZIP_TAG = "_regen14"

# Gene whose protein sequence is being mutated. Keep this separate from MUT_NAME:
# MUT_NAME may describe a complex (for example, "MutS_alpha"), while ClinVar rows
# must be matched against the actual gene annotation (for example, "MSH2").
# TARGET_GENE: str | None = "KRAS"
TARGET_GENE: str | None = "PTEN"

# Mutation source mode:
#   None → load mutations from MUTS_DF (ClinVar CSV)
#   1    → enumerate every substitution reachable in ≤1 nt change from the sequence
#   2    → enumerate every substitution reachable in ≤2 nt changes
#   3    → enumerate all possible single-AA substitutions
# When set, MUTS_DF is ignored entirely.
MAX_NT_SUBS: int | None = None

# Mutations to always include regardless of MAX_NT_SUBS or ClinVar source.
# Use standard 1-letter notation, e.g. ["G12C", "G12D"].

# The 13 labeled KRAS variants that exist only in the 4B batch and have no 4A
# structure. Regenerating them in 4A completes an isoform-consistent KRAS set
# (WT + all variants on 4A, matching AlphaMissense). All 13 pass the WT-residue
# check against the 4A sequence. Setting this makes MUTS_DF None (below), so no
# PTEN ClinVar rows leak in.
# The 14 PTEN variants whose archives on $DRIFT_ARCHIVE_ROOT/PTEN were misnamed:
# each contained a different variant's run (verified at the sequence level, both
# positions checked), so these 14 were never actually predicted. Their previous
# pLDDT CSVs duplicated other variants already in the dataset. All 14 pass the
# WT-residue check against the 403-aa PTEN sequence below.
FORCE_ADD_MUTATIONS: list[str] = [
    'R173P', 'R47G', 'S170I', 'S170R', 'T131I', 'T202I', 'T26I',
    'T26P', 'T277A', 'T277I', 'V175M', 'W111C', 'W111R', 'W274L',
]

MUTS_DF: pd.DataFrame | None = (
    pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "Ground_Truths/ClinVar/clinvar_result_kras.csv")
    )
    if MAX_NT_SUBS is None and not FORCE_ADD_MUTATIONS
    else None
)

def parse_mutation_name(mut_name: str):
    """
    Parse mutation names in either 1-letter form (e.g., V3L)
    or 3-letter form (e.g., Val3Leu).
    Returns: (orig_aa_one_letter, position_int, new_aa_one_letter)
    """
    s = str(mut_name).strip()

    # 1-letter amino-acid notation
    m1 = re.search(r"([A-Z])(\d+)([A-Z])", s)
    if m1:
        orig_aa, pos, new_aa = m1.groups()
        return orig_aa, int(pos), new_aa

    # 3-letter amino-acid notation
    m3 = re.search(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", s)
    if m3:
        orig3, pos, new3 = m3.groups()
        if orig3 not in THREE_TO_ONE or new3 not in THREE_TO_ONE:
            raise ValueError(f"Unknown amino acid code in mutation: {s}")
        return THREE_TO_ONE[orig3], int(pos), THREE_TO_ONE[new3]

    raise ValueError(f"Unsupported mutation format: {s}")


def _first_row_text(row: pd.Series, columns: tuple[str, ...]) -> str | None:
    """Return the first populated text value from alternative CSV column names."""
    for column in columns:
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column]).strip()
        if value:
            return value
    return None


def clinvar_row_matches_gene(row: pd.Series, target_gene: str | None) -> bool:
    """Whether a ClinVar row is annotated for the gene whose sequence we mutate.

    The transcript-level annotation in ``Variation``/``Name`` is authoritative.
    This prevents broad locus fields such as ``KLLN|PTEN`` from admitting a KLLN
    variant when the target sequence is PTEN. Older exports without a transcript-
    level gene annotation fall back to exact tokens in ``Genes``/``Gene(s)``.
    """
    if target_gene is None:
        return True

    target = target_gene.strip().upper()
    variation = _first_row_text(row, ("Variation", "Name"))
    if variation is not None:
        annotated_gene = re.search(r"\(([^()]+)\):", variation)
        if annotated_gene is not None:
            return annotated_gene.group(1).strip().upper() == target

    genes = _first_row_text(row, ("Genes", "Gene(s)"))
    if genes is None:
        return False
    gene_tokens = {
        gene.strip().upper() for gene in re.split(r"[|,;]", genes) if gene.strip()
    }
    # Without a transcript-level annotation, multi-gene locus rows are ambiguous.
    return gene_tokens == {target}


def clinvar_mutation_text(row: pd.Series) -> str | None:
    """Select the most gene-specific mutation annotation available in a row."""
    return _first_row_text(row, ("Variation", "Name", "Protein change"))


# Your base AF3 JSON structure
# base_json = {
#     "name": "MutS_alpha_with_Mg_and_ATP",
#     "modelSeeds": [
#         1,
#         2,
#         3,
#         4,
#         5,
#         6,
#         7,
#         8,
#         9,
#         10,
#         11,
#         12,
#         13,
#         14,
#         15,
#         16,
#         17,
#         18,
#         19,
#         20,
#         21,
#         22,
#         23,
#         24,
#         25,
#     ],
#     "sequences": [
#         {
#             "protein": {
#                 "id": "A",
#                 "sequence": "MAVQPKETLQLESAAEVGFVRFFQGMPEKPTTTVRLFDRGDFYTAHGEDALLAAREVFKTQGVIKYMGPAGAKNLQSVVLSKMNFESFVKDLLLVRQYRVEVYKNRAGNKASKENDWYLAYKASPGNLSQFEDILFGNNDMSASIGVVGVKMSAVDGQRQVGVGYVDSIQRKLGLCEFPDNDQFSNLEALLIQIGPKECVLPGGETAGDMGKLRQIIQRGGILITERKKADFSTKDIYQDLNRLLKGKKGEQMNSAVLPEMENQVAVSSLSAVIKFLELLSDDSNFGQFELTTFDFSQYMKLDIAAVRALNLFQGSVEDTTGSQSLAALLNKCKTPQGQRLVNQWIKQPLMDKNRIEERLNLVEAFVEDAELRQTLQEDLLRRFPDLNRLAKKFQRQAANLQDCYRLYQGINQLPNVIQALEKHEGKHQKLLLAVFVTPLTDLRSDFSKFQEMIETTLDMDQVENHEFLVKPSFDPNLSELREIMNDLEKKMQSTLISAARDLGLDPGKQIKLDSSAQFGYYFRVTCKEEKVLRNNKNFSTVDIQKNGVKFTNSKLTSLNEEYTKNKTEYEEAQDAIVKEIVNISSGYVEPMQTLNDVLAQLDAVVSFAHVSNGAPVPYVRPAILEKGQGRIILKASRHACVEVQDEIAFIPNDVYFEKDKQMFHIITGPNMGGKSTYIRQTGVIVLMAQIGCFVPCESAEVSIVDCILARVGAGDSQLKGVSTFMAEMLETASILRSATKDSLIIIDELGRGTSTYDGFGLAWAISEYIATKIGAFCMFATHFHELTALANQIPTVNNLHVTALTTEETLTMLYQVKKGVCDQSFGIHVAELANFPKHVIECAKQKALELEEFQYIGESQGYDIMEPAAKKCYLEREQGEKIIQEFLSKVKQMPFTEMSEENITIKLKQLKAEVIAKNNSFVNEIISRIKVTT",
#             }
#         },
#         {
#             "protein": {
#                 "id": "B",
#                 "sequence": "MSRQSTLYSFFPKSPALSDANKASARASREGGRAAAAPGASPSPGGDAAWSEAGPGPRPLARSASPPKAKNLNGGLRRSVAPAAPTSCDFSPGDLVWAKMEGYPWWPCLVYNHPFDGTFIREKGKSVRVHVQFFDDSPTRGWVSKRLLKPYTGSKSKEAQKGGHFYSAKPEILRAMQRADEALNKDKIKRLELAVCDEPSEPEEEEEMEVGTTYVTDKSEEDNEIESEEEVQPKTQGSRRSSRQIKKRRVISDSESDIGGSDVEFKPDTKEEGSSDEISSGVGDSESEGLNSPVKVARKRKRMVTGNGSLKRKSSRKETPSATKQATSISSETKNTLRAFSAPQNSESQAHVSGGGDDSSRPTVWYHETLEWLKEEKRRDEHRRRPDHPDFDASTLYVPEDFLNSCTPGMRKWWQIKSQNFDLVICYKVGKFYELYHMDALIGVSELGLVFMKGNWAHSGFPEIAFGRYSDSLVQKGYKVARVEQTETPEMMEARCRKMAHISKYDRVVRREICRIITKGTQTYSVLEGDPSENYSKYLLSLKEKEEDSSGHTRAYGVCFVDTSLGKFFIGQFSDDRHCSRFRTLVAHYPPVQVLFEKGNLSKETKTILKSSLSCSLQEGLIPGSQFWDASKTLRTLLEEEYFREKLSDGIGVMLPQVLKGMTSESDSIGLTPGEKSELALSALGGCVFYLKKCLIDQELLSMANFEEYIPLDSDTVSTTRSGAIFTKAYQRMVLDAVTLNNLEIFLNGTNGSTEGTLLERVDTCHTPFGKRLLKQWLCAPLCNHYAINDRLDAIEDLMVVPDKISEVVELLKKLPDLERLLSKIHNVGSPLKSQNHPDSRAIMYEETTYSKKKIIDFLSALEGFKVMCKIIGIMEEVADGFKSKILKQVISLQTKNPEGRFPDLTVELNRWDTAFDHEKARKTGLITPKAGFDSDYDQALADIRENEQSLLEYLEKQRNRIGCRTIVYWGIGRNRYQLEIPENFTTRNLPEEYELKSTKKGCKRYWTKTIEKKLANLINAEERRDVSLKDCMRRLFYNFDKNYKDWQSAVECIAVLDVLLCLANYSRGGDGPMCRPVILLPEDTPPFLELKGSRHPCITKTFFGDDFIPNDILIGCEEEEQENGKAYCVLVTGPNMGGKSTLMRQAGLLAVMAQMGCYVPAEVCRLTPIDRVFTRLGASDRIMSGESTFFVELSETASILMHATAHSLVLVDELGRGTATFDGTAIANAVVKELAETIKCRTLFSTHYHSLVEDYSQNVAVRLGHMACMVENECEDPSQETITFLYKFIKGACPKSYGFNAARLANLPEEVIQKGHRKAREFEKMNQSLRLFREVCLASERSTVDAEAVHKLLTLIKEL",
#             }
#         },
#         {"dna": {"id": "C", "sequence": "GGATCGTTTTTTTTTGTCGAC"}},
#         {"dna": {"id": "D", "sequence": "GTCGACAAAAAAAACGATCC"}},
#         {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
#         {"ligand": {"id": "F", "ccdCodes": ["MG"]}},
#         {"ligand": {"id": "G", "ccdCodes": ["ATP"]}},
#         {"ligand": {"id": "H", "ccdCodes": ["MG"]}},
#     ],
#     "dialect": "alphafold3",
#     "version": 2,
# }

# --- KRAS template (kept for reference) ---
# base_json = {
#     "name": "KRAS_WT",
#     "modelSeeds": [
#         1, 2, 3, 4, 5,
#         6, 7, 8, 9, 10,
#         11, 12, 13, 14, 15,
#         16, 17, 18, 19, 20,
#         21, 22, 23, 24, 25,
#     ],
#     "sequences": [
#         {
#             "protein": {
#                 "id": "A",
#                 # KRAS isoform 4A (UniProt P01116-1 / ENST00000256078), 189 aa.
#                 # Matches AlphaMissense's reference and the existing 4A archive so
#                 # newly generated structures are isoform-consistent with it.
#                 "sequence": (
#                     "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET"
#                     "CLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQI"
#                     "KRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQ"
#                     "RVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM"
#                 ),
#             }
#         },
#         {
#             "ligand": {
#                 "id": "B",
#                 "ccdCodes": ["GDP"],
#             }
#         },
#         {
#             "ligand": {
#                 "id": "C",
#                 "ccdCodes": ["MG"],
#             }
#         },
#     ],
#     "dialect": "alphafold3",
#     "version": 4,
# }

# --- PTEN template ---
# Reproduced from the job spec of an existing correct run
# ($DRIFT_ARCHIVE_ROOT/PTEN/PTEN_A121E_data.tar.gz -> PTEN_A121E_data.json) so the
# 14 regenerated variants are modelled under identical conditions to the other
# 236: same 403-aa sequence, same PO4 ligand, same 25 model seeds (25 x 5 = 125
# structures). Changing any of these would make the new ensembles
# non-comparable with the existing ones.
base_json = {
    "name": "PTEN_WT",
    "modelSeeds": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
    ],
    "sequences": [
        {
            "protein": {
                "id": "A",
                # PTEN, UniProt P60484 / ENST00000371953.8, 403 aa.
                "sequence": (
                    "MTAIIKEIVSRNKRRYQEDGFDLDLTYIYPNIIAMGFPAERLEGVYRNNI"
                    "DDVVRFLDSKHKNHYKIYNLCAERHYDTAKFNCRVAQYPFEDHNPPQLEL"
                    "IKPFCEDLDQWLSEDDNHVAAIHCKAGKGRTGVMICAYLLHRGKFLKAQE"
                    "ALDFYGEVRTRDKKGVTIPSQRRYVYYYSYLLKNHLDYRPVALLFHKMMF"
                    "ETIPMFSGGTCNPQFVVCQLKVKIYSSNSGPTRREDKFMYFEFPQPLPVC"
                    "GDIKVEFFHKQNKMLKKDKMFHFWVNTFFIPGPEETSEKVENGSLCDQEI"
                    "DSICSIERADNDKEYLVLTLTKNDLDKANKDKANRYFSPNFKVKLYFTKT"
                    "VEEPSNPEASSSTSVTPDVSDNEPDHYRYSDTTDSDPENEPFDEDQHTQI"
                    "TKV"
                ),
            }
        },
        {
            "ligand": {
                "id": "B",
                "ccdCodes": ["PO4"],
            }
        },
    ],
    "dialect": "alphafold3",
    "version": 4,
}

"""    "sequences": [
    {
        "protein": {
            "id": "A",
            "sequence": "MAVQPKETLQLESAAEVGFVRFFQGMPEKPTTTVRLFDRGDFYTAHGEDALLAAREVFKTQGVIKYMGPAGAKNLQSVVLSKMNFESFVKDLLLVRQYRVEVYKNRAGNKASKENDWYLAYKASPGNLSQFEDILFGNNDMSASIGVVGVKMSAVDGQRQVGVGYVDSIQRKLGLCEFPDNDQFSNLEALLIQIGPKECVLPGGETAGDMGKLRQIIQRGGILITERKKADFSTKDIYQDLNRLLKGKKGEQMNSAVLPEMENQVAVSSLSAVIKFLELLSDDSNFGQFELTTFDFSQYMKLDIAAVRALNLFQGSVEDTTGSQSLAALLNKCKTPQGQRLVNQWIKQPLMDKNRIEERLNLVEAFVEDAELRQTLQEDLLRRFPDLNRLAKKFQRQAANLQDCYRLYQGINQLPNVIQALEKHEGKHQKLLLAVFVTPLTDLRSDFSKFQEMIETTLDMDQVENHEFLVKPSFDPNLSELREIMNDLEKKMQSTLISAARDLGLDPGKQIKLDSSAQFGYYFRVTCKEEKVLRNNKNFSTVDIQKNGVKFTNSKLTSLNEEYTKNKTEYEEAQDAIVKEIVNISSGYVEPMQTLNDVLAQLDAVVSFAHVSNGAPVPYVRPAILEKGQGRIILKASRHACVEVQDEIAFIPNDVYFEKDKQMFHIITGPNMGGKSTYIRQTGVIVLMAQIGCFVPCESAEVSIVDCILARVGAGDSQLKGVSTFMAEMLETASILRSATKDSLIIIDELGRGTSTYDGFGLAWAISEYIATKIGAFCMFATHFHELTALANQIPTVNNLHVTALTTEETLTMLYQVKKGVCDQSFGIHVAELANFPKHVIECAKQKALELEEFQYIGESQGYDIMEPAAKKCYLEREQGEKIIQEFLSKVKQMPFTEMSEENITIKLKQLKAEVIAKNNSFVNEIISRIKVTT",
        }
    },
    {
        "protein": {
            "id": "B",
            "sequence": "MSRQSTLYSFFPKSPALSDANKASARASREGGRAAAAPGASPSPGGDAAWSEAGPGPRPLARSASPPKAKNLNGGLRRSVAPAAPTSCDFSPGDLVWAKMEGYPWWPCLVYNHPFDGTFIREKGKSVRVHVQFFDDSPTRGWVSKRLLKPYTGSKSKEAQKGGHFYSAKPEILRAMQRADEALNKDKIKRLELAVCDEPSEPEEEEEMEVGTTYVTDKSEEDNEIESEEEVQPKTQGSRRSSRQIKKRRVISDSESDIGGSDVEFKPDTKEEGSSDEISSGVGDSESEGLNSPVKVARKRKRMVTGNGSLKRKSSRKETPSATKQATSISSETKNTLRAFSAPQNSESQAHVSGGGDDSSRPTVWYHETLEWLKEEKRRDEHRRRPDHPDFDASTLYVPEDFLNSCTPGMRKWWQIKSQNFDLVICYKVGKFYELYHMDALIGVSELGLVFMKGNWAHSGFPEIAFGRYSDSLVQKGYKVARVEQTETPEMMEARCRKMAHISKYDRVVRREICRIITKGTQTYSVLEGDPSENYSKYLLSLKEKEEDSSGHTRAYGVCFVDTSLGKFFIGQFSDDRHCSRFRTLVAHYPPVQVLFEKGNLSKETKTILKSSLSCSLQEGLIPGSQFWDASKTLRTLLEEEYFREKLSDGIGVMLPQVLKGMTSESDSIGLTPGEKSELALSALGGCVFYLKKCLIDQELLSMANFEEYIPLDSDTVSTTRSGAIFTKAYQRMVLDAVTLNNLEIFLNGTNGSTEGTLLERVDTCHTPFGKRLLKQWLCAPLCNHYAINDRLDAIEDLMVVPDKISEVVELLKKLPDLERLLSKIHNVGSPLKSQNHPDSRAIMYEETTYSKKKIIDFLSALEGFKVMCKIIGIMEEVADGFKSKILKQVISLQTKNPEGRFPDLTVELNRWDTAFDHEKARKTGLITPKAGFDSDYDQALADIRENEQSLLEYLEKQRNRIGCRTIVYWGIGRNRYQLEIPENFTTRNLPEEYELKSTKKGCKRYWTKTIEKKLANLINAEERRDVSLKDCMRRLFYNFDKNYKDWQSAVECIAVLDVLLCLANYSRGGDGPMCRPVILLPEDTPPFLELKGSRHPCITKTFFGDDFIPNDILIGCEEEEQENGKAYCVLVTGPNMGGKSTLMRQAGLLAVMAQMGCYVPAEVCRLTPIDRVFTRLGASDRIMSGESTFFVELSETASILMHATAHSLVLVDELGRGTATFDGTAIANAVVKELAETIKCRTLFSTHYHSLVEDYSQNVAVRLGHMACMVENECEDPSQETITFLYKFIKGACPKSYGFNAARLANLPEEVIQKGHRKAREFEKMNQSLRLFREVCLASERSTVDAEAVHKLLTLIKEL",
        }
    },
    {"dna": {"id": "C", "sequence": "ATGCTTACAGTAACTGCTGATGATCTTACCTGAACTGATGCCATTTGCTGATATGACCTTGAATGCTAGTCTGACATTAAGCTTGATGCTTACAGGTTATGATCCTGAACTGTTGATGACTTACAGCTGATGTTAC"}},
    {"dna": {"id": "D", "sequence": "GTAACATCAGCTGTAAGTCATCAACAGTTCAGGATCATAACCTGTAAGCATCAAGCTTAATGTCAGACTAGCATTTCAAGGTCATATCAGCAAATGGCATCAGTTCAGGTAAGATCATCAGCAGTTACTGTAAGCAT"}},
    {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
    {"ligand": {"id": "F", "ccdCodes": ["MG"]}},
    {"ligand": {"id": "G", "ccdCodes": ["ATP"]}},
    {"ligand": {"id": "H", "ccdCodes": ["MG"]}},
],
"dialect": "alphafold3",
"version": 2,
"""
# {"dna": {"id": "C", "sequence": "GGATCGTTTTTTTTTGTCGAC"}},
# {"dna": {"id": "D", "sequence": "GTCGACAAAAAAAACGATCC"}},

mutations: dict[str, tuple[str, int, str]] = {}

if MAX_NT_SUBS is not None:
    # Enumerate every substitution reachable within MAX_NT_SUBS nucleotide changes
    _seq = base_json["sequences"][0]["protein"]["sequence"]
    for _i, _orig in enumerate(_seq):
        if _orig not in _AA_TO_CODONS:
            continue
        for _new in sorted(_AA_TO_CODONS):
            if _new == _orig:
                continue
            if min_nt_distance(_orig, _new) > MAX_NT_SUBS:
                continue
            _name = f"{_orig}{_i + 1}{_new}"
            if _name not in mutations_done:
                mutations[_name] = (_orig, _i + 1, _new)
    print(f"Enumerated {len(mutations)} mutations with MAX_NT_SUBS={MAX_NT_SUBS}")
elif MUTS_DF is not None:
    skipped_other_gene = 0
    for _, row in MUTS_DF.iterrows():
        if not clinvar_row_matches_gene(row, TARGET_GENE):
            skipped_other_gene += 1
            continue

        mut_name = clinvar_mutation_text(row)
        if mut_name is None:
            print("Skipping ClinVar row without a mutation annotation.")
            continue
        try:
            orig_aa, pos, new_aa = parse_mutation_name(mut_name)
        except Exception as e:
            print(f"Error parsing mutation {mut_name}: {e}")
            continue
        normalized_name = f"{orig_aa}{pos}{new_aa}"
        if normalized_name in mutations_done:
            print(f"Skipping {normalized_name} as it's already done.")
            continue
        mutations[normalized_name] = (orig_aa, pos, new_aa)
    print(
        f"Loaded {len(mutations)} {TARGET_GENE or 'unfiltered'} mutations from ClinVar CSV "
        f"(skipped {skipped_other_gene} rows annotated to other genes)"
    )

if FORCE_ADD_MUTATIONS is not None:
    for _forced in FORCE_ADD_MUTATIONS:
        try:
            _fo, _fp, _fn = parse_mutation_name(_forced)
        except Exception as e:
            print(f"FORCE_ADD: could not parse '{_forced}': {e}")
            continue
        _fname = f"{_fo}{_fp}{_fn}"
        if _fname in mutations_done:
            print(f"FORCE_ADD: skipping {_fname} (already done)")
        elif _fname not in mutations:
            mutations[_fname] = (_fo, _fp, _fn)
            print(f"FORCE_ADD: added {_fname}")
        else:
            print(f"FORCE_ADD: {_fname} already present")


def generate_ensembles():
    seq = base_json["sequences"][0]["protein"]["sequence"]
    output_dir = "muts_jsons"
    os.makedirs(output_dir, exist_ok=True)
    # ZIP_TAG keeps a targeted re-run from overwriting an earlier batch's zip.
    # The JSON names inside, and therefore the AF3 job names, stay "{MUT_NAME}_*"
    # so the new archives match the existing ones.
    zip_path = os.path.join(
        output_dir, f"{MUT_NAME}_Mutations{ZIP_TAG}.zip"
    )

    with zipfile.ZipFile(zip_path, "w") as zf:

        # --- WT entry ---
        wt_json = copy.deepcopy(base_json)
        wt_json["name"] = f"{MUT_NAME}_WT"
        zf.writestr(f"{MUT_NAME}_WT.json", json.dumps(wt_json, indent=2))
        print(f"Successfully packaged: {MUT_NAME}_WT.json")

        # --- Mutation entries ---
        for mut_name, (orig_aa, pos, new_aa) in mutations.items():

            # Zero-index the position for Python
            idx = pos - 1

            # The Skeptical Validation Check
            actual_aa = seq[idx]
            if actual_aa != orig_aa:
                print(
                    f"ERROR on {mut_name}: Expected {orig_aa} at position {pos}, but found {actual_aa}. Skipping."
                )
                continue

            # Apply the substitution
            mutated_seq = seq[:idx] + new_aa + seq[idx + 1 :]

            # Create a deep copy to avoid overwriting the base template
            new_json = copy.deepcopy(base_json)
            new_json["name"] = f"{MUT_NAME}_{mut_name}"
            new_json["sequences"][0]["protein"]["sequence"] = mutated_seq

            # Write directly to the zip archive
            json_str = json.dumps(new_json, indent=2)
            zf.writestr(f"{MUT_NAME}_{mut_name}.json", json_str)
            print(f"Successfully packaged: {MUT_NAME}_{mut_name}.json")

        print(f"Saved archive: {zip_path}")

if __name__ == "__main__":
    generate_ensembles()
