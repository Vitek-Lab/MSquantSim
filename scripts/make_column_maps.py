"""Precompute UniProt -> HGNC column maps so the cluster jobs don't need protmapper/INDRA.

Mirrors the renaming done in the MEL (cell 5) and PDAC DIA (cell 6) notebooks.
"""
import json
import pandas as pd
from protmapper import uniprot_client
from indra.databases import hgnc_client


def hgnc_converter(id, mnemonic_id=False):
    if mnemonic_id:
        uniprot_id = uniprot_client.get_id_from_mnemonic(id)
    else:
        uniprot_id = id
    hgnc_id = uniprot_client.get_hgnc_id(uniprot_id)
    hgnc_name = hgnc_client.get_hgnc_name(hgnc_id)
    return hgnc_name


# ---- MEL: split on '|', convert every column except Condition ----
mel = pd.read_csv("data/mel/MEL_protein_level_data.csv", nrows=1)
mel = mel.loc[:, ~mel.columns.str.contains(';', na=False)]
mel.columns = [c.split('|')[1] if '|' in c else c for c in mel.columns]
mel = mel.drop(columns=["Run", "Channel"])

mel_map = {}
for col in mel.columns:
    if col == "Condition":
        mel_map[col] = "Condition"
        continue
    try:
        mel_map[col] = hgnc_converter(col, mnemonic_id=False)
    except Exception:
        mel_map[col] = None

with open("data/mel/MEL_column_map.json", "w") as f:
    json.dump(mel_map, f, indent=1)
resolved = sum(v is not None for v in mel_map.values())
print(f"MEL: {len(mel_map)} columns, {resolved} resolved, {len(mel_map) - resolved} -> None")

# ---- PDAC DIA: mnemonic ids, keep originalRUN/GROUP, drop failures ----
dia = pd.read_csv("data/pdac/PDAC_DIA_protein_level_data.csv", nrows=1)
dia = dia.loc[:, ~dia.columns.str.contains(';', na=False)]

keep = ["originalRUN", "GROUP"]
dia_map = {}
for col in dia.columns:
    if col in keep:
        dia_map[col] = col
        continue
    try:
        dia_map[col] = hgnc_converter(col, mnemonic_id=True)
    except Exception:
        dia_map[col] = "__DROP__"   # notebook drops columns whose conversion raises

with open("data/pdac/PDAC_DIA_column_map.json", "w") as f:
    json.dump(dia_map, f, indent=1)
dropped = sum(v == "__DROP__" for v in dia_map.values())
nones = sum(v is None for v in dia_map.values())
print(f"PDAC DIA: {len(dia_map)} columns, {dropped} raised -> dropped, {nones} -> None")
