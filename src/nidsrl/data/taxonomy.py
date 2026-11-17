"""Shared coarse attack taxonomy across the NetFlow-v3 datasets.

Fine-grained attack names differ between datasets, so cross-dataset evaluation
uses these coarse groups for per-group recall. The mapping is a documented
modelling choice (see docs/data.md); unknown names fall back to 'other' and are
counted in the preparation report so nothing is silently misfiled.
"""

from __future__ import annotations

import re

BENIGN = "benign"
DOS = "dos"
RECON = "recon"
BRUTEFORCE = "bruteforce"
WEB = "web"
EXPLOIT = "exploit"
MALWARE = "malware"
OTHER = "other"

GROUPS = (BENIGN, DOS, RECON, BRUTEFORCE, WEB, EXPLOIT, MALWARE, OTHER)

_RAW_TO_GROUP = {
    # benign
    "benign": BENIGN,
    # denial of service (all DoS and DDoS variants)
    "dos": DOS,
    "ddos": DOS,
    "dosattacksgoldeneye": DOS,
    "dosattacksslowloris": DOS,
    "dosattacksslowhttptest": DOS,
    "dosattackshulk": DOS,
    "ddosattacksloichttp": DOS,
    "ddosattackloicudp": DOS,
    "ddosattackhoic": DOS,
    # reconnaissance / scanning
    "reconnaissance": RECON,
    "scanning": RECON,
    "analysis": RECON,
    # brute force / password guessing
    "ftpbruteforce": BRUTEFORCE,
    "sshbruteforce": BRUTEFORCE,
    "password": BRUTEFORCE,
    "bruteforceweb": BRUTEFORCE,
    # web application attacks
    "injection": WEB,
    "xss": WEB,
    "sqlinjection": WEB,
    "bruteforcexss": WEB,
    # exploitation
    "exploits": EXPLOIT,
    "fuzzers": EXPLOIT,
    "generic": EXPLOIT,
    "shellcode": EXPLOIT,
    # malware / persistence / infiltration
    "backdoor": MALWARE,
    "bot": MALWARE,
    "ransomware": MALWARE,
    "worms": MALWARE,
    "infilteration": MALWARE,
    "infiltration": MALWARE,
    # other
    "mitm": OTHER,
    "theft": OTHER,
}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def coarse_attack(name: str) -> str:
    """Map a dataset-specific attack name to a coarse group ('other' if unknown)."""
    return _RAW_TO_GROUP.get(_key(name), OTHER)


def is_known(name: str) -> bool:
    return _key(name) in _RAW_TO_GROUP
