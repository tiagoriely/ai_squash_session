# src/grammar_tools/planner.py
import random, math
from datetime import timedelta
from .loader import load_yaml_dir

POINTS_PER_CG = 7
REST_MIN      = 1.5
WARMUP_MIN    = 10

def pick_warmup(acts):
    warmups = [
        a for a in acts.values()
        if not a.is_abstract
        and any("warmup" in e for e in (a.extends or []))
    ]
    return random.choice(warmups)

def pick_block(acts, tag):
    if tag == "drill":
        pool = [a for a in acts.values()
                if not a.is_abstract and ".drill." in a.id]
    elif tag == "conditioned":
        pool = [a for a in acts.values()
                if (".conditioned." in a.id or ".front_and_back." in a.id
                    or ".boast_drive_cross." in a.id)]
    else:
        raise ValueError(tag)
    return random.choice(pool)

def build_session(total_min=60):
    acts   = load_yaml_dir()
    warmup = pick_warmup(acts)
    rem    = total_min - WARMUP_MIN
    sched  = []

    while rem >= (POINTS_PER_CG + REST_MIN):
        block = pick_block(acts, "conditioned")
        sched.append((block, POINTS_PER_CG))
        rem -= (POINTS_PER_CG + REST_MIN)

    return warmup, sched
