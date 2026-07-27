# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Per-stream TPC-H substitution parameters (spec clause 2.4).

stream_params(stream, seed, sf) draws one value set per query for a stream,
like qgen -p <stream> does. Draws are deterministic for a given (seed, stream)
so runs are reproducible; stream 0 is the power run.
"""

import random
from datetime import date, timedelta

REGIONS = ("AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST")

# nation -> REGIONS index, per dbgen dists.dss
NATIONS = (
    ("ALGERIA", 0),
    ("ARGENTINA", 1),
    ("BRAZIL", 1),
    ("CANADA", 1),
    ("EGYPT", 4),
    ("ETHIOPIA", 0),
    ("FRANCE", 3),
    ("GERMANY", 3),
    ("INDIA", 2),
    ("INDONESIA", 2),
    ("IRAN", 4),
    ("IRAQ", 4),
    ("JAPAN", 2),
    ("JORDAN", 4),
    ("KENYA", 0),
    ("MOROCCO", 0),
    ("MOZAMBIQUE", 0),
    ("PERU", 1),
    ("CHINA", 2),
    ("ROMANIA", 3),
    ("SAUDI ARABIA", 4),
    ("VIETNAM", 2),
    ("RUSSIA", 3),
    ("UNITED KINGDOM", 3),
    ("UNITED STATES", 1),
)
NATION_NAMES = tuple(name for name, _ in NATIONS)

SEGMENTS = ("AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD")
TYPE_S1 = ("STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO")
TYPE_S2 = ("ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED")
TYPE_S3 = ("TIN", "NICKEL", "BRASS", "STEEL", "COPPER")
CONTAINER_S1 = ("SM", "LG", "MED", "JUMBO", "WRAP")
CONTAINER_S2 = ("CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM")
SHIPMODES = ("REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB")
Q13_WORD1 = ("special", "pending", "unusual", "express")
Q13_WORD2 = ("packages", "requests", "accounts", "deposits")

# The 92 P_NAME words (dbgen dists.dss colors), used by q9 and q20.
COLORS = (
    "almond",
    "antique",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanched",
    "blue",
    "blush",
    "brown",
    "burlywood",
    "burnished",
    "chartreuse",
    "chiffon",
    "chocolate",
    "coral",
    "cornflower",
    "cornsilk",
    "cream",
    "cyan",
    "dark",
    "deep",
    "dim",
    "dodger",
    "drab",
    "firebrick",
    "floral",
    "forest",
    "frosted",
    "gainsboro",
    "ghost",
    "goldenrod",
    "green",
    "grey",
    "honeydew",
    "hot",
    "indian",
    "ivory",
    "khaki",
    "lace",
    "lavender",
    "lawn",
    "lemon",
    "light",
    "lime",
    "linen",
    "magenta",
    "maroon",
    "medium",
    "metallic",
    "midnight",
    "mint",
    "misty",
    "moccasin",
    "navajo",
    "navy",
    "olive",
    "orange",
    "orchid",
    "pale",
    "papaya",
    "peach",
    "peru",
    "pink",
    "plum",
    "powder",
    "puff",
    "purple",
    "red",
    "rose",
    "rosy",
    "royal",
    "saddle",
    "salmon",
    "sandy",
    "seashell",
    "sienna",
    "sky",
    "slate",
    "smoke",
    "snow",
    "spring",
    "steel",
    "tan",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "yellow",
)


def _month(k):
    """First day of the k-th month counting from 1993-01."""
    return date(1993 + k // 12, k % 12 + 1, 1)


def _add_months(d, n):
    return _month((d.year - 1993) * 12 + d.month - 1 + n)


def _brand(rng):
    return f"{rng.randint(1, 5)}{rng.randint(1, 5)}"


def stream_params(stream, seed=1, sf=1):
    """Substitution parameters for every query of one stream."""
    rng = random.Random(seed * 4096 + stream)
    p = {}

    d = date(1998, 12, 1) - timedelta(days=rng.randint(60, 120))
    p["q1"] = {"DATE": d.isoformat()}
    p["q2"] = {
        "SIZE": rng.randint(1, 50),
        "TYPE": rng.choice(TYPE_S3),
        "REGION": rng.choice(REGIONS),
    }
    p["q3"] = {
        "SEGMENT": rng.choice(SEGMENTS),
        "DATE": date(1995, 3, rng.randint(1, 31)).isoformat(),
    }
    d = _month(rng.randint(0, 57))
    p["q4"] = {"DATE": d.isoformat(), "DATE_END": _add_months(d, 3).isoformat()}
    year = rng.randint(1993, 1997)
    p["q5"] = {
        "REGION": rng.choice(REGIONS),
        "DATE": date(year, 1, 1).isoformat(),
        "DATE_END": date(year + 1, 1, 1).isoformat(),
    }
    year = rng.randint(1993, 1997)
    p["q6"] = {
        "DATE": date(year, 1, 1).isoformat(),
        "DATE_END": date(year + 1, 1, 1).isoformat(),
        "DISCOUNT": f"0.0{rng.randint(2, 9)}",
        "QUANTITY": rng.randint(24, 25),
    }
    nation1, nation2 = rng.sample(NATION_NAMES, 2)
    p["q7"] = {"NATION1": nation1, "NATION2": nation2}
    nation, region_idx = rng.choice(NATIONS)
    p["q8"] = {
        "NATION": nation,
        "REGION": REGIONS[region_idx],
        "TYPE": f"{rng.choice(TYPE_S1)} {rng.choice(TYPE_S2)} {rng.choice(TYPE_S3)}",
    }
    p["q9"] = {"COLOR": rng.choice(COLORS)}
    d = _month(rng.randint(1, 24))
    p["q10"] = {"DATE": d.isoformat(), "DATE_END": _add_months(d, 3).isoformat()}
    p["q11"] = {
        "NATION": rng.choice(NATION_NAMES),
        "FRACTION": f"{0.0001 / sf:.10f}",
    }
    mode1, mode2 = rng.sample(SHIPMODES, 2)
    year = rng.randint(1993, 1997)
    p["q12"] = {
        "SHIPMODE1": mode1,
        "SHIPMODE2": mode2,
        "DATE": date(year, 1, 1).isoformat(),
        "DATE_END": date(year + 1, 1, 1).isoformat(),
    }
    p["q13"] = {"WORD1": rng.choice(Q13_WORD1), "WORD2": rng.choice(Q13_WORD2)}
    d = _month(rng.randint(0, 59))
    p["q14"] = {"DATE": d.isoformat(), "DATE_END": _add_months(d, 1).isoformat()}
    d = _month(rng.randint(0, 57))
    p["q15"] = {"DATE": d.isoformat(), "DATE_END": _add_months(d, 3).isoformat()}
    p["q16"] = {
        "BRAND": _brand(rng),
        "TYPE": f"{rng.choice(TYPE_S1)} {rng.choice(TYPE_S2)}",
        "SIZES": ", ".join(str(s) for s in rng.sample(range(1, 51), 8)),
    }
    p["q17"] = {
        "BRAND": _brand(rng),
        "CONTAINER": f"{rng.choice(CONTAINER_S1)} {rng.choice(CONTAINER_S2)}",
    }
    p["q18"] = {"QUANTITY": rng.randint(312, 315)}
    p["q19"] = {
        "BRAND1": _brand(rng),
        "QTY1": rng.randint(1, 10),
        "BRAND2": _brand(rng),
        "QTY2": rng.randint(10, 20),
        "BRAND3": _brand(rng),
        "QTY3": rng.randint(20, 30),
    }
    year = rng.randint(1993, 1997)
    p["q20"] = {
        "COLOR": rng.choice(COLORS),
        "DATE": date(year, 1, 1).isoformat(),
        "DATE_END": date(year + 1, 1, 1).isoformat(),
        "NATION": rng.choice(NATION_NAMES),
    }
    p["q21"] = {"NATION": rng.choice(NATION_NAMES)}
    p["q22"] = {"CODES": ", ".join(f"'{c}'" for c in rng.sample(range(10, 35), 7))}
    return p
