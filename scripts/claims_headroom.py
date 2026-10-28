from scripts.claims_base import CheckedClaim, boolean, is_true
from src.config import Technique
from src.headroom import CELLS, HEADROOM_P90, NO_HEADROOM_MEDIAN


def headroom_claims() -> list[CheckedClaim]:
    claims = []
    for technique in Technique:
        for n_cells in CELLS:
            base = f"headroom.results.{technique.key}.{n_cells}.verdict"
            where = f"{technique.label.lower()}, {n_cells} cells"
            claims.append(
                CheckedClaim(
                    f"No meaningful controller headroom: the rule's median gap to the "
                    f"ideal-transfer reference is below {NO_HEADROOM_MEDIAN * 100:.0f} % ({where})",
                    f"{base}.no_meaningful_headroom",
                    f"median gap < {NO_HEADROOM_MEDIAN * 100:.0f} %",
                    is_true,
                    boolean,
                    "Mixed 1 and Mixed 7 pooled, calibrated default models",
                )
            )
            claims.append(
                CheckedClaim(
                    f"Controller headroom exists: the 90th percentile gap exceeds "
                    f"{HEADROOM_P90 * 100:.0f} % ({where})",
                    f"{base}.headroom_exists",
                    f"90th percentile gap > {HEADROOM_P90 * 100:.0f} %",
                    is_true,
                    boolean,
                )
            )
    return claims
