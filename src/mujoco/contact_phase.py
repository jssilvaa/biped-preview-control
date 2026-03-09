from __future__ import annotations
from dataclasses import dataclass
import numpy as np 

from damping_control import DampingGains


@dataclass(frozen=True)
class PhaseGains:
    contact: DampingGains
    noncontact: DampingGains
    single_contact_linear_relax: bool = True  # from the paper: if only one limb is in contact linear gains use noncontact

    @staticmethod
    def murooka_table_ii() -> PhaseGains:
        # Table II
        contact = DampingGains(
            Kd=np.array([10000, 10000, 10000, 100, 100, 100], dtype=float),
            Ks=np.array([0, 0, 0, 500, 500, 2000], dtype=float),
            Kf=np.array([1, 1, 1, 1, 1, 0], dtype=float),
        )
        noncontact = DampingGains(
            Kd=np.array([300, 300, 300, 40, 40, 40], dtype=float),
            Ks=np.array([2250, 2250, 2250, 400, 400, 400], dtype=float),
            Kf=np.array([0, 0, 0, 0, 0, 0], dtype=float),
        )
        return PhaseGains(contact=contact, noncontact=noncontact, single_contact_linear_relax=True)
    

@dataclass
class ContactHysteresis:
    fn_on: float = 30.0     # [N], tune later
    fn_off: float = 10.0    # [N], tune later
    active: np.ndarray | None = None # (nc,)

    def reset(self, nc: int): 
        self.active = np.zeros(nc, dtype=bool)

    def update(self, fn: np.ndarray) -> np.ndarray:
        nc = fn.shape[0]
        self.active = np.zeros(nc, dtype=bool) if self.active is None else self.active
        for i in range(fn.shape[0]):
            self.active[i] = fn[i] >= (self.fn_off if self.active[i] else self.fn_on)
        return self.active.copy()

def normal_force_in_patch_frame(R_wc: np.ndarray, F_world: np.ndarray) -> float: 
    """
    Returns f_n in patch contact frame (i.e. only z component)
    """
    R_wc = np.asarray(R_wc, dtype=float).reshape(3,3)
    F_world = np.asarray(F_world, dtype=float).reshape(3,)
    f_c = R_wc.T @ F_world 
    return f_c[2]


def select_patch_gains(
    active: np.ndarray,
    gains: PhaseGains,
) -> list[DampingGains]:
    active = np.asarray(active, dtype=bool).reshape(-1)
    nc = active.shape[0]
    out = [gains.noncontact for _ in range(nc)]
    for i in range(nc):
        out[i] = gains.contact if active[i] else gains.noncontact

    if gains.single_contact_linear_relax and int(np.sum(active)) == 1:
        i = int(np.flatnonzero(active)[0])
        # replace linear (first 3) components with noncontact values
        c = gains.contact
        n = gains.noncontact
        out[i] = DampingGains(
            Kd=np.hstack([n.Kd[:3], c.Kd[3:]]),
            Ks=np.hstack([n.Ks[:3], c.Ks[3:]]),
            Kf=np.hstack([n.Kf[:3], c.Kf[3:]]),
        )

    return out