import numpy as np
import pylab as plt


def within(val, bounds):
    return bounds[0] <= val <= bounds[1]


class Bounds:
    """Sort antennas into good/suspect/bad categories based on bounds."""

    def __init__(self, absolute, good):
        self.abs_bound = absolute
        self.good_bound = good
        self.bad = set()
        self.suspect = set()
        self.good = set()

    def classify(self, k, val):
        """Assign k to internal sets of good/suspect/bad based on value."""
        if not within(val, self.abs_bound):
            self.bad.add(k)
        elif not within(val, self.good_bound):
            self.suspect.add(k)
        else:
            self.good.add(k)


def _antenna_str(ants):
    """Turn a set of (ant, pol) keys into a string."""
    return ",".join(["%d%s" % (ant[0], ant[1][-1]) for ant in sorted(ants)])


class AntennaClassification:
    """Injests Bounds to create sets of good/suspect/bad antennas."""

    def __init__(self, *bounds_list):
        self.clear()
        for b in bounds_list:
            self.add_bounds(b)

    def clear(self):
        """Clear good/suspect/bad sets."""
        self.bad = set()
        self.suspect = set()
        self.good = set()

    def add_bounds(self, bound):
        """Add antennas from Bounds to good/suspect/bad sets and remove
        intersections from superior categories."""
        self.bad.update(bound.bad)
        self.suspect.update(bound.suspect)
        self.good.update(bound.good)
        self.good.difference_update(self.bad)  # remove bad from good
        self.good.difference_update(self.suspect)  # remove suspect from good
        self.suspect.difference_update(self.bad)  # remove bad from suspect

    def __str__(self):
        s = []
        s.append(f"Good: {_antenna_str(self.good)}")
        s.append(f"Suspect: {_antenna_str(self.suspect)}")
        s.append(f"Bad: {_antenna_str(self.bad)}")
        return "\n\n".join(s)

    def is_good(self, k):
        return k in self.good

    def is_bad(self, k):
        return k in self.bad


def classify_antennas(CEN_FREQ=136e6, RFI_THRESH=1e-2):
    """
    """
    # First-pass antenna classification based on auto levels
    CEN_FREQ = 136e6  # Hz
    RFI_THRESH = 1e-2  # fraction of mean

    pwr_bound = Bounds(absolute=(1, 50), good=(5, 20))
    slope_bound = Bounds(absolute=(-0.2, 0.2), good=(-0.12, 0.12))
    rfi_bound = Bounds(absolute=(0, 0.15), good=(0, 0.1))

    for k, v in autos.items():
        mean = np.mean(v, axis=0) / intcnt
        hi_pwr = np.median(mean[hc.freqs > CEN_FREQ])
        lo_pwr = np.median(mean[hc.freqs <= CEN_FREQ])
        pwr = 0.5 * (hi_pwr + lo_pwr)
        slope = (hi_pwr - lo_pwr) / pwr
        rfi = np.abs(mean[1:-1] - 0.5 * (mean[:-2] + mean[2:])) / mean[1:-1]
        rfi_frac = np.mean(np.where(rfi > RFI_THRESH, 1, 0))
        pwr_bound.classify(k, pwr)
        slope_bound.classify(k, slope)
        rfi_bound.classify(k, rfi_frac)

    ant_class = AntennaClassification(pwr_bound, slope_bound, rfi_bound)
