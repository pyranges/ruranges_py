"""Every kernel accepts strided arrays, as pandas >= 3.1 can return from Series.values.

The Rust kernels read their inputs as slices, which need C-contiguous memory.
Each case runs with small coordinates, which the wrapper casts to int32 (a
copy), and with coordinates past int32, which it passes through as int64 (no
copy), so a strided array reaches the kernel unless the wrapper fixes it.
"""

import numpy as np
import pytest

from ruranges import numpy as rn

BIG = 2**40


def strided(values, dtype=np.int64):
    block = np.zeros((len(values), 3), dtype=dtype)
    block[:, 1] = values
    column = block[:, 1]
    assert not column.flags.c_contiguous
    return column


def starts(lo):
    return strided([lo, lo + 4])


def ends(lo):
    return strided([lo + 2, lo + 8])


def groups():
    return strided([0, 0], np.uint32)


def negative_strand():
    return strided([False, True], np.bool_)


def pair(lo):
    return dict(starts=starts(lo), ends=ends(lo), starts2=starts(lo), ends2=ends(lo), groups=groups(), groups2=groups())


CASES = {
    "overlaps": lambda lo: rn.overlaps(**pair(lo)),
    "nearest": lambda lo: rn.nearest(**pair(lo)),
    "subtract": lambda lo: rn.subtract(**pair(lo)),
    "count_overlaps": lambda lo: rn.count_overlaps(**pair(lo)),
    "sort_intervals": lambda lo: rn.sort_intervals(
        starts=starts(lo), ends=ends(lo), groups=groups(), sort_reverse_direction=negative_strand()
    ),
    "cluster": lambda lo: rn.cluster(starts=starts(lo), ends=ends(lo), groups=groups()),
    "merge": lambda lo: rn.merge(starts=starts(lo), ends=ends(lo), groups=groups()),
    "max_disjoint": lambda lo: rn.max_disjoint(starts=starts(lo), ends=ends(lo), groups=groups()),
    "extend": lambda lo: rn.extend(
        starts=starts(lo), ends=ends(lo), groups=groups(), negative_strand=negative_strand(), ext_3=1, ext_5=1
    ),
    "window": lambda lo: rn.window(
        starts=starts(lo), ends=ends(lo), groups=groups(), negative_strand=negative_strand(), window_size=2
    ),
    "tile": lambda lo: rn.tile(starts=starts(lo), ends=ends(lo), negative_strand=negative_strand(), tile_size=2),
    "group_cumsum": lambda lo: rn.group_cumsum(
        starts=starts(lo), ends=ends(lo), groups=groups(), negative_strand=negative_strand()
    ),
    "spliced_subsequence": lambda lo: rn.spliced_subsequence(
        starts=starts(lo), ends=ends(lo), groups=groups(), strand_flags=negative_strand(), start=0, end=1
    ),
    "complement": lambda lo: rn.complement(
        starts=starts(lo),
        ends=ends(lo),
        groups=groups(),
        chrom_len_ids=strided([0, 1], np.uint32),
        chrom_lens=strided([lo + 100, lo + 100]),
    ),
    "genome_bounds": lambda lo: rn.genome_bounds(
        groups=groups(), starts=starts(lo), ends=ends(lo), chrom_length=strided([lo + 100, lo + 100])
    ),
    "sort_groups": lambda lo: rn.sort_groups(groups()),
    "fold_ranks": lambda lo: rn.fold_ranks(groups(), 1, groups(), 1),
}


@pytest.mark.parametrize("lo", [1, BIG])
@pytest.mark.parametrize("name", CASES)
def test_kernel_accepts_strided_input(name, lo):
    CASES[name](lo)


@pytest.mark.parametrize("lo", [1, BIG])
def test_strided_projection_checks_coordinates_and_query_attribution(lo):
    # One query spans two exons. Exchanging the tables must fail this check.
    idx, out_starts, out_ends, strand = rn.map_to_global(
        starts=strided([1, 3]), ends=strided([5, 6]), groups=strided([7, 7], np.uint32),
        strand=strided([True, True], np.bool_),
        starts2=strided([0, 2, 6]), ends2=strided([2, 6, 8]),
        groups2=strided([7, 7, 7], np.uint32), chr_code2=strided([1, 1, 1], np.uint32),
        genome_start2=strided([lo, lo + 10, lo + 20]),
        genome_end2=strided([lo + 2, lo + 14, lo + 22]),
        strand2=strided([True, True, True], np.bool_),
    )
    np.testing.assert_array_equal(idx, [0, 0, 1])
    np.testing.assert_array_equal(out_starts, [lo + 1, lo + 10, lo + 11])
    np.testing.assert_array_equal(out_ends, [lo + 2, lo + 13, lo + 14])
    np.testing.assert_array_equal(strand, [True, True, True])
