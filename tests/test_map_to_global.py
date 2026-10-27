"""Query rows map through the exon table supplied by the ``*2`` arguments."""
import numpy as np
import pytest

from ruranges.numpy import map_to_global


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("query_forward", [True, False])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_query_spanning_two_exons(dtype, forward, query_forward):
    genome_starts = [100, 200] if forward else [200, 100]
    idx, starts, ends, strand = map_to_global(
        starts=np.array([5], dtype=dtype),
        ends=np.array([15], dtype=dtype),
        groups=np.array([7]),
        strand=np.array([query_forward]),
        starts2=np.array([0, 10]),
        ends2=np.array([10, 20]),
        groups2=np.array([7, 7]),
        chr_code2=np.array([1, 1]),
        genome_start2=np.array(genome_starts),
        genome_end2=np.array(genome_starts) + 10,
        strand2=np.array([forward, forward]),
    )
    np.testing.assert_array_equal(idx, [0, 0])
    np.testing.assert_array_equal(starts, [105, 200] if forward else [200, 105])
    np.testing.assert_array_equal(ends, [110, 205] if forward else [205, 110])
    np.testing.assert_array_equal(strand, [forward == query_forward] * 2)
    assert starts.dtype == ends.dtype == np.dtype(dtype)


def test_equal_query_and_exon_counts_do_not_exchange_local_coordinates():
    idx, starts, ends, strand = map_to_global(
        starts=np.array([5, 15]), ends=np.array([10, 20]),
        groups=np.array([0, 0]), strand=np.array([True, False]),
        starts2=np.array([0, 10]), ends2=np.array([10, 20]),
        groups2=np.array([0, 0]), strand2=np.array([True, True]),
        chr_code2=np.array([1, 1]),
        genome_start2=np.array([100, 200]), genome_end2=np.array([110, 210]),
    )
    np.testing.assert_array_equal(idx, [0, 1])
    np.testing.assert_array_equal(starts, [105, 205])
    np.testing.assert_array_equal(ends, [110, 210])
    np.testing.assert_array_equal(strand, [True, False])


@pytest.mark.parametrize("shift", [0, 2**32])
def test_query_attribution_across_transcripts_with_strided_inputs(shift):
    def view(values, dtype=np.int64):
        storage = np.zeros(2 * len(values), dtype=dtype)
        storage[::2] = values
        return storage[::2]

    idx, starts, ends, strand = map_to_global(
        starts=view([2, 4, 3]), ends=view([5, 7, 6]),
        groups=view([1, 1, 2]), strand=view([True, False, True], bool),
        starts2=view([0, 0]), ends2=view([10, 10]),
        groups2=view([1, 2]), strand2=view([True, False], bool),
        chr_code2=view([1, 2]),
        genome_start2=view([shift + 100, shift + 200]),
        genome_end2=view([shift + 110, shift + 210]),
    )
    np.testing.assert_array_equal(idx, [0, 1, 2])
    np.testing.assert_array_equal(starts, np.array([102, 104, 204]) + shift)
    np.testing.assert_array_equal(ends, np.array([105, 107, 207]) + shift)
    np.testing.assert_array_equal(strand, [True, False, False])


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("sort_output", [False, True])
def test_unsorted_projection_retains_original_query_and_exon_indices(dtype, sort_output):
    out = map_to_global(
        starts=np.array([5, 5, 0], dtype=dtype),
        ends=np.array([15, 15, 2], dtype=dtype),
        groups=np.array([7, 1, 7]), strand=np.array([True, False, True]),
        starts2=np.array([10, 0, 0, 10], dtype=dtype),
        ends2=np.array([20, 10, 10, 20], dtype=dtype),
        groups2=np.array([7, 1, 7, 1]), chr_code2=np.array([1, 1, 1, 1]),
        genome_start2=np.array([200, 1000, 100, 2000], dtype=dtype),
        genome_end2=np.array([210, 1010, 110, 2010], dtype=dtype),
        strand2=np.array([True, True, True, True]),
        sort_output=sort_output, return_exon_indices=True,
    )
    expected = [(0, 200, 205, True, 0), (0, 105, 110, True, 2),
                (1, 1005, 1010, False, 1), (1, 2000, 2005, False, 3),
                (2, 100, 102, True, 2)]
    actual = list(zip(*(a.tolist() for a in out)))
    assert (actual if sort_output else sorted(actual)) == (expected if sort_output else sorted(expected))


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("query_start,query_end,local_start,local_end", [
    (-5, 5, 0, 10), (0, 5, 10, 20), (5, 15, 10, 20), (15, 25, 10, 20),
])
def test_projection_clips_both_exon_boundaries(forward, query_start, query_end, local_start, local_end):
    out = map_to_global(
        starts=np.array([query_start]), ends=np.array([query_end]),
        groups=np.array([7]), strand=np.array([True]),
        starts2=np.array([local_start]), ends2=np.array([local_end]),
        groups2=np.array([7]), chr_code2=np.array([1]),
        genome_start2=np.array([100]), genome_end2=np.array([110]),
        strand2=np.array([forward]),
    )
    lo, hi = max(query_start, local_start), min(query_end, local_end)
    if lo >= hi:
        assert all(a.size == 0 for a in out)
    else:
        expected = (100 + lo - local_start, 100 + hi - local_start) if forward else (110 - hi + local_start, 110 - lo + local_start)
        np.testing.assert_array_equal(out[0], [0])
        np.testing.assert_array_equal(out[1], [expected[0]])
        np.testing.assert_array_equal(out[2], [expected[1]])
        np.testing.assert_array_equal(out[3], [forward])
