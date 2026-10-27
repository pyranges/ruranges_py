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
