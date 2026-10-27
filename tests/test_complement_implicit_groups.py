import numpy as np
import pytest

from ruranges.numpy import complement


@pytest.mark.parametrize("chromosome", [0, 1, 23, 24, 2**32 - 1])
@pytest.mark.parametrize("include_first", [False, True])
def test_implicit_group_uses_the_supplied_chromosome(chromosome, include_first):
    kwargs = dict(
        starts=np.array([30, 10]),
        ends=np.array([40, 20]),
        chrom_len_ids=np.array([chromosome], dtype=np.uint64),
        chrom_lens=np.array([100]),
        include_first_interval=include_first,
    )
    implicit = complement(groups=None, **kwargs)
    explicit = complement(groups=np.full(2, chromosome, dtype=np.uint32), **kwargs)
    for actual, expected in zip(implicit, explicit):
        np.testing.assert_array_equal(actual, expected)
    n = 3 if include_first else 2
    np.testing.assert_array_equal(implicit[0], [chromosome] * n)
    np.testing.assert_array_equal(implicit[1], [0, 20, 40] if include_first else [20, 40])
    np.testing.assert_array_equal(implicit[2], [10, 30, 100] if include_first else [30, 100])
    assert implicit[0].dtype == np.uint32
    # The attribution index identifies the group, not a specific interval.
    assert np.all(implicit[3] < len(kwargs["starts"]))
