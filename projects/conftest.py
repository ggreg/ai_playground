"""Shared fixture for the from-scratch project acceptance tests.

Policy (mirrors tests/milestones/): a stub that still raises NotImplementedError means
the reader hasn't built that part yet -> pytest.skip. Any other outcome is judged by the
test itself — wrong output FAILS. Green means done; nothing else does.
"""

import pytest


@pytest.fixture
def attempt():
    """Call reader code, converting NotImplementedError into a skip (not a failure)."""

    def _attempt(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError:
            pytest.skip("not implemented yet — this part of the project hasn't been built")

    return _attempt
