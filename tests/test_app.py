"""The dashboard runs end to end on the committed results, for every sector and model."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_default_view():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.startswith("Singapore sales outlook")
    assert len(at.dataframe) == 1


@pytest.mark.parametrize("sector", ["Retail: Total", "Retail: Watches & Jewellery", "Food & beverage: Restaurants"])
def test_app_handles_each_model(sector):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].set_value(sector).run()
    seen = set()
    for i, label in enumerate(at.selectbox[1].options):
        at.selectbox[1].select_index(i).run()
        assert not at.exception, (sector, label)
        text = " ".join(m.value for m in at.markdown)
        assert label in text  # the track-record sentence names the selected model
        seen.add(text)
    assert len(seen) == 4
