import pytest

from touchstone_editor.names import NameRule, natural_key, validate_names


def test_bulk_pipeline_and_unicode():
    rule = NameRule(find="old", replacement="new", whitespace=True,
                    special=True, prefix="P_", suffix="_OUT")
    assert rule.apply("old 채널 / 2") == "P_new채널2_OUT"
    assert NameRule(operation="channel").apply("Port19_U1::CH/2") == "CH/2"
    assert NameRule(operation="channel").apply("U1-A2 CH/2") == "CH/2"
    assert NameRule(operation="channel").apply("OTHER CHANNEL") == "OTHER CHANNEL"
    validate_names(["한글_채널/2"])


def test_literal_split_and_delete():
    assert NameRule(operation="delete", find=" /.").apply("A / B.2") == "AB2"
    assert NameRule(operation="before", find="/").apply("A/2/3") == "A"
    assert NameRule(operation="after", find="::").apply("U1::CH::P") == "CH::P"
    assert NameRule(operation="after", find="missing").apply("CH") == "CH"
    assert sorted(["CH10", "CH2", "CH1"], key=natural_key) == ["CH1", "CH2", "CH10"]
    assert sorted(["CH10", "CH2", "CH1²"], key=natural_key) == ["CH1²", "CH2", "CH10"]


@pytest.mark.parametrize("name", ["", "   ", "A\nB", "A!B", "A\x00B", "A\u200bB"])
def test_invalid_names(name):
    with pytest.raises(ValueError):
        validate_names([name])
