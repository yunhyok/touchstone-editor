import pytest

from touchstone_editor.names import NameRule, extract_channel_name, natural_key, validate_names


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


@pytest.mark.parametrize("prefix", ["U1_0-J49", "Z3-2K17", "LGA-QB123", "LGA2-TH7", "UFP_TEST_9_20-F12"])
def test_device_pin_prefixes(prefix):
    assert extract_channel_name(f"{prefix} CHANNEL WITH SPACE") == "CHANNEL WITH SPACE"
    assert extract_channel_name(f"{prefix} BUS::NET") == "BUS::NET"
    assert extract_channel_name("HIGH-SPEED CHANNEL") == "HIGH-SPEED CHANNEL"
    assert extract_channel_name("BUS::NET") == "BUS::NET"
    assert extract_channel_name("Port19_U1::BUS::NET") == "BUS::NET"


def test_regex_channel_numbers_and_delimiters():
    rule = NameRule(operation="regex", find=r"^SITE([0-9]+)-(.*)$", replacement="FIXTURE-$2/$1")
    assert rule.apply("SITE12-SIGNAL_A") == "FIXTURE-SIGNAL_A/12"
    assert rule.apply("OTHER") == "OTHER"
    rule = NameRule(operation="regex", find=r"^U([0-9]+)_(.*)$", replacement="L$1-$2")
    assert rule.apply("U23-SIGNAL") == "U23-SIGNAL"
    assert rule.apply("U23_SIGNAL_A") == "L23-SIGNAL_A"
    rule = NameRule(operation="regex", find=r"^([0-9]+)_(.*)$", replacement="SOCKET-$2/$1")
    assert rule.apply("1_SIGNAL_A") == "SOCKET-SIGNAL_A/1"


@pytest.mark.parametrize("pattern,replacement", [("[", ""), ("(A)", "$2"), ("A{99999999999999999}", "")])
def test_invalid_regex_and_references(pattern, replacement):
    with pytest.raises(ValueError, match="정규식"):
        NameRule(operation="regex", find=pattern, replacement=replacement).apply("A")


def test_regex_unicode_empty_matches_and_literal_dollar():
    rule = NameRule(operation="regex", find=r"CH([0-9]+)", replacement="$$$1")
    assert rule.apply("🧪CH12/CH3") == "🧪$12/$3"
    assert NameRule(operation="regex", find="(?=.)", replacement="_").apply("🧪A") == "_🧪_A"
    assert rule.apply("다른 채널") == "다른 채널"


def test_regex_backtracking_limit_never_returns_partial_replacement():
    rule = NameRule(operation="regex", find=r"(*NO_START_OPT)CH[0-9]+|\G(.*)+X", replacement="changed")
    with pytest.raises(ValueError, match="한도"):
        rule.apply("CH1" + "A" * 30)
