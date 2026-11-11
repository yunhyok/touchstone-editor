"""Channel-name transformations with bounded regular expressions."""

import re
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=32)
def _bounded_regex(pattern: str):
    from PySide6.QtCore import QRegularExpression

    # PCRE2 limits need JIT disabled to bound accidental backtracking.
    expression = QRegularExpression("(*NO_JIT)(*LIMIT_MATCH=10000)(*LIMIT_DEPTH=1000)" + pattern)
    expression.setPatternOptions(QRegularExpression.UseUnicodePropertiesOption)
    if not expression.isValid():
        raise ValueError(f"정규식 오류: {expression.errorString()}")
    return expression


def _regex_replace(pattern: str, replacement: str, name: str) -> str:
    expression = _bounded_regex(pattern)
    tokens = re.split(r"(\$\$|\$[0-9]+)", replacement)
    for token in tokens:
        if token.startswith("$") and token[1:].isdecimal() and int(token[1:]) > expression.captureCount():
            raise ValueError(f"정규식 치환문 오류: {token} 그룹이 없습니다.")
    original = name.encode("utf-16-le")
    result = bytearray()
    position = 0
    matches = expression.globalMatch(name)
    while matches.hasNext():
        match = matches.next()
        result.extend(original[position:match.capturedStart() * 2])
        for token in tokens:
            if token == "$$":
                token = "$"
            elif token.startswith("$") and token[1:].isdecimal():
                token = match.captured(int(token[1:]))
            result.extend(token.encode("utf-16-le"))
        position = match.capturedEnd() * 2
    if not matches.isValid():
        raise ValueError("정규식 처리 한도를 초과했습니다. 검색식을 더 단순하게 바꾸세요.")
    result.extend(original[position:])
    return result.decode("utf-16-le")


def natural_key(value: str) -> list:
    return [int(part) if part.isdecimal() else part.casefold()
            for part in re.split(r"(\d+)", value)]


def extract_channel_name(name: str) -> str:
    parts = name.split(maxsplit=1)
    if len(parts) == 2 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*-[A-Za-z0-9]*[0-9][A-Za-z0-9]*", parts[0]):
        return parts[1]
    if re.match(r"Port[0-9]+_[^:\s]+::", name, re.IGNORECASE):
        return name.split("::", 1)[1]
    return name


@dataclass
class NameRule:
    operation: str = "replace"
    find: str = ""
    replacement: str = ""
    prefix: str = ""
    suffix: str = ""
    whitespace: bool = False
    special: bool = False
    start: int = 1
    end: int = 1

    def apply(self, name: str) -> str:
        if self.operation == "channel":
            name = extract_channel_name(name)
        elif self.operation == "replace" and self.find:
            name = name.replace(self.find, self.replacement)
        elif self.operation == "regex" and self.find:
            name = _regex_replace(self.find, self.replacement, name)
        elif self.operation == "range":
            if self.start < 1 or self.end < self.start:
                raise ValueError("문자 위치는 1부터 시작하며, 끝 위치는 시작 위치 이상이어야 합니다.")
            if self.start <= len(name):
                name = name[:self.start - 1] + self.replacement + name[self.end:]
        elif self.operation == "delete" and self.find:
            name = name.translate(str.maketrans("", "", self.find))
        elif self.operation == "before" and self.find:
            name = name.split(self.find, 1)[0]
        elif self.operation == "after" and self.find in name and self.find:
            name = name.split(self.find, 1)[1]
        if self.whitespace:
            name = re.sub(r"\s+", "", name)
        if self.special:
            name = re.sub(r"[^\w]", "", name)
        return (self.prefix + name + self.suffix).strip()


def validate_names(names: list[str]) -> None:
    for index, name in enumerate(names, 1):
        if not name.strip():
            raise ValueError(f"{index}번 포트 이름이 비어 있습니다.")
        if "!" in name or not name.isprintable():
            raise ValueError(f"{index}번 포트 이름에 줄바꿈·제어 문자 또는 !가 있습니다.")
