"""Small, literal transformations for channel names."""

import re
from dataclasses import dataclass


def natural_key(value: str) -> list:
    return [int(part) if part.isdecimal() else part.casefold()
            for part in re.split(r"(\d+)", value)]


@dataclass
class NameRule:
    operation: str = "replace"
    find: str = ""
    replacement: str = ""
    prefix: str = ""
    suffix: str = ""
    whitespace: bool = False
    special: bool = False

    def apply(self, name: str) -> str:
        if self.operation == "channel":
            if "::" in name:
                name = name.rsplit("::", 1)[1]
            else:
                parts = name.split(maxsplit=1)
                if len(parts) == 2 and re.fullmatch(r"(?:\S+_[0-9]+-\S+|[A-Za-z]+[0-9]+-[A-Za-z]*[0-9]+)", parts[0]):
                    name = parts[1]
        elif self.operation == "replace" and self.find:
            name = name.replace(self.find, self.replacement)
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
