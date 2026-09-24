"""Structured results for customisation checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

Severity = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    """One atomic check."""

    name: str
    status: Severity
    message: str
    details: str = ""
    hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "fail"


@dataclass
class ValidationReport:
    """Collection of checks with a printable summary."""

    title: str
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def extend(self, checks: Iterable[CheckResult]) -> None:
        self.checks.extend(checks)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    def summary_str(self) -> str:
        lines = [f"=== {self.title} ==="]
        for c in self.checks:
            mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[c.status]
            lines.append(f"[{mark}] {c.name}: {c.message}")
            if c.details:
                for detail in str(c.details).splitlines():
                    lines.append(f"         {detail}")
            if c.hint and c.status != "ok":
                for hint in str(c.hint).splitlines():
                    lines.append(f"         → {hint}")
        status = "PASSED" if self.ok else "FAILED"
        lines.append(
            f"--- {status}  ({len(self.checks)} checks, "
            f"{self.n_fail} fail, {self.n_warn} warn) ---"
        )
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValidationError(self)


class ValidationError(Exception):
    """Raised when a validation report contains failures."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(report.summary_str())
