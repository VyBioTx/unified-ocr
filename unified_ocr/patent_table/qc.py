"""siRNA 专利表格 OCR 结果的质量控制（QC）过滤器。

论文（FENNEC, Methods → Data curation）定义的 QC 规则：
  - 列数据类型纯度 ≥ 95%
  - 序列长度 ≥ 16 nt
  - 化学修饰占比 > 20%
  - guide/passenger 编辑距离 ≤ 6
  - guide ↔ mRNA 反向互补编辑距离 ≤ 6
  - 浓度或修饰信息不清晰 → 剔除
  - 化学修饰 < 10 个 → 剔除
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# 常见核苷酸字符（大写，含修饰前缀如 m, f 等）
NUCL_PATTERN = re.compile(r"^[ACGUacgu]+$")

# 化学修饰标记（2'-OMe 等常见修饰前缀）
MOD_PREFIXES = {"m", "f", "r", "d", "inv", "invAb", "ch", "s", "ps", "ome", "2'-ome",
                "2'-f", "2'-deoxy", "thio", "amino", "biotin", "cy3", "cy5", "fam"}

# 常见修饰字符模式（大写字母前的修饰小写前缀）
MOD_PATTERN = re.compile(r"([a-z]+)([ACGU])")


@dataclass
class QCSpec:
    """QC 规格参数，默认使用论文中的阈值。"""

    min_sequence_length: int = 16
    min_modification_ratio: float = 0.20
    min_modification_count: int = 10
    max_edit_distance: int = 6
    min_column_purity: float = 0.95


@dataclass
class QCFilter:
    """siRNA 序列 QC 过滤器。

    对 OCR 抽取出的每一行 siRNA 数据进行质量检查，
    过滤掉不符合论文标准的条目。
    """

    spec: QCSpec = field(default_factory=QCSpec)

    @staticmethod
    def count_modifications(seq: str) -> int:
        """统计序列中的化学修饰数。

        修饰通常以小写字母/前缀形式出现，如:
          mAmC → 2 个修饰 (mA, mC)
          fUfC → 2 个修饰 (fU, fC)
          invAb → 1 个修饰
        """
        matches = MOD_PATTERN.findall(seq)
        return len(matches)

    @staticmethod
    def calculate_modification_ratio(seq: str) -> float:
        """计算序列中化学修饰的比例。"""
        mod_count = QCFilter.count_modifications(seq)
        if not seq:
            return 0.0
        return mod_count / len(seq)

    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        """计算两个序列间的编辑距离（Levenshtein 距离提升版）。"""
        if not s1 or not s2:
            return max(len(s1), len(s2))
        n, m = len(s1), len(s2)
        dp = list(range(n + 1))
        for j in range(1, m + 1):
            prev = dp[0]
            dp[0] = j
            for i in range(1, n + 1):
                temp = dp[i]
                cost = 0 if s1[i - 1].upper() == s2[j - 1].upper() else 1
                dp[i] = min(dp[i] + 1, dp[i - 1] + 1, prev + cost)
                prev = temp
        return dp[n]

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """计算反向互补序列（仅标准碱基）。"""
        complement = {"A": "U", "U": "A", "C": "G", "G": "C",
                      "a": "u", "u": "a", "c": "g", "g": "c"}
        return "".join(complement.get(base, base) for base in reversed(seq))

    def check_sequence(
        self,
        seq: str,
        guide: str | None = None,
        passenger: str | None = None,
        mrna: str | None = None,
    ) -> dict[str, Any]:
        """对一条 siRNA 序列进行全部 QC 检查。

        Args:
            seq: 完整 siRNA 序列（含修饰标记）。
            guide: guide 链序列（可选，用于编辑距离检查）。
            passenger: passenger 链序列（可选，用于编辑距离检查）。
            mrna: 靶标 mRNA 序列（可选，用于反向互补检查）。

        Returns:
            {
                "passed": bool,
                "reasons": list[str],
                "mod_count": int,
                "mod_ratio": float,
                "length": int,
                "guide_passenger_ed": int | None,
                "guide_mrna_rc_ed": int | None,
            }
        """
        reasons: list[str] = []
        seq_stripped = seq.strip()

        mod_count = self.count_modifications(seq_stripped)
        mod_ratio = self.calculate_modification_ratio(seq_stripped)
        length = len(seq_stripped)

        result: dict[str, Any] = {
            "passed": True,
            "reasons": [],
            "mod_count": mod_count,
            "mod_ratio": mod_ratio,
            "length": length,
            "guide_passenger_ed": None,
            "guide_mrna_rc_ed": None,
        }

        if length < self.spec.min_sequence_length:
            reasons.append(f"序列长度 {length} < {self.spec.min_sequence_length}")
            result["passed"] = False

        if mod_ratio < self.spec.min_modification_ratio:
            reasons.append(
                f"修饰比例 {mod_ratio:.1%} < {self.spec.min_modification_ratio:.0%}"
            )
            result["passed"] = False

        if mod_count < self.spec.min_modification_count and length >= 16:
            reasons.append(
                f"修饰数 {mod_count} < {self.spec.min_modification_count}"
            )
            result["passed"] = False

        if guide and passenger:
            ed = self.edit_distance(guide, passenger)
            result["guide_passenger_ed"] = ed
            if ed > self.spec.max_edit_distance:
                reasons.append(
                    f"guide/passenger 编辑距离 {ed} > {self.spec.max_edit_distance}"
                )
                result["passed"] = False

        if guide and mrna:
            rc = self.reverse_complement(guide)
            ed = self.edit_distance(rc, mrna)
            result["guide_mrna_rc_ed"] = ed
            if ed > self.spec.max_edit_distance:
                reasons.append(
                    f"guide↔mRNA 反向互补编辑距离 {ed} > {self.spec.max_edit_distance}"
                )
                result["passed"] = False

        result["reasons"] = reasons
        return result

    def filter_rows(
        self,
        rows: list[dict[str, str]],
        seq_column: str = "sequence",
        guide_column: str | None = None,
        passenger_column: str | None = None,
        mrna_column: str | None = None,
    ) -> list[dict[str, Any]]:
        """批量过滤 CSV/表格行。

        Args:
            rows: 字典列表，每行一个 siRNA 条目。
            seq_column: 完整序列的列名。
            guide_column: guide 链列名（可选）。
            passenger_column: passenger 链列名（可选）。
            mrna_column: 靶标 mRNA 序列列名（可选）。

        Returns:
            过滤后的行列表，每行附加 QC 结果。
        """
        filtered: list[dict[str, Any]] = []
        for row in rows:
            seq = row.get(seq_column, "")
            guide = row.get(guide_column) if guide_column else None
            passenger = row.get(passenger_column) if passenger_column else None
            mrna = row.get(mrna_column) if mrna_column else None

            qc = self.check_sequence(seq, guide, passenger, mrna)
            row["_qc"] = qc
            if qc["passed"]:
                filtered.append(row)
            else:
                log.debug("行被 QC 过滤: %s — %s", seq[:30], qc["reasons"])

        log.info("QC 过滤: %d/%d 行通过", len(filtered), len(rows))
        return filtered


def default_qc_filter() -> QCFilter:
    """使用论文默认阈值的 QC 过滤器。"""
    return QCFilter(spec=QCSpec())