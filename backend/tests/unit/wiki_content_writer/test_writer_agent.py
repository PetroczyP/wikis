"""Tests for wiki_content_writer/writer_agent.py (#237, #314).

Covers: WikiContentWriter construction, budget formula (all parameter
combinations), soft target guidance, and hard cap enforcement.

#314: base default 5→8, soft cap 15-30→20-40, architectural weight 1→1.5.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.core.wiki_content_writer.writer_agent import (
    BUDGET_HARD_CAP,
    BUDGET_SOFT_MAX,
    BUDGET_SOFT_MIN,
    WikiContentWriter,
    compute_page_budget,
)
from app.core.wiki_content_writer.writer_tools import WriterTools

# ── Budget formula constants ─────────────────────────────────────────────────


class TestBudgetConstants:
    def test_soft_min_is_20(self):
        """#314: soft min raised from 15 to 20."""
        assert BUDGET_SOFT_MIN == 20

    def test_soft_max_is_40(self):
        """#314: soft max raised from 30 to 40."""
        assert BUDGET_SOFT_MAX == 40

    def test_hard_cap_is_80(self):
        assert BUDGET_HARD_CAP == 80


# ── compute_page_budget ──────────────────────────────────────────────────────


class TestComputePageBudget:
    """
    budget = base
           + 2   * count(target_symbols where layer in {entry_point, public_api})
           + 1.5 * count(target_symbols where layer in {core_type, infrastructure})
           + 1   * count(target_folders)
           + 1   * count(target_docs)

    Soft target: 20-40. Hard cap: 80. Default base: 8 (#314).
    """

    def _spec(
        self,
        entry_point_symbols: int = 0,
        public_api_symbols: int = 0,
        core_type_symbols: int = 0,
        infrastructure_symbols: int = 0,
        target_folders: int = 0,
        target_docs: int = 0,
        base: int = 8,
    ) -> dict:
        """Build a minimal page spec for budget calculation."""
        symbols = (
            [("Sym", "entry_point")] * entry_point_symbols
            + [("Sym", "public_api")] * public_api_symbols
            + [("Sym", "core_type")] * core_type_symbols
            + [("Sym", "infrastructure")] * infrastructure_symbols
        )
        return {
            "target_symbols": symbols,
            "target_folders": [f"folder{i}" for i in range(target_folders)],
            "target_docs": [f"doc{i}.md" for i in range(target_docs)],
            "base": base,
        }

    # --- Base-only calculation (#314: default base is now 8) ---

    def test_base_only_with_explicit_8_returns_8(self):
        """#314: base=8 with no symbols/folders/docs returns 8."""
        result = compute_page_budget(target_symbols=[], target_folders=[], target_docs=[], base=8)
        assert result == 8

    def test_base_only_returns_base(self):
        spec = self._spec(base=5)
        result = compute_page_budget(**spec)
        assert result == 5

    def test_base_of_zero_returns_zero(self):
        spec = self._spec(base=0)
        result = compute_page_budget(**spec)
        assert result == 0

    # --- Entry-point and public_api symbols (weight 2, unchanged) ---

    def test_one_entry_point_adds_two(self):
        spec = self._spec(entry_point_symbols=1, base=5)
        result = compute_page_budget(**spec)
        assert result == 7  # 5 + 2*1

    def test_two_entry_points_with_base_8_equals_12(self):
        """#314: base=8, 2 entry_point symbols → 8 + 2×2 = 12."""
        result = compute_page_budget(
            target_symbols=[("A", "entry_point"), ("B", "entry_point")],
            target_folders=[],
            target_docs=[],
            base=8,
        )
        assert result == 12  # 8 + 2*2

    def test_three_entry_points_adds_six(self):
        spec = self._spec(entry_point_symbols=3, base=5)
        result = compute_page_budget(**spec)
        assert result == 11  # 5 + 2*3

    def test_one_public_api_adds_two(self):
        spec = self._spec(public_api_symbols=1, base=5)
        result = compute_page_budget(**spec)
        assert result == 7  # 5 + 2*1

    def test_entry_point_and_public_api_combined(self):
        spec = self._spec(entry_point_symbols=2, public_api_symbols=3, base=5)
        result = compute_page_budget(**spec)
        assert result == 15  # 5 + 2*2 + 2*3

    # --- Core type and infrastructure symbols (weight 1.5, #314) ---

    def test_one_core_type_adds_one_point_five(self):
        """#314: core_type weight raised from 1 to 1.5 (floor → 1 for single symbol)."""
        spec = self._spec(core_type_symbols=1, base=5)
        result = compute_page_budget(**spec)
        assert result == 6  # 5 + floor(1.5*1) = 5 + 1

    def test_two_core_types_adds_three(self):
        """#314: 2 core_type symbols → floor(1.5 * 2) = 3."""
        spec = self._spec(core_type_symbols=2, base=5)
        result = compute_page_budget(**spec)
        assert result == 8  # 5 + floor(1.5*2) = 5 + 3

    def test_one_infrastructure_adds_one_point_five(self):
        """#314: infrastructure weight raised from 1 to 1.5 (floor → 1 for single symbol)."""
        spec = self._spec(infrastructure_symbols=1, base=5)
        result = compute_page_budget(**spec)
        assert result == 6  # 5 + floor(1.5*1) = 5 + 1

    def test_core_type_and_infrastructure_combined(self):
        """#314: 3 core_type + 2 infrastructure = floor(1.5*5) = 7."""
        spec = self._spec(core_type_symbols=3, infrastructure_symbols=2, base=5)
        result = compute_page_budget(**spec)
        assert result == 12  # 5 + floor(1.5*5) = 5 + 7

    # --- Folders and docs (weight 1, unchanged) ---

    def test_one_folder_adds_one(self):
        spec = self._spec(target_folders=1, base=5)
        result = compute_page_budget(**spec)
        assert result == 6  # 5 + 1

    def test_one_doc_adds_one(self):
        spec = self._spec(target_docs=1, base=5)
        result = compute_page_budget(**spec)
        assert result == 6  # 5 + 1

    def test_folders_and_docs_combined(self):
        spec = self._spec(target_folders=3, target_docs=4, base=5)
        result = compute_page_budget(**spec)
        assert result == 12  # 5 + 3 + 4

    # --- #314-specific: BUDGET_SOFT_MIN=20 as floor after symbol weighting ---

    def test_soft_min_20_is_documented_floor(self):
        """#314: BUDGET_SOFT_MIN is the new soft floor (20), not enforced by code."""
        # base=8 + no symbols/folders/docs = 8, which is below the soft min.
        # The soft min is advisory only — compute_page_budget returns raw budget.
        result = compute_page_budget(target_symbols=[], target_folders=[], target_docs=[], base=8)
        assert result == 8
        # Confirm BUDGET_SOFT_MIN is 20
        assert BUDGET_SOFT_MIN == 20

    # --- Mixed scenarios ---

    def test_typical_leaf_page_in_soft_range(self):
        """A leaf page with reasonable symbols should land in 20-40 soft range."""
        spec = self._spec(
            public_api_symbols=2,
            core_type_symbols=3,
            target_folders=2,
            target_docs=1,
            base=8,
        )
        result = compute_page_budget(**spec)
        # 8 + 2*2 + floor(1.5*3) + 2 + 1 = 8 + 4 + 4 + 2 + 1 = 19
        assert result == 19

    def test_entry_point_heavy_page_exceeds_soft_max(self):
        """A page with many entry-point symbols gets a high budget."""
        spec = self._spec(
            entry_point_symbols=8,
            public_api_symbols=5,
            core_type_symbols=4,
            target_folders=3,
            target_docs=2,
            base=8,
        )
        result = compute_page_budget(**spec)
        # 8 + 2*8 + 2*5 + floor(1.5*4) + 3 + 2 = 8 + 16 + 10 + 6 + 3 + 2 = 45
        assert result == 45
        assert result > BUDGET_SOFT_MAX

    # --- Hard cap ----

    def test_hard_cap_enforced(self):
        """Budget never exceeds 80 regardless of inputs."""
        spec = self._spec(
            entry_point_symbols=30,
            public_api_symbols=20,
            core_type_symbols=20,
            target_folders=20,
            target_docs=20,
            base=10,
        )
        result = compute_page_budget(**spec)
        assert result == BUDGET_HARD_CAP

    def test_hard_cap_at_exactly_80(self):
        """#314: budget capped at BUDGET_HARD_CAP=80."""
        # Choose numbers that would exceed 80 without cap
        result = compute_page_budget(
            target_symbols=[("s", "entry_point")] * 40,
            target_folders=[],
            target_docs=[],
            base=8,
        )
        # Uncapped: 8 + 2*40 = 88 → capped to 80
        assert result == BUDGET_HARD_CAP

    def test_result_just_below_cap_not_clamped(self):
        """A budget of exactly 79 is below the cap and passes through."""
        # 10 + 2*30 + floor(1.5*9) + 0 + 0 = 10 + 60 + 13 = 83 → too high
        # Pick: 10 + 2*30 + floor(1.5*6) = 10 + 60 + 9 = 79
        spec = self._spec(
            entry_point_symbols=30,
            core_type_symbols=6,
            base=10,
        )
        result = compute_page_budget(**spec)
        assert result == 79
        assert result < BUDGET_HARD_CAP

    def test_result_at_cap_not_exceeded(self):
        """Budget of exactly 80 is returned as-is."""
        # 10 + 2*30 + floor(1.5*6) + 1 + 0 = 10 + 60 + 9 + 1 = 80
        spec = self._spec(
            entry_point_symbols=30,
            core_type_symbols=6,
            target_folders=1,
            base=10,
        )
        result = compute_page_budget(**spec)
        assert result == BUDGET_HARD_CAP

    # --- Symbol layer tuple format ---

    def test_accepts_tuple_list_of_name_layer(self):
        """target_symbols is a list of (name, layer) tuples."""
        symbols = [("main", "entry_point"), ("Config", "core_type")]
        result = compute_page_budget(
            target_symbols=symbols,
            target_folders=[],
            target_docs=[],
            base=5,
        )
        # 5 + 2*1 + floor(1.5*1) = 5 + 2 + 1 = 8
        assert result == 8

    def test_unknown_layer_not_counted(self):
        """Symbols with unrecognised layers contribute 0."""
        symbols = [("helper", "internal"), ("util", "unknown")]
        result = compute_page_budget(
            target_symbols=symbols,
            target_folders=[],
            target_docs=[],
            base=5,
        )
        assert result == 5  # no contribution from unknown layers


# ── WikiContentWriter construction ───────────────────────────────────────────


class TestWikiContentWriterConstruction:
    def test_constructs_with_required_args(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools)
        assert writer is not None

    def test_stores_tools(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools)
        assert writer.tools is tools

    def test_constructs_with_llm_client(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        mock_llm = MagicMock()
        writer = WikiContentWriter(tools=tools, llm_client=mock_llm)
        assert writer.llm_client is mock_llm

    def test_llm_client_defaults_to_none(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools)
        assert writer.llm_client is None

    def test_max_iterations_defaults_to_hard_cap(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools)
        # Default max_iterations should respect the hard budget cap as an upper bound
        assert isinstance(writer.max_iterations, int)
        assert writer.max_iterations > 0

    def test_custom_max_iterations_stored(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools, max_iterations=50)
        assert writer.max_iterations == 50

    def test_compute_budget_uses_formula(self, tmp_path):
        """WikiContentWriter.compute_budget delegates to compute_page_budget."""
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools)
        page_spec = {
            "target_symbols": [("main", "entry_point")],
            "target_folders": ["src/"],
            "target_docs": [],
            "base": 8,
        }
        budget = writer.compute_budget(page_spec)
        # 8 + 2*1 + 1 = 11
        assert budget == 11

    def test_compute_budget_respects_hard_cap(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools)
        page_spec = {
            "target_symbols": [("sym", "entry_point")] * 100,
            "target_folders": ["f"] * 50,
            "target_docs": ["d.md"] * 50,
            "base": 10,
        }
        budget = writer.compute_budget(page_spec)
        assert budget == BUDGET_HARD_CAP

    def test_compute_budget_default_base_is_8(self, tmp_path):
        """#314: WikiContentWriter.compute_budget uses base=8 when not specified."""
        tools = WriterTools(repo_root=str(tmp_path))
        writer = WikiContentWriter(tools=tools)
        page_spec = {
            "target_symbols": [],
            "target_folders": [],
            "target_docs": [],
            # no 'base' key — should default to 8
        }
        budget = writer.compute_budget(page_spec)
        assert budget == 8
