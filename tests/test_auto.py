"""Tests for the auto_instrument() one-line instrumentation system."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest import mock

import pytest

from agentsight._state import reset as reset_state
from agentsight.auto import (
    _REGISTRY,
    _framework_available,
    auto_instrument,
    available_frameworks,
    uninstrument,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global state before and after each test."""
    reset_state()
    yield
    reset_state()


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------


class TestFrameworkDetection:
    def test_available_frameworks_returns_list(self):
        result = available_frameworks()
        assert isinstance(result, list)

    def test_framework_available_returns_false_for_missing(self):
        assert _framework_available("nonexistent_package_xyz") is False

    def test_framework_available_returns_true_for_present(self):
        # pytest is always installed in the test environment
        assert _framework_available("pytest") is True

    def test_registry_has_expected_frameworks(self):
        expected = {
            "langchain",
            "langgraph",
            "openai_agents",
            "anthropic",
            "crewai",
            "autogen",
            "llamaindex",
            "semantic_kernel",
            "google_adk",
            "bedrock",
            "haystack",
            "smolagents",
            "pydantic_ai",
            "phidata",
        }
        assert set(_REGISTRY.keys()) == expected

    def test_registry_entries_have_correct_shape(self):
        for name, entry in _REGISTRY.items():
            assert isinstance(entry, tuple), f"{name}: expected tuple"
            assert len(entry) == 2, f"{name}: expected 2 elements"
            import_probe, patch_module = entry
            assert isinstance(import_probe, str)
            assert patch_module.startswith("agentsight._patch.")


# ---------------------------------------------------------------------------
# auto_instrument() — no frameworks installed
# ---------------------------------------------------------------------------


class TestAutoInstrumentBasics:
    def test_returns_dict(self):
        result = auto_instrument()
        assert isinstance(result, dict)

    def test_all_values_bool(self):
        result = auto_instrument()
        for name, instrumented in result.items():
            assert isinstance(instrumented, bool), f"{name} should be bool"

    def test_selective_unknown_framework_ignored(self):
        result = auto_instrument(frameworks=["nonexistent_framework"])
        assert result == {}

    def test_selective_with_uninstalled_framework(self):
        # Use a framework guaranteed to not be installed
        result = auto_instrument(frameworks=["google_adk"])
        assert result == {"google_adk": False}

    def test_unavailable_framework_returns_false(self):
        """When a framework's import probe fails, result is False."""
        with mock.patch(
            "agentsight.auto._framework_available", return_value=False
        ):
            result = auto_instrument(frameworks=["langchain"])
        assert result == {"langchain": False}

    def test_idempotent(self):
        result1 = auto_instrument()
        result2 = auto_instrument()
        assert result1 == result2


# ---------------------------------------------------------------------------
# auto_instrument() — with mocked frameworks
# ---------------------------------------------------------------------------


class TestAutoInstrumentWithMocks:
    def test_instruments_detected_framework(self):
        """Mock a framework as importable and verify its patch module is called."""
        fake_patch = mock.MagicMock()
        fake_patch.install = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"langchain_core": mock.MagicMock()}):
            with mock.patch(
                "agentsight.auto.importlib.import_module"
            ) as mock_import:
                # Return a real module for the probe, and our fake for the patch
                def side_effect(name: str) -> Any:
                    if name == "langchain_core":
                        return mock.MagicMock()
                    if name == "agentsight._patch.langchain":
                        return fake_patch
                    return importlib.import_module(name)

                mock_import.side_effect = side_effect

                result = auto_instrument(frameworks=["langchain"])

        assert result["langchain"] is True
        fake_patch.install.assert_called_once()

    def test_skips_already_instrumented(self):
        """Once a framework is marked instrumented, it should not be patched again."""
        fake_patch = mock.MagicMock()
        fake_patch.install = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"langchain_core": mock.MagicMock()}):
            with mock.patch(
                "agentsight.auto.importlib.import_module"
            ) as mock_import:
                def side_effect(name: str) -> Any:
                    if name == "langchain_core":
                        return mock.MagicMock()
                    if name == "agentsight._patch.langchain":
                        return fake_patch
                    return importlib.import_module(name)

                mock_import.side_effect = side_effect

                result1 = auto_instrument(frameworks=["langchain"])
                result2 = auto_instrument(frameworks=["langchain"])

        assert result1["langchain"] is True
        assert result2["langchain"] is True
        # install() should only be called once
        fake_patch.install.assert_called_once()

    def test_handles_patch_install_failure(self):
        """If a patch module's install() raises, it should not crash."""
        fake_patch = mock.MagicMock()
        fake_patch.install.side_effect = RuntimeError("patch failed")

        with mock.patch.dict(sys.modules, {"anthropic": mock.MagicMock()}):
            with mock.patch(
                "agentsight.auto.importlib.import_module"
            ) as mock_import:
                def side_effect(name: str) -> Any:
                    if name == "anthropic":
                        return mock.MagicMock()
                    if name == "agentsight._patch.anthropic":
                        return fake_patch
                    return importlib.import_module(name)

                mock_import.side_effect = side_effect

                result = auto_instrument(frameworks=["anthropic"])

        assert result["anthropic"] is False

    def test_multiple_frameworks(self):
        """Instrument multiple frameworks at once."""
        fake_lc = mock.MagicMock()
        fake_lc.install = mock.MagicMock()
        fake_anth = mock.MagicMock()
        fake_anth.install = mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {"langchain_core": mock.MagicMock(), "anthropic": mock.MagicMock()},
        ):
            with mock.patch(
                "agentsight.auto.importlib.import_module"
            ) as mock_import:
                def side_effect(name: str) -> Any:
                    mapping = {
                        "langchain_core": mock.MagicMock(),
                        "anthropic": mock.MagicMock(),
                        "agentsight._patch.langchain": fake_lc,
                        "agentsight._patch.anthropic": fake_anth,
                    }
                    if name in mapping:
                        return mapping[name]
                    return importlib.import_module(name)

                mock_import.side_effect = side_effect

                result = auto_instrument(frameworks=["langchain", "anthropic"])

        assert result["langchain"] is True
        assert result["anthropic"] is True
        fake_lc.install.assert_called_once()
        fake_anth.install.assert_called_once()


# ---------------------------------------------------------------------------
# uninstrument()
# ---------------------------------------------------------------------------


class TestUninstrument:
    def test_uninstrument_calls_uninstall(self):
        """uninstrument() should call each patch module's uninstall()."""
        fake_patch = mock.MagicMock()
        fake_patch.install = mock.MagicMock()
        fake_patch.uninstall = mock.MagicMock()

        with mock.patch.dict(sys.modules, {"crewai": mock.MagicMock()}):
            with mock.patch(
                "agentsight.auto.importlib.import_module"
            ) as mock_import:
                def side_effect(name: str) -> Any:
                    if name == "crewai":
                        return mock.MagicMock()
                    if name == "agentsight._patch.crewai":
                        return fake_patch
                    return importlib.import_module(name)

                mock_import.side_effect = side_effect

                auto_instrument(frameworks=["crewai"])
                uninstrument(frameworks=["crewai"])

        fake_patch.uninstall.assert_called_once()

    def test_uninstrument_noop_when_not_instrumented(self):
        """uninstrument() should do nothing if nothing was instrumented."""
        # Should not raise
        uninstrument()


# ---------------------------------------------------------------------------
# Per-framework instrument_*() convenience functions
# ---------------------------------------------------------------------------


class TestPerFrameworkFunctions:
    def test_instrument_functions_exist(self):
        from agentsight.auto import (
            instrument_anthropic,
            instrument_autogen,
            instrument_bedrock,
            instrument_crewai,
            instrument_google_adk,
            instrument_haystack,
            instrument_langchain,
            instrument_langgraph,
            instrument_llamaindex,
            instrument_openai_agents,
            instrument_phidata,
            instrument_pydantic_ai,
            instrument_semantic_kernel,
            instrument_smolagents,
        )

        # All should be callable
        funcs = [
            instrument_langchain,
            instrument_langgraph,
            instrument_openai_agents,
            instrument_anthropic,
            instrument_crewai,
            instrument_autogen,
            instrument_llamaindex,
            instrument_semantic_kernel,
            instrument_google_adk,
            instrument_bedrock,
            instrument_haystack,
            instrument_smolagents,
            instrument_pydantic_ai,
            instrument_phidata,
        ]
        for fn in funcs:
            assert callable(fn)

    def test_instrument_function_returns_false_when_not_installed(self):
        from agentsight.auto import instrument_google_adk

        # google.adk is not installed in the test environment
        result = instrument_google_adk()
        assert result is False


# ---------------------------------------------------------------------------
# Exports from __init__.py
# ---------------------------------------------------------------------------


class TestExports:
    def test_auto_instrument_importable_from_top_level(self):
        from agentsight import auto_instrument as ai
        assert callable(ai)

    def test_uninstrument_importable_from_top_level(self):
        from agentsight import uninstrument as ui
        assert callable(ui)

    def test_available_frameworks_importable_from_top_level(self):
        from agentsight import available_frameworks as af
        assert callable(af)

    def test_per_framework_importable_from_top_level(self):
        from agentsight import (
            instrument_anthropic,
            instrument_langchain,
        )
        assert callable(instrument_anthropic)
        assert callable(instrument_langchain)


# ---------------------------------------------------------------------------
# _state.py integration
# ---------------------------------------------------------------------------


class TestStateIntegration:
    def test_initialize_creates_observer(self):
        from agentsight._state import get_observer, initialize

        obs = initialize()
        assert obs is not None
        assert get_observer() is obs

    def test_initialize_is_idempotent(self):
        from agentsight._state import initialize

        obs1 = initialize()
        obs2 = initialize()
        assert obs1 is obs2

    def test_mark_and_check_instrumented(self):
        from agentsight._state import (
            instrumented_frameworks,
            is_instrumented,
            mark_instrumented,
        )

        assert not is_instrumented("test_fw")
        mark_instrumented("test_fw")
        assert is_instrumented("test_fw")
        assert "test_fw" in instrumented_frameworks()

    def test_reset_clears_everything(self):
        from agentsight._state import (
            get_observer,
            initialize,
            is_instrumented,
            mark_instrumented,
            reset,
        )

        initialize()
        mark_instrumented("test_fw")
        reset()

        assert get_observer() is None
        assert not is_instrumented("test_fw")
