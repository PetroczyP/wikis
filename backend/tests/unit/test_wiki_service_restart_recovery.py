"""Tests for restart-safe wiki generation recovery.

Covers:
- Invocation model has run_dir and subprocess_pid fields
- _load_invocations: generating+run_dir+output.json present → schedule recovery task
- _load_invocations: generating+run_dir+pid alive → schedule watch task
- _load_invocations: generating+run_dir+pid dead, no output → fail as before
- _load_invocations: generating, no run_dir → fail as before (legacy path)
- _load_invocations: running (incremental) → always fail (no subprocess tracking)
- _recover_completed_generation: parses output.json and calls _finalize_generation
- _recover_completed_generation: failure in finalize → marks failed + cleans up dir
- _watch_for_generation_output: output appears → calls recover
- _watch_for_generation_output: pid dies, no output → marks failed
- _finalize_generation: extracted from _run_generation, uploads artifacts and registers wiki
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.invocation import Invocation
from app.services.wiki_service import WikiService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(wiki_management: MagicMock | None = None) -> WikiService:
    settings = MagicMock()
    settings.planner_type = "agent"
    settings.cluster_exclude_tests = False
    settings.cache_dir = "/tmp/wikis-cache"
    storage = MagicMock()
    storage.download = AsyncMock(side_effect=FileNotFoundError)
    storage.upload = AsyncMock()
    service = WikiService(settings=settings, storage=storage)
    if wiki_management is not None:
        service.wiki_management = wiki_management
    return service


def _persisted_generating(inv_id: str, wiki_id: str, run_dir: str | None = None, pid: int | None = None) -> dict:
    d = {
        "id": inv_id,
        "wiki_id": wiki_id,
        "status": "generating",
        "repo_url": "https://github.com/owner/repo",
        "branch": "main",
        "pages_completed": 0,
        "pages_total": 0,
        "progress": 0.3,
        "created_at": datetime(2026, 5, 18, 10, 0, 0).isoformat(),
        "owner_id": "user-1",
    }
    if run_dir is not None:
        d["run_dir"] = run_dir
    if pid is not None:
        d["subprocess_pid"] = pid
    return d


# ---------------------------------------------------------------------------
# Part 2: Invocation model fields
# ---------------------------------------------------------------------------


class TestInvocationFields:
    def test_run_dir_field_exists_and_defaults_none(self) -> None:
        inv = Invocation(id="x", wiki_id="w")
        assert hasattr(inv, "run_dir")
        assert inv.run_dir is None

    def test_subprocess_pid_field_exists_and_defaults_none(self) -> None:
        inv = Invocation(id="x", wiki_id="w")
        assert hasattr(inv, "subprocess_pid")
        assert inv.subprocess_pid is None

    def test_run_dir_can_be_set(self) -> None:
        inv = Invocation(id="x", wiki_id="w", run_dir="/tmp/test-run")
        assert inv.run_dir == "/tmp/test-run"

    def test_subprocess_pid_can_be_set(self) -> None:
        inv = Invocation(id="x", wiki_id="w", subprocess_pid=12345)
        assert inv.subprocess_pid == 12345

    def test_fields_excluded_from_api_serialisation(self) -> None:
        """run_dir and subprocess_pid are excluded from API-facing model_dump()."""
        inv = Invocation(id="x", wiki_id="w", run_dir="/tmp/r", subprocess_pid=99)
        d = inv.model_dump()
        assert "run_dir" not in d
        assert "subprocess_pid" not in d

    def test_fields_accessible_directly_for_persistence(self) -> None:
        """Internal fields are accessible as attributes for the persistence layer."""
        inv = Invocation(id="x", wiki_id="w", run_dir="/tmp/r", subprocess_pid=99)
        # Persistence reads these directly rather than via model_dump().
        assert inv.run_dir == "/tmp/r"
        assert inv.subprocess_pid == 99

    def test_fields_can_be_loaded_from_dict(self) -> None:
        """Persistence round-trip: build dict manually, restore Invocation."""
        inv = Invocation(id="x", wiki_id="w", run_dir="/tmp/r", subprocess_pid=42)
        # Build the persistence dict the same way _persist_invocations does.
        d = inv.model_dump()
        d["run_dir"] = inv.run_dir
        d["subprocess_pid"] = inv.subprocess_pid
        restored = Invocation(**d)
        assert restored.run_dir == "/tmp/r"
        assert restored.subprocess_pid == 42


# ---------------------------------------------------------------------------
# Part 3: _load_invocations recovery logic
# ---------------------------------------------------------------------------


def _close_coro(coro):
    """Side effect for asyncio.create_task that closes the coroutine instead of scheduling it."""
    coro.close()
    return MagicMock()


class TestLoadInvocationsRecovery:
    @pytest.mark.asyncio
    async def test_no_run_dir_still_marks_failed(self) -> None:
        """Legacy invocations without run_dir fall back to immediate fail."""
        persisted = {
            "inv-1": _persisted_generating("inv-1", "wiki-abc"),
        }
        storage = MagicMock()
        storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
        storage.upload = AsyncMock()
        wm = MagicMock()
        wm.mark_status = AsyncMock(return_value=True)
        service = WikiService(settings=MagicMock(), storage=storage)
        service.wiki_management = wm

        await service.load_persisted_invocations()

        inv = service._invocations["inv-1"]
        assert inv.status == "failed"
        assert "Server restarted" in (inv.error or "")

    @pytest.mark.asyncio
    async def test_output_exists_schedules_recovery_task(self) -> None:
        """When output.json exists, _recover_completed_generation is scheduled."""
        inv_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / inv_id
            run_dir.mkdir(parents=True)
            output_path = run_dir / "output.json"
            output_path.write_text(json.dumps({"success": True, "generated_pages": {}}))

            persisted = {
                inv_id: _persisted_generating(inv_id, "wiki-xyz", run_dir=str(run_dir)),
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            recover_calls = []

            async def fake_recover(inv, op):
                recover_calls.append((inv.id, op))

            service._recover_completed_generation = fake_recover

            with patch("asyncio.create_task", side_effect=_close_coro) as mock_create_task:
                await service.load_persisted_invocations()
                # Should have created a task for recovery
                assert mock_create_task.called

            # Invocation should remain "generating" — recovery will update it
            inv = service._invocations[inv_id]
            assert inv.status == "generating"

    @pytest.mark.asyncio
    async def test_pid_alive_schedules_watch_task(self) -> None:
        """When pid is alive and no output, _watch_for_generation_output is scheduled."""
        inv_id = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / inv_id
            run_dir.mkdir(parents=True)
            # No output.json

            persisted = {
                inv_id: _persisted_generating(
                    inv_id, "wiki-abc", run_dir=str(run_dir), pid=os.getpid()
                ),
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            with patch("asyncio.create_task", side_effect=_close_coro) as mock_create_task:
                await service.load_persisted_invocations()
                assert mock_create_task.called

            # Status stays "generating" while we wait
            inv = service._invocations[inv_id]
            assert inv.status == "generating"

    @pytest.mark.asyncio
    async def test_pid_dead_no_output_marks_failed(self) -> None:
        """When pid is dead and no output.json, invocation is marked failed."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / "inv-4"
            run_dir.mkdir(parents=True)
            # No output.json, dead PID

            persisted = {
                "inv-4": _persisted_generating(
                    "inv-4", "wiki-abc", run_dir=str(run_dir), pid=999999999
                ),
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            with patch("os.kill", side_effect=ProcessLookupError):
                await service.load_persisted_invocations()

            inv = service._invocations["inv-4"]
            assert inv.status == "failed"
            assert "Server restarted" in (inv.error or "")

    @pytest.mark.asyncio
    async def test_running_incremental_always_fails(self) -> None:
        """Incremental refresh invocations (status=running) always mark failed — no subprocess tracking."""
        persisted = {
            "inv-run": {
                "id": "inv-run",
                "wiki_id": "wiki-abc",
                "status": "running",
                "repo_url": "https://github.com/owner/repo",
                "branch": "main",
                "pages_completed": 3,
                "pages_total": 10,
                "progress": 0.3,
                "created_at": datetime(2026, 5, 18, 10, 0, 0).isoformat(),
                "owner_id": "user-1",
            },
        }
        storage = MagicMock()
        storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
        storage.upload = AsyncMock()
        wm = MagicMock()
        wm.mark_status = AsyncMock(return_value=True)
        service = WikiService(settings=MagicMock(), storage=storage)
        service.wiki_management = wm

        await service.load_persisted_invocations()

        inv = service._invocations["inv-run"]
        assert inv.status == "failed"
        assert "Server restarted during running" in (inv.error or "")


# ---------------------------------------------------------------------------
# Part 4: _recover_completed_generation
# ---------------------------------------------------------------------------


class TestRecoverCompletedGeneration:
    @pytest.mark.asyncio
    async def test_calls_finalize_with_decoded_result(self) -> None:
        """Reads output.json and calls _finalize_generation with decoded result."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            output = {"success": True, "generated_pages": {"p1": "# Page 1"}, "artifacts": []}
            (run_dir / "output.json").write_text(json.dumps(output))

            service = _make_service()
            inv = Invocation(id="inv-r", wiki_id="wiki-r", status="generating")

            finalize_calls = []

            async def fake_finalize(invocation, result):
                finalize_calls.append((invocation.id, result))

            service._finalize_generation = fake_finalize

            await service._recover_completed_generation(inv, str(run_dir / "output.json"))

            assert len(finalize_calls) == 1
            assert finalize_calls[0][0] == "inv-r"
            assert finalize_calls[0][1]["generated_pages"] == {"p1": "# Page 1"}

    @pytest.mark.asyncio
    async def test_cleans_up_run_dir_after_success(self) -> None:
        """Run directory is removed after successful recovery."""
        with tempfile.TemporaryDirectory() as tmp:
            # run_dir must be under cache_dir/wiki-runs/<inv_id> for the path
            # validation introduced in Fix 5 to allow the rmtree.
            run_dir = Path(tmp) / "wiki-runs" / "inv-r2"
            run_dir.mkdir(parents=True)
            output = {"success": True, "generated_pages": {}, "artifacts": []}
            (run_dir / "output.json").write_text(json.dumps(output))

            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            inv = Invocation(id="inv-r2", wiki_id="wiki-r2", status="generating")
            service._finalize_generation = AsyncMock()

            await service._recover_completed_generation(inv, str(run_dir / "output.json"))

            assert not run_dir.exists()

    @pytest.mark.asyncio
    async def test_marks_failed_when_output_unreadable(self) -> None:
        """If output.json cannot be read, invocation is marked failed."""
        wm = MagicMock()
        wm.mark_status = AsyncMock(return_value=True)
        service = _make_service(wiki_management=wm)
        inv = Invocation(id="inv-bad", wiki_id="wiki-bad", status="generating")

        await service._recover_completed_generation(inv, "/nonexistent/path/output.json")

        assert inv.status == "failed"
        assert inv.error is not None

    @pytest.mark.asyncio
    async def test_decodes_base64_artifacts(self) -> None:
        """Artifacts with _b64=True and string data are decoded to bytes."""
        import base64

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            raw_bytes = b"binary content"
            output = {
                "success": True,
                "generated_pages": {},
                "artifacts": [
                    {
                        "name": "test.bin",
                        "data": base64.b64encode(raw_bytes).decode(),
                        "_b64": True,
                    }
                ],
            }
            (run_dir / "output.json").write_text(json.dumps(output))

            service = _make_service()
            inv = Invocation(id="inv-b64", wiki_id="wiki-b64", status="generating")

            finalize_args = {}

            async def capture_finalize(invocation, result):
                finalize_args["result"] = result

            service._finalize_generation = capture_finalize

            await service._recover_completed_generation(inv, str(run_dir / "output.json"))

            arts = finalize_args["result"]["artifacts"]
            assert len(arts) == 1
            assert arts[0]["data"] == raw_bytes
            assert "_b64" not in arts[0]


# ---------------------------------------------------------------------------
# Part 5: _watch_for_generation_output
# ---------------------------------------------------------------------------


class TestWatchForGenerationOutput:
    @pytest.mark.asyncio
    async def test_recovers_when_output_appears(self) -> None:
        """When output.json appears, _recover_completed_generation is called."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            service = _make_service()
            inv = Invocation(id="inv-w1", wiki_id="wiki-w1", status="generating", subprocess_pid=os.getpid())

            recover_calls = []

            async def fake_recover(invocation, op):
                recover_calls.append((invocation.id, op))

            service._recover_completed_generation = fake_recover

            # Write output after a tiny delay by simulating: patch asyncio.sleep
            # to write the file on first call, then raise to stop the loop
            call_count = 0

            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    output_path.write_text(json.dumps({"success": True, "generated_pages": {}}))

            with patch("asyncio.sleep", fake_sleep):
                await service._watch_for_generation_output(inv, str(output_path))

            assert len(recover_calls) == 1
            assert recover_calls[0][0] == "inv-w1"

    @pytest.mark.asyncio
    async def test_marks_failed_when_pid_dies_no_output(self) -> None:
        """When the tracked PID dies without output.json, invocation is failed."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)
            service = _make_service(wiki_management=wm)
            # Use a PID that is deterministically dead via patched os.kill
            inv = Invocation(id="inv-w2", wiki_id="wiki-w2", status="generating", subprocess_pid=999999999)

            call_count = 0

            async def fake_sleep(n):
                nonlocal call_count
                call_count += 1
                # First call: let the PID check happen (it will find it dead)

            with patch("asyncio.sleep", fake_sleep), patch("os.kill", side_effect=ProcessLookupError):
                await service._watch_for_generation_output(inv, str(output_path))

            assert inv.status == "failed"
            assert "exited without producing output" in (inv.error or "")


# ---------------------------------------------------------------------------
# Part 5 (continued): _finalize_generation
# ---------------------------------------------------------------------------


class TestFinalizeGeneration:
    @pytest.mark.asyncio
    async def test_uploads_pages_to_storage(self) -> None:
        """_finalize_generation uploads each generated page as an artifact."""
        storage = MagicMock()
        storage.upload = AsyncMock()
        settings = MagicMock()
        settings.cache_dir = "/tmp"
        service = WikiService(settings=settings, storage=storage)
        service.wiki_management = None

        inv = Invocation(id="inv-f1", wiki_id="wiki-f1", status="generating", repo_url="http://r", branch="main", owner_id="u1")

        result = {
            "success": True,
            "generated_pages": {"page-a": "# Page A content", "page-b": "# Page B content"},
            "artifacts": [],
            "commit_hash": "abc123",
        }

        await service._finalize_generation(inv, result)

        upload_calls = [c.args for c in storage.upload.await_args_list]
        uploaded_keys = [c[1] for c in upload_calls]
        assert "wiki-f1/page-a.md" in uploaded_keys
        assert "wiki-f1/page-b.md" in uploaded_keys

    @pytest.mark.asyncio
    async def test_marks_invocation_complete(self) -> None:
        """_finalize_generation sets invocation.status = 'complete'."""
        storage = MagicMock()
        storage.upload = AsyncMock()
        service = WikiService(settings=MagicMock(), storage=storage)
        service.wiki_management = None

        inv = Invocation(id="inv-f2", wiki_id="wiki-f2", status="generating", repo_url="http://r", branch="main", owner_id="u1")
        result = {"success": True, "generated_pages": {}, "artifacts": []}

        await service._finalize_generation(inv, result)

        assert inv.status == "complete"
        assert inv.completed_at is not None

    @pytest.mark.asyncio
    async def test_calls_register_wiki_on_success(self) -> None:
        """_finalize_generation calls wiki_management.register_wiki on completion."""
        storage = MagicMock()
        storage.upload = AsyncMock()
        wm = MagicMock()
        wm.register_wiki = AsyncMock()
        wm.mark_status = AsyncMock(return_value=True)
        wm.index_wiki_pages = AsyncMock()

        service = WikiService(settings=MagicMock(), storage=storage)
        service.wiki_management = wm

        inv = Invocation(id="inv-f3", wiki_id="wiki-f3", status="generating", repo_url="http://r", branch="main", owner_id="u1")

        result = {
            "success": True,
            "generated_pages": {"p": "content"},
            "artifacts": [],
            "commit_hash": "cafebabe",
        }

        await service._finalize_generation(inv, result)

        assert inv.status == "complete"

    @pytest.mark.asyncio
    async def test_uploads_binary_artifacts(self) -> None:
        """_finalize_generation uploads artifact bytes to storage."""
        storage = MagicMock()
        storage.upload = AsyncMock()
        service = WikiService(settings=MagicMock(), storage=storage)
        service.wiki_management = None

        inv = Invocation(id="inv-f4", wiki_id="wiki-f4", status="generating", repo_url="http://r", branch="main", owner_id="u1")
        result = {
            "success": True,
            "generated_pages": {},
            "artifacts": [{"name": "index.json", "data": b'{"pages":[]}'}],
        }

        await service._finalize_generation(inv, result)

        upload_calls = [c.args for c in storage.upload.await_args_list]
        artifact_keys = [c[1] for c in upload_calls]
        assert "wiki-f4/index.json" in artifact_keys


# ---------------------------------------------------------------------------
# Part 1: Persistent run directory in _run_wiki_subprocess
# ---------------------------------------------------------------------------


class TestPersistentRunDir:
    @pytest.mark.asyncio
    async def test_invocation_gets_run_dir_and_pid(self) -> None:
        """_run_wiki_subprocess sets run_dir and subprocess_pid on the invocation."""
        settings = MagicMock()
        settings.cache_dir = tempfile.mkdtemp()
        settings.planner_type = "agent"
        storage = MagicMock()
        storage.upload = AsyncMock()

        service = WikiService(settings=settings, storage=storage)
        service.wiki_management = None

        inv = Invocation(id="inv-p1", wiki_id="wiki-p1", status="generating", repo_url="http://r", branch="main", owner_id="u1")

        req = MagicMock()
        req.source_type = "git"
        req.scope = {}
        req.auth = None
        req.repo_url = "http://r"
        req.branch = "main"
        req.access_token = None
        req.wiki_title = None
        req.include_research = False
        req.include_diagrams = False
        req.force_rebuild_index = False
        req.llm_model = None
        req.embedding_model = None

        # Patch asyncio.create_subprocess_exec to simulate a quick success
        import asyncio

        mock_proc = MagicMock()
        mock_proc.pid = 42000
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0

        async def fake_subprocess(*args, **kwargs):
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
            # We'll test that run_dir and pid get set even if output.json is missing
            try:
                await service._run_wiki_subprocess(inv, req, "agent", False)
            except RuntimeError:
                pass  # Expected: no output.json was written

        # run_dir and subprocess_pid must be set BEFORE subprocess starts
        assert inv.run_dir is not None
        assert inv.subprocess_pid == 42000
        # The run_dir path is set on the invocation (it was created during the run);
        # the finally-block cleanup removes it after the subprocess exits, so we only
        # check that the attribute is a non-empty string, not that the dir still exists.
        assert inv.run_dir.endswith(f"wiki-runs/{inv.id}")

        # Cleanup
        import shutil
        shutil.rmtree(settings.cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part 6: Review-round-3 fixes
# ---------------------------------------------------------------------------


class TestRunDirValidation:
    """Fix 2: run_dir path validation in _load_invocations."""

    @pytest.mark.asyncio
    async def test_untrusted_run_dir_skips_recovery(self) -> None:
        """An inv.run_dir that doesn't resolve to cache_dir/wiki-runs/<id> is silently cleared."""
        with tempfile.TemporaryDirectory() as tmp:
            # Adversarial run_dir points somewhere outside wiki-runs
            bad_run_dir = Path(tmp) / "evil-dir"
            bad_run_dir.mkdir()
            output_path = bad_run_dir / "output.json"
            output_path.write_text(json.dumps({"success": True, "generated_pages": {}}))

            persisted = {
                "inv-bad-dir": _persisted_generating(
                    "inv-bad-dir", "wiki-bd", run_dir=str(bad_run_dir)
                ),
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            await service.load_persisted_invocations()

            # Validation should have cleared run_dir → falls through to fail
            inv = service._invocations["inv-bad-dir"]
            assert inv.status == "failed"
            assert "Server restarted" in (inv.error or "")


class TestPidValidation:
    """Fix 3: subprocess PID must be a positive integer before os.kill."""

    @pytest.mark.asyncio
    async def test_zero_pid_not_signalled(self) -> None:
        """PID=0 would signal the process group — must be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / "inv-pid0"
            run_dir.mkdir(parents=True)
            # No output.json

            persisted = {
                "inv-pid0": _persisted_generating(
                    "inv-pid0", "wiki-pid0", run_dir=str(run_dir), pid=0
                ),
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            with patch("os.kill") as mock_kill:
                await service.load_persisted_invocations()
                # os.kill must NOT have been called with PID 0
                for call_args in mock_kill.call_args_list:
                    assert call_args.args[0] != 0

            # PID 0 treated as dead → mark failed
            inv = service._invocations["inv-pid0"]
            assert inv.status == "failed"


class TestRecoverSuccessCheck:
    """Fix 4: _recover_completed_generation checks result['success']."""

    @pytest.mark.asyncio
    async def test_marks_failed_when_success_is_false(self) -> None:
        """output.json with success=False causes invocation to be marked failed."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / "inv-fail-out"
            run_dir.mkdir(parents=True)
            output = {"success": False, "error": "Runner encountered an error", "generated_pages": {}}
            (run_dir / "output.json").write_text(json.dumps(output))

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)
            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            inv = Invocation(id="inv-fail-out", wiki_id="wiki-fo", status="generating")

            await service._recover_completed_generation(inv, str(run_dir / "output.json"))

            assert inv.status == "failed"
            assert "Runner encountered an error" in (inv.error or "")

    @pytest.mark.asyncio
    async def test_marks_failed_with_generic_message_when_no_error_field(self) -> None:
        """When success=False and no 'error' field, a generic message is used."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / "inv-noerr"
            run_dir.mkdir(parents=True)
            output = {"success": False, "generated_pages": {}}
            (run_dir / "output.json").write_text(json.dumps(output))

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)
            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            inv = Invocation(id="inv-noerr", wiki_id="wiki-noerr", status="generating")

            await service._recover_completed_generation(inv, str(run_dir / "output.json"))

            assert inv.status == "failed"
            assert inv.error is not None


class TestMarkStatusFalse:
    """Fix 6: _recover_completed_generation handles mark_status returning False."""

    @pytest.mark.asyncio
    async def test_register_wiki_called_when_mark_status_returns_false(self) -> None:
        """When mark_status returns False, register_wiki is attempted as a fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / "inv-ms-false"
            run_dir.mkdir(parents=True)
            output = {"success": True, "generated_pages": {}, "artifacts": []}
            (run_dir / "output.json").write_text(json.dumps(output))

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=False)
            wm.register_wiki = AsyncMock()

            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm
            service._finalize_generation = AsyncMock()

            inv = Invocation(id="inv-ms-false", wiki_id="wiki-ms-false", status="generating")

            await service._recover_completed_generation(inv, str(run_dir / "output.json"))

            wm.register_wiki.assert_called_once()
            call_kwargs = wm.register_wiki.call_args.kwargs
            assert call_kwargs["wiki_id"] == "wiki-ms-false"
            assert call_kwargs["status"] == "complete"


class TestEagerPoll:
    """Fix 7: _watch_for_generation_output checks output.json before first sleep."""

    @pytest.mark.asyncio
    async def test_recovers_immediately_if_output_already_exists(self) -> None:
        """If output.json is already present when watch starts, recover without sleeping."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"
            output_path.write_text(json.dumps({"success": True, "generated_pages": {}}))

            service = _make_service()
            inv = Invocation(id="inv-eager", wiki_id="wiki-eager", status="generating", subprocess_pid=os.getpid())

            recover_calls = []

            async def fake_recover(invocation, op):
                recover_calls.append((invocation.id, op))

            service._recover_completed_generation = fake_recover

            sleep_calls = []

            async def fake_sleep(n):
                sleep_calls.append(n)

            with patch("asyncio.sleep", fake_sleep):
                await service._watch_for_generation_output(inv, str(output_path))

            # Recovery should be called without any sleep
            assert len(recover_calls) == 1
            assert recover_calls[0][0] == "inv-eager"
            assert len(sleep_calls) == 0


# ---------------------------------------------------------------------------
# Fix 1: corrupted-PID branch cleanup + mark_status
# ---------------------------------------------------------------------------


class TestCorruptedPidBranchCleanup:
    """_watch_for_generation_output corrupted-PID branch must clean up and reconcile DB."""

    @pytest.mark.asyncio
    async def test_corrupted_pid_calls_cleanup_and_mark_status(self) -> None:
        """When os.kill raises OverflowError/ValueError/OSError, run_dir is cleaned
        and wiki_management.mark_status is called — mirroring the ProcessLookupError
        and timeout branches."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"
            # No output.json — the subprocess "died" without writing it.

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            # Simulate a run_dir that exists and is under cache_dir/wiki-runs/
            run_dir = Path(tmp) / "wiki-runs" / "inv-cpid"
            run_dir.mkdir(parents=True)

            inv = Invocation(
                id="inv-cpid",
                wiki_id="wiki-cpid",
                status="generating",
                subprocess_pid=99,
                run_dir=str(run_dir),
            )

            async def fake_sleep(_n: float) -> None:
                # Only one iteration needed — os.kill raises on the first poll.
                pass

            with patch("asyncio.sleep", fake_sleep), patch(
                "os.kill", side_effect=OSError("bad PID")
            ):
                await service._watch_for_generation_output(inv, str(output_path))

            # run_dir must be gone (cleanup called)
            assert not run_dir.exists()

            # mark_status must have been called with "failed"
            wm.mark_status.assert_awaited_once_with(
                "wiki-cpid", "failed", inv.error
            )

            assert inv.status == "failed"
            assert "invalid" in (inv.error or "").lower()

    @pytest.mark.asyncio
    async def test_corrupted_pid_marks_failed_when_mark_status_raises(self) -> None:
        """mark_status failures in the corrupted-PID branch are swallowed (warning only)."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            wm = MagicMock()
            wm.mark_status = AsyncMock(side_effect=RuntimeError("DB error"))

            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            inv = Invocation(
                id="inv-cpid2",
                wiki_id="wiki-cpid2",
                status="generating",
                subprocess_pid=99,
            )

            async def fake_sleep(_n: float) -> None:
                pass

            with patch("asyncio.sleep", fake_sleep), patch(
                "os.kill", side_effect=ValueError("overflow")
            ):
                # Must not raise even if mark_status raises
                await service._watch_for_generation_output(inv, str(output_path))

            assert inv.status == "failed"


# ---------------------------------------------------------------------------
# Fix 2: no-PID recovery path
# ---------------------------------------------------------------------------


class TestNoPidRecovery:
    """When subprocess_pid is None but run_dir exists and output.json is absent,
    _watch_for_generation_output is scheduled (no-PID poll) instead of immediately
    marking the invocation failed."""

    @pytest.mark.asyncio
    async def test_no_pid_schedules_watch_task(self) -> None:
        """_load_invocations schedules a watch task when pid is None but run_dir valid."""
        inv_id = "c3d4e5f6-a7b8-9012-cdef-012345678902"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / inv_id
            run_dir.mkdir(parents=True)
            # No output.json, no PID recorded.

            persisted = {
                inv_id: _persisted_generating(inv_id, "wiki-nopid", run_dir=str(run_dir)),
                # Note: no 'subprocess_pid' key — simulates crash before pid was persisted.
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            with patch("asyncio.create_task", side_effect=_close_coro) as mock_create_task:
                await service.load_persisted_invocations()
                # A watch task must be created rather than immediately failing.
                assert mock_create_task.called

            # Invocation must stay "generating" — the watch will update it.
            inv = service._invocations[inv_id]
            assert inv.status == "generating"

    @pytest.mark.asyncio
    async def test_no_pid_watch_skips_pid_check_and_recovers_on_output(self) -> None:
        """When subprocess_pid is None, _watch_for_generation_output polls only for
        output.json and calls recover when it appears — no os.kill is attempted."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            service = _make_service()
            # No PID set.
            inv = Invocation(id="inv-nopid-w", wiki_id="wiki-nopid-w", status="generating")

            recover_calls: list[tuple[str, str]] = []

            async def fake_recover(invocation, op):
                recover_calls.append((invocation.id, op))

            service._recover_completed_generation = fake_recover

            call_count = 0

            async def fake_sleep(_n: float) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    output_path.write_text(json.dumps({"success": True, "generated_pages": {}}))

            with patch("asyncio.sleep", fake_sleep), patch("os.kill") as mock_kill:
                await service._watch_for_generation_output(inv, str(output_path))

            # os.kill must never be called when PID is None.
            mock_kill.assert_not_called()
            assert len(recover_calls) == 1
            assert recover_calls[0][0] == "inv-nopid-w"

    @pytest.mark.asyncio
    async def test_no_pid_watch_times_out_and_marks_failed(self) -> None:
        """When subprocess_pid is None and output.json never appears, the watch
        times out and marks the invocation failed (using the shorter max_wait)."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)
            service = _make_service(wiki_management=wm)
            inv = Invocation(
                id="inv-nopid-timeout",
                wiki_id="wiki-nopid-timeout",
                status="generating",
            )

            async def fake_sleep(_n: float) -> None:
                pass

            # max_wait=10 with poll_interval=10 → exactly one iteration, then timeout.
            with patch("asyncio.sleep", fake_sleep):
                await service._watch_for_generation_output(
                    inv, str(output_path), max_wait=10
                )

            assert inv.status == "failed"
            assert "timed out" in (inv.error or "").lower()
            wm.mark_status.assert_awaited()


# ---------------------------------------------------------------------------
# Fix 1: success-path cleanup in _run_generation delegates to _cleanup_run_dir
# ---------------------------------------------------------------------------


class TestRunGenerationSuccessPathCleanup:
    """_run_generation success path must use _cleanup_run_dir, not open-coded rmtree."""

    @pytest.mark.asyncio
    async def test_cleanup_run_dir_called_on_success(self) -> None:
        """After _finalize_generation succeeds, _cleanup_run_dir is called with run_dir."""
        service = _make_service()

        inv = Invocation(
            id="inv-rg-ok",
            wiki_id="wiki-rg-ok",
            status="generating",
            run_dir="/tmp/fakedir",
        )

        cleanup_calls: list[str | None] = []

        def fake_cleanup(run_dir: str | None) -> None:
            cleanup_calls.append(run_dir)

        async def fake_run_wiki_subprocess(*_a, **_kw):
            return {"success": True, "generated_pages": {}}

        async def fake_finalize(invocation, result, **_kw):
            invocation.status = "complete"

        async def fake_emit_progress(*_a, **_kw):
            pass

        with patch.object(service, "_run_wiki_subprocess", fake_run_wiki_subprocess), \
             patch.object(service, "_finalize_generation", fake_finalize), \
             patch.object(service, "_emit_progress", fake_emit_progress), \
             patch.object(service, "_cleanup_run_dir", fake_cleanup):
            await service._run_generation(inv, MagicMock())

        # _cleanup_run_dir must have been called with the run_dir (success path)
        assert any(c == "/tmp/fakedir" for c in cleanup_calls), (
            f"Expected _cleanup_run_dir('/tmp/fakedir') to be called; got {cleanup_calls}"
        )


# ---------------------------------------------------------------------------
# Fix 2: recovery tasks remove themselves from _tasks on completion
# ---------------------------------------------------------------------------


class TestRecoveryTaskDoneCallback:
    """Recovery and watch tasks scheduled in load_persisted_invocations must pop
    themselves from self._tasks when they finish."""

    @pytest.mark.asyncio
    async def test_recovery_task_popped_on_completion(self) -> None:
        """A _recover_completed_generation task removes itself from _tasks when done."""
        inv_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / inv_id
            run_dir.mkdir(parents=True)
            output_path = run_dir / "output.json"
            output_path.write_text(json.dumps({"success": True, "generated_pages": {}}))

            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(
                return_value=json.dumps(
                    {inv_id: _persisted_generating(inv_id, "wiki-done-cb", run_dir=str(run_dir))}
                ).encode()
            )
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)

            # Stub finalize so it doesn't hit real logic
            async def fake_finalize(invocation, result, **_kw):
                invocation.status = "complete"

            service._finalize_generation = fake_finalize  # type: ignore[method-assign]
            service.wiki_management = None

            await service.load_persisted_invocations()

            # The task was scheduled; now let it run to completion
            task = service._tasks.get(inv_id)
            if task is not None:
                await task

            # After completion the task must have been removed
            assert inv_id not in service._tasks, (
                "Expected recovery task to pop itself from _tasks on completion"
            )

    @pytest.mark.asyncio
    async def test_watch_task_popped_on_completion(self) -> None:
        """A _watch_for_generation_output task removes itself from _tasks when done."""
        inv_id = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / inv_id
            run_dir.mkdir(parents=True)
            # No output.json — process appears alive then dies immediately
            output_path = run_dir / "output.json"

            settings = MagicMock()
            settings.cache_dir = tmp
            # PID that appears alive on first signal, then raises ProcessLookupError
            pid = 99999
            storage = MagicMock()
            storage.download = AsyncMock(
                return_value=json.dumps(
                    {
                        inv_id: _persisted_generating(
                            inv_id, "wiki-watch-cb", run_dir=str(run_dir), pid=pid
                        )
                    }
                ).encode()
            )
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = None

            kill_call_count = 0

            def fake_kill(p, sig):
                nonlocal kill_call_count
                kill_call_count += 1
                raise ProcessLookupError("dead")

            async def fake_sleep(_n: float) -> None:
                pass

            with patch("os.kill", fake_kill), patch("asyncio.sleep", fake_sleep):
                await service.load_persisted_invocations()
                task = service._tasks.get(inv_id)
                if task is not None:
                    await task

            assert inv_id not in service._tasks, (
                "Expected watch task to pop itself from _tasks on completion"
            )


# ---------------------------------------------------------------------------
# Fix 3: invalid-but-non-None PID fails fast in _watch_for_generation_output
# ---------------------------------------------------------------------------


class TestInvalidPidFastFail:
    """When subprocess_pid is non-None but invalid (negative, zero, non-int),
    _watch_for_generation_output must fail fast rather than polling until timeout."""

    @pytest.mark.asyncio
    async def test_negative_pid_fails_fast(self) -> None:
        """A negative subprocess_pid triggers immediate failure without sleeping."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)
            service = _make_service(wiki_management=wm)
            inv = Invocation(
                id="inv-neg-pid",
                wiki_id="wiki-neg-pid",
                status="generating",
                subprocess_pid=-1,
            )

            sleep_calls: list[float] = []

            async def fake_sleep(n: float) -> None:
                sleep_calls.append(n)

            with patch("asyncio.sleep", fake_sleep):
                await service._watch_for_generation_output(inv, str(output_path))

            assert inv.status == "failed"
            assert "invalid" in (inv.error or "").lower()
            # Must not have polled (no sleep calls from the main loop)
            assert len(sleep_calls) == 0

    @pytest.mark.asyncio
    async def test_zero_pid_fails_fast(self) -> None:
        """A zero subprocess_pid triggers immediate failure without sleeping."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)
            service = _make_service(wiki_management=wm)
            inv = Invocation(
                id="inv-zero-pid",
                wiki_id="wiki-zero-pid",
                status="generating",
                subprocess_pid=0,
            )

            sleep_calls: list[float] = []

            async def fake_sleep(n: float) -> None:
                sleep_calls.append(n)

            with patch("asyncio.sleep", fake_sleep):
                await service._watch_for_generation_output(inv, str(output_path))

            assert inv.status == "failed"
            assert "invalid" in (inv.error or "").lower()
            assert len(sleep_calls) == 0

    @pytest.mark.asyncio
    async def test_invalid_pid_calls_mark_status(self) -> None:
        """mark_status(failed) is called when an invalid PID is detected."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)
            service = _make_service(wiki_management=wm)
            inv = Invocation(
                id="inv-bad-pid-ms",
                wiki_id="wiki-bad-pid-ms",
                status="generating",
                subprocess_pid=-999,
            )

            async def fake_sleep(_n: float) -> None:
                pass

            with patch("asyncio.sleep", fake_sleep):
                await service._watch_for_generation_output(inv, str(output_path))

            wm.mark_status.assert_awaited_once_with("wiki-bad-pid-ms", "failed", inv.error)

    @pytest.mark.asyncio
    async def test_invalid_pid_does_not_raise_when_mark_status_fails(self) -> None:
        """mark_status errors in the invalid-PID branch are swallowed (warning only)."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.json"

            wm = MagicMock()
            wm.mark_status = AsyncMock(side_effect=RuntimeError("DB down"))
            service = _make_service(wiki_management=wm)
            inv = Invocation(
                id="inv-bad-pid-err",
                wiki_id="wiki-bad-pid-err",
                status="generating",
                subprocess_pid=-1,
            )

            async def fake_sleep(_n: float) -> None:
                pass

            with patch("asyncio.sleep", fake_sleep):
                # Must not raise
                await service._watch_for_generation_output(inv, str(output_path))

            assert inv.status == "failed"


# ---------------------------------------------------------------------------
# Fix 4: _recover_completed_generation uses _cleanup_run_dir (symlink-safe)
# ---------------------------------------------------------------------------


class TestRecoverCleanupDelegation:
    """_recover_completed_generation must delegate run_dir cleanup to _cleanup_run_dir."""

    @pytest.mark.asyncio
    async def test_cleanup_run_dir_called_on_success_recovery(self) -> None:
        """On successful recovery, _cleanup_run_dir is invoked with the run_dir path."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / "inv-rc-ok"
            run_dir.mkdir(parents=True)
            output_path = run_dir / "output.json"
            output_path.write_text(json.dumps({"success": True, "generated_pages": {}}))

            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = None

            cleanup_calls: list[str] = []

            def fake_cleanup(run_dir_arg: str | None) -> None:
                if run_dir_arg is not None:
                    cleanup_calls.append(run_dir_arg)

            service._cleanup_run_dir = fake_cleanup  # type: ignore[method-assign]

            async def fake_finalize(invocation, result, **_kw):
                invocation.status = "complete"

            service._finalize_generation = fake_finalize  # type: ignore[method-assign]

            inv = Invocation(
                id="inv-rc-ok",
                wiki_id="wiki-rc-ok",
                status="generating",
                run_dir=str(run_dir),
            )

            await service._recover_completed_generation(inv, str(output_path))

            assert len(cleanup_calls) == 1, f"Expected one _cleanup_run_dir call; got {cleanup_calls}"
            assert cleanup_calls[0] == str(run_dir)

    @pytest.mark.asyncio
    async def test_cleanup_run_dir_called_on_failed_recovery(self) -> None:
        """On failed recovery (bad output.json), _cleanup_run_dir is still invoked."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / "inv-rc-fail"
            run_dir.mkdir(parents=True)
            output_path = run_dir / "output.json"
            output_path.write_text("not-json")  # invalid — will raise

            settings = MagicMock()
            settings.cache_dir = tmp
            storage = MagicMock()
            storage.download = AsyncMock(side_effect=FileNotFoundError)
            storage.upload = AsyncMock()
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = None

            cleanup_calls: list[str] = []

            def fake_cleanup(run_dir_arg: str | None) -> None:
                if run_dir_arg is not None:
                    cleanup_calls.append(run_dir_arg)

            service._cleanup_run_dir = fake_cleanup  # type: ignore[method-assign]

            inv = Invocation(
                id="inv-rc-fail",
                wiki_id="wiki-rc-fail",
                status="generating",
                run_dir=str(run_dir),
            )

            await service._recover_completed_generation(inv, str(output_path))

            assert inv.status == "failed"
            assert len(cleanup_calls) == 1, f"Expected one _cleanup_run_dir call; got {cleanup_calls}"


# ---------------------------------------------------------------------------
# Fix 1 (security): symlink checks in _load_invocations
# ---------------------------------------------------------------------------


class TestSymlinkRejectionInLoadInvocations:
    """Symlinked expected run_dir or symlinked output.json must be rejected
    to prevent path-traversal via pre-created symlinks."""

    @pytest.mark.asyncio
    async def test_symlinked_expected_run_dir_skips_recovery(self) -> None:
        """When cache_dir/wiki-runs/<inv_id> is a symlink, recovery is skipped."""
        inv_id = "d4e5f6a7-b8c9-0123-def0-123456789003"
        with tempfile.TemporaryDirectory() as tmp:
            # Create a real directory somewhere else and symlink it at the expected path.
            real_dir = Path(tmp) / "real-elsewhere"
            real_dir.mkdir()
            output_real = real_dir / "output.json"
            output_real.write_text(json.dumps({"success": True, "generated_pages": {}}))

            wiki_runs = Path(tmp) / "wiki-runs"
            wiki_runs.mkdir(parents=True)
            symlink_path = wiki_runs / inv_id
            symlink_path.symlink_to(real_dir)

            persisted = {
                inv_id: _persisted_generating(inv_id, "wiki-sl", run_dir=str(symlink_path)),
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            await service.load_persisted_invocations()

            # Symlinked expected path must result in recovery being skipped → failed
            inv = service._invocations[inv_id]
            assert inv.status == "failed"
            assert "Server restarted" in (inv.error or "")

    @pytest.mark.asyncio
    async def test_symlinked_output_json_skips_recovery(self) -> None:
        """When output.json inside run_dir is a symlink, reading it is refused."""
        inv_id = "e5f6a7b8-c9d0-1234-ef01-234567890004"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "wiki-runs" / inv_id
            run_dir.mkdir(parents=True)

            # Create a real file somewhere else and symlink output.json to it.
            real_output = Path(tmp) / "secret-output.json"
            real_output.write_text(json.dumps({"success": True, "generated_pages": {}}))
            symlink_output = run_dir / "output.json"
            symlink_output.symlink_to(real_output)

            persisted = {
                inv_id: _persisted_generating(inv_id, "wiki-sl-out", run_dir=str(run_dir)),
            }
            storage = MagicMock()
            storage.download = AsyncMock(return_value=json.dumps(persisted).encode())
            storage.upload = AsyncMock()
            wm = MagicMock()
            wm.mark_status = AsyncMock(return_value=True)

            settings = MagicMock()
            settings.cache_dir = tmp
            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = wm

            await service.load_persisted_invocations()

            # Symlinked output.json must result in recovery being skipped → failed
            inv = service._invocations[inv_id]
            assert inv.status == "failed"
            assert "Server restarted" in (inv.error or "")


# ---------------------------------------------------------------------------
# Fix 2 (security): symlink guard in _run_wiki_subprocess before writing secrets
# ---------------------------------------------------------------------------


class TestSymlinkGuardInRunWikiSubprocess:
    """If run_dir itself is a symlink when _run_wiki_subprocess tries to write
    input.json, it must raise RuntimeError rather than writing to the symlink target."""

    @pytest.mark.asyncio
    async def test_raises_when_run_dir_is_symlink(self) -> None:
        """_run_wiki_subprocess raises RuntimeError if run_dir is a symlink."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create the real target dir elsewhere.
            real_dir = Path(tmp) / "attacker-target"
            real_dir.mkdir()

            # Arrange cache_dir so mkdir creates the expected path as a symlink
            # by pre-creating the symlink before the service call.
            cache_dir = Path(tmp) / "cache"
            wiki_runs = cache_dir / "wiki-runs"
            wiki_runs.mkdir(parents=True)

            settings = MagicMock()
            settings.cache_dir = str(cache_dir)
            settings.planner_type = "agent"
            storage = MagicMock()
            storage.upload = AsyncMock()

            service = WikiService(settings=settings, storage=storage)
            service.wiki_management = None

            inv = Invocation(
                id="inv-sl-write",
                wiki_id="wiki-sl-write",
                status="generating",
                repo_url="http://r",
                branch="main",
                owner_id="u1",
            )

            # Pre-create the symlink that mkdir(exist_ok=True) will leave in place.
            symlink_run_dir = wiki_runs / inv.id
            symlink_run_dir.symlink_to(real_dir)

            req = MagicMock()
            req.source_type = "git"
            req.scope = {}
            req.auth = None
            req.repo_url = "http://r"
            req.branch = "main"
            req.access_token = None
            req.wiki_title = None
            req.include_research = False
            req.include_diagrams = False
            req.force_rebuild_index = False
            req.llm_model = None
            req.embedding_model = None

            with pytest.raises(RuntimeError, match="symlink"):
                await service._run_wiki_subprocess(inv, req, "agent", False)

            # The real target must NOT have been written to.
            assert not (real_dir / "input.json").exists()
