"""
Tests for the actuator seam (S6): dry-run default, helm command
construction, token precedence, and never-raise error semantics.
"""

import subprocess
from unittest.mock import patch

import pytest

from fonpr.actuators import DryRunActuator, HelmActuator, get_repo_token

ACTIONS = {
    "target_pod": "amf",
    "requests": {"memory": 2, "cpu": 1},
    "limits": {"memory": 4, "cpu": 2},
}


class TestDryRun:
    def test_applies_nothing_and_records(self):
        actuator = DryRunActuator()
        result = actuator.apply(ACTIONS)
        assert result.success
        assert result.reference == "dry-run"
        assert actuator.history == [ACTIONS]


class TestHelm:
    def test_command_construction_flattens_nested_values(self):
        actuator = HelmActuator("respons", "charts/respons", namespace="open5gs")
        cmd = actuator.build_command(ACTIONS)
        assert cmd[:4] == ["helm", "upgrade", "respons", "charts/respons"]
        assert "--reuse-values" in cmd
        joined = " ".join(cmd)
        assert "limits.cpu=2" in joined
        assert "requests.memory=2" in joined
        assert "target_pod=amf" in joined

    def test_nonzero_exit_is_failure_not_exception(self):
        actuator = HelmActuator("r", "c")
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
        with patch("subprocess.run", return_value=fake):
            result = actuator.apply(ACTIONS)
        assert not result.success
        assert "boom" in result.error

    def test_missing_binary_is_failure_not_exception(self):
        actuator = HelmActuator("r", "c")
        with patch("subprocess.run", side_effect=FileNotFoundError("no helm")):
            result = actuator.apply(ACTIONS)
        assert not result.success

    def test_success_carries_reference(self):
        actuator = HelmActuator("respons", "c")
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=fake):
            result = actuator.apply(ACTIONS)
        assert result.success
        assert result.reference == "helm:respons"


class TestTokenPrecedence:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "tok-from-env")
        monkeypatch.setenv("GH_TOKEN_FILE", "/nonexistent")
        assert get_repo_token() == "tok-from-env"

    def test_file_second(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GH_TOKEN", raising=False)
        token_file = tmp_path / "token"
        token_file.write_text("tok-from-file\n")
        monkeypatch.setenv("GH_TOKEN_FILE", str(token_file))
        assert get_repo_token() == "tok-from-file"

    def test_secrets_manager_is_last_resort(self, monkeypatch):
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN_FILE", raising=False)
        # Without the [live] extra the AWS path is an ImportError, proving
        # it is only reached when both local sources are absent.
        with pytest.raises(ImportError):
            get_repo_token()
