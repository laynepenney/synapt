from unittest.mock import patch


def test_dashboard_pid_and_log_paths_live_under_synapt_root(tmp_path):
    from synapt.dashboard.app import _dashboard_log_path, _dashboard_pid_path

    with patch("synapt.dashboard.app.project_data_dir", return_value=tmp_path / ".synapt" / "recall"):
        assert _dashboard_pid_path() == tmp_path / ".synapt" / "dashboard.pid"
        assert _dashboard_log_path() == tmp_path / ".synapt" / "dashboard.log"


def test_read_pid_returns_none_for_missing_or_invalid(tmp_path):
    from synapt.dashboard.app import _read_pid

    missing = tmp_path / "missing.pid"
    invalid = tmp_path / "invalid.pid"
    invalid.write_text("abc\n")

    assert _read_pid(missing) is None
    assert _read_pid(invalid) is None


def test_background_command_uses_foreground_child_mode():
    from synapt.dashboard.app import _background_command

    cmd = _background_command("127.0.0.1", 9000)
    assert cmd[1:4] == ["-m", "synapt.cli", "dashboard"]
    assert "--foreground" in cmd
    assert "--no-open" in cmd
    assert "9000" in cmd


def test_stop_dashboard_cleans_stale_pidfile(tmp_path):
    from synapt.dashboard.app import _stop_dashboard

    pid_path = tmp_path / ".synapt" / "dashboard.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("12345\n")

    with patch("synapt.dashboard.app._dashboard_pid_path", return_value=pid_path), \
         patch("synapt.dashboard.app._pid_is_running", return_value=False):
        assert _stop_dashboard() is False
        assert not pid_path.exists()


def test_agent_overlay_reuses_query_only_windows_pid_probe(tmp_path):
    """The dashboard status overlay reaches the shared Windows-safe probe."""
    from synapt.dashboard import app
    from synapt.recall import session_start

    assert app._pid_alive is session_start._pid_alive
    team_db = tmp_path / ".synapt" / "orgs" / "synapt-dev" / "team.db"
    team_db.parent.mkdir(parents=True)
    team_db.touch()
    pid_calls = []

    def query_only_probe(pid):
        pid_calls.append(pid)
        return True

    registered = [{
        "agent_id": "agent-001",
        "display_name": "Agent",
        "role": "agent",
        "status": "running",
        "pid": 424243,
        "tmux_target": "workspace:agent",
    }]
    with patch.object(session_start.sys, "platform", "win32"), \
         patch.object(
             session_start.os,
             "kill",
             side_effect=AssertionError("os.kill must not run on Windows"),
         ), \
         patch.object(session_start, "_pid_alive_win32", query_only_probe), \
         patch("synapt.dashboard.app.Path.home", return_value=tmp_path), \
         patch("synapt.dashboard.app._resolve_org_id", return_value="synapt-dev"), \
         patch("synapt.dashboard.app._registry_list_agents", return_value=registered), \
         patch("synapt.dashboard.app.channel_agents_json", return_value=[]), \
         patch("synapt.dashboard.app._tmux_window_agents", return_value={}):
        agents = app._combined_agents_json_sync()

    assert [agent["display_name"] for agent in agents] == ["Agent"]
    assert agents[0]["status"] == "running"
    assert pid_calls == [424243]


def test_stale_pid_cleanup_reuses_query_only_windows_pid_probe(tmp_path):
    """PID-file cleanup reaches the shared Windows-safe liveness probe."""
    from synapt.dashboard import app
    from synapt.recall import session_start

    assert app._pid_alive is session_start._pid_alive
    pid_path = tmp_path / "dashboard.pid"
    pid_path.write_text("424244\n")
    pid_calls = []

    def query_only_probe(pid):
        pid_calls.append(pid)
        return True

    with patch.object(session_start.sys, "platform", "win32"), \
         patch.object(
             session_start.os,
             "kill",
             side_effect=AssertionError("os.kill must not run on Windows"),
         ), \
         patch.object(session_start, "_pid_alive_win32", query_only_probe):
        app._cleanup_stale_pidfile(pid_path)

    assert pid_path.read_text() == "424244\n"
    assert pid_calls == [424244]


def test_synapt_help_lists_dashboard(capsys):
    from synapt.cli import main

    with patch("synapt.cli._discover_commands", return_value={}):
        import sys

        sys.argv = ["synapt", "--help"]
        main()

    captured = capsys.readouterr()
    assert "dashboard" in captured.out
