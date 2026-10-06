"""Regression test: the metasploit wrapper must run non-interactively.

Bug: `app/tools/metasploit.yaml` passed the msfconsole command as a POSITIONAL
arg (`msfconsole "<command>" -q`). msfconsole treats a positional arg as a
resource-script path, so it printed an error, dropped into the interactive
`msf >` REPL, emitted `stty: ... Inappropriate ioctl for device`, triggered the
executor's PTY retry, and then HUNG at the interactive prompt — the agent's
metasploit call never actually executed (and held the scan for up to 3600s).

Fix: the command is now passed via `-x` and the wrapper auto-appends `exit`,
producing `msfconsole -q -x "<command>; exit"`, which runs the module and exits.

Run:
    pytest tests/unit/test_metasploit_wrapper.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.tools.loader import ParameterSpec, load_all_tools


class TestMetasploitCommandBuilding:
    def test_command_is_x_flag_not_positional(self):
        tool = load_all_tools(force_reload=True)["metasploit"]
        param = next(p for p in tool.parameters if p.name == "command")
        assert param.flag == "-x", "msfconsole command must be passed via -x"
        assert param.position is None, "command must NOT be a positional (resource file) arg"

    def test_auto_appends_exit(self):
        tool = load_all_tools(force_reload=True)["metasploit"]
        cmd = tool.build_command_args(
            command="use auxiliary/scanner/http/http_version; set RHOSTS pentest-ground.com; set RPORT 81; run",
        )
        joined = " ".join(cmd)
        assert "-x" in cmd and "-q" in cmd
        x_value = cmd[cmd.index("-x") + 1]
        assert x_value.rstrip().endswith("; exit"), (
            f"expected auto-appended '; exit', got {x_value!r}"
        )
        # The raw command must not be a bare positional that msfconsole would
        # interpret as a resource file.
        assert cmd[0].startswith("-"), f"first arg should be a flag, got {cmd[0]!r}"

    def test_suffix_mechanism_is_generic(self):
        p = ParameterSpec(name="command", type="string", description="d", flag="-x", suffix="; exit")
        assert str("run") + p.suffix == "run; exit"


class TestTimeout:
    def test_timeout_not_interactive_scale(self):
        tool = load_all_tools(force_reload=True)["metasploit"]
        # 3600s let a broken interactive invocation hang for an hour.
        assert tool.timeout <= 600
