from pathlib import Path

from generate_workflows_lib import (
    generate_extension_lint_workflow,
    generate_extension_misc_workflow,
    generate_extension_test_workflow,
)

tox_ini_path = Path(__file__).parent.parent.parent.joinpath("tox.ini")
# loongsuite instrumentation modules
tox_loongsuite_ini_path = Path(__file__).parent.parent.parent.joinpath(
    "tox-loongsuite.ini"
)
workflows_directory_path = Path(__file__).parent
arc_runner_label = "loongsuite-python-arc"

generate_extension_test_workflow(
    tox_ini_path,
    workflows_directory_path,
    tox_loongsuite_ini_path,
    arc_runner_label,
)
generate_extension_lint_workflow(
    tox_ini_path,
    workflows_directory_path,
    tox_loongsuite_ini_path,
    arc_runner_label,
)
generate_extension_misc_workflow(
    tox_ini_path, workflows_directory_path, tox_loongsuite_ini_path
)
