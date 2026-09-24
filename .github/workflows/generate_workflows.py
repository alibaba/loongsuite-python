from pathlib import Path

from generate_workflows_lib import (
    generate_contrib_workflow,
    generate_lint_workflow,
    generate_misc_workflow,
    generate_test_workflow,
)

tox_ini_path = Path(__file__).parent.parent.parent.joinpath("tox.ini")
workflows_directory_path = Path(__file__).parent
arc_runner_label = "loongsuite-python-arc"

generate_test_workflow(
    tox_ini_path,
    workflows_directory_path,
    arc_runner_label,
    batch_size=4,
    parallelism=4,
)
generate_lint_workflow(
    tox_ini_path,
    workflows_directory_path,
    arc_runner_label,
    batch_size=4,
    parallelism=4,
)
generate_misc_workflow(
    tox_ini_path,
    workflows_directory_path,
    runner=arc_runner_label,
)
generate_contrib_workflow(workflows_directory_path)
