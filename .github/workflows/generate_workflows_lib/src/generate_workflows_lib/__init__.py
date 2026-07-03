from collections import defaultdict
from pathlib import Path
from re import compile as re_compile

from jinja2 import Environment, FileSystemLoader
from tox.config.cli.parse import get_options
from tox.config.sets import CoreConfigSet
from tox.config.source.tox_ini import ToxIni
from tox.session.state import State

_tox_test_env_regex = re_compile(
    r"(?P<python_version>py\w+)-test-"
    r"(?P<name>[-\w]+\w)-?(?P<test_requirements>\d+)?"
)
_tox_lint_env_regex = re_compile(r"lint-(?P<name>[-\w]+)")
_tox_contrib_env_regex = re_compile(
    r"py39-test-(?P<name>[-\w]+\w)-?(?P<contrib_requirements>\d+)?"
)
_OS_ALIASES = {
    "ubuntu-latest": "Ubuntu",
    "windows-latest": "Windows",
    "loongsuite-python-arc": "ARC",
}


def _slug(value: str) -> str:
    return (
        str(value).lower().replace(".", "").replace("-", "_").replace(" ", "_")
    )


def _get_job_package_name(name: str, package_names=None) -> str:
    if package_names is None:
        return name

    for package_name in sorted(package_names, key=len, reverse=True):
        if name == package_name or name.startswith(f"{package_name}-"):
            return package_name

    return name


def get_tox_envs(tox_ini_path: Path) -> list:
    tox_ini = ToxIni(tox_ini_path)

    conf = State(get_options(), []).conf

    tox_section = next(tox_ini.sections())

    core_config_set = CoreConfigSet(
        conf, tox_section, tox_ini_path.parent, tox_ini_path
    )

    (
        core_config_set.loaders.extend(
            tox_ini.get_loaders(
                tox_section,
                base=[],
                override_map=defaultdict(list, {}),
                conf=core_config_set,
            )
        )
    )

    return core_config_set.load("env_list")


def get_test_job_datas(
    tox_envs: list,
    operating_systems: list,
    package_names=None,
) -> list:
    python_version_alias = {
        "pypy3": "pypy-3.9",
        "pypy310": "pypy-3.10",
        "py39": "3.9",
        "py310": "3.10",
        "py311": "3.11",
        "py312": "3.12",
        "py313": "3.13",
        "py314": "3.14",
    }

    test_job_datas = []

    for operating_system in operating_systems:
        for tox_env in tox_envs:
            tox_test_env_match = _tox_test_env_regex.match(tox_env)

            if tox_test_env_match is None:
                continue

            groups = tox_test_env_match.groupdict()

            aliased_python_version = python_version_alias[
                groups["python_version"]
            ]
            tox_env = tox_test_env_match.string

            test_requirements = groups["test_requirements"]

            if test_requirements is None:
                test_requirements = " "

            else:
                test_requirements = f"-{test_requirements} "

            test_job_datas.append(
                {
                    "name": f"{tox_env}_{operating_system}",
                    "ui_name": (
                        f"{groups['name']}"
                        f"{test_requirements}"
                        f"{aliased_python_version} "
                        f"{_OS_ALIASES[operating_system]}"
                    ),
                    "package": _get_job_package_name(
                        groups["name"],
                        package_names,
                    ),
                    "python_version": aliased_python_version,
                    "tox_env": tox_env,
                    "os": operating_system,
                }
            )

    return test_job_datas


def get_lint_job_datas(tox_envs: list) -> list:
    lint_job_datas = []

    for tox_env in tox_envs:
        tox_lint_env_match = _tox_lint_env_regex.match(tox_env)

        if tox_lint_env_match is None:
            continue

        tox_env = tox_lint_env_match.string

        lint_job_datas.append(
            {
                "name": f"{tox_env}",
                "ui_name": f"{tox_lint_env_match.groupdict()['name']}",
                "package": _get_job_package_name(
                    tox_lint_env_match.groupdict()["name"]
                ),
                "tox_env": tox_env,
            }
        )

    return lint_job_datas


def get_contrib_job_datas(tox_envs: list) -> list:
    contrib_job_datas = []

    for tox_env in tox_envs:
        tox_contrib_env_match = _tox_contrib_env_regex.match(tox_env)

        if tox_contrib_env_match is None:
            continue

        groups = tox_contrib_env_match.groupdict()

        tox_env = tox_contrib_env_match.string

        contrib_requirements = groups["contrib_requirements"]

        if contrib_requirements is None:
            contrib_requirements = " "

        else:
            contrib_requirements = f"-{contrib_requirements} "

        contrib_job_datas.append(
            {
                "ui_name": (f"{groups['name']}{contrib_requirements}"),
                "tox_env": tox_env,
            }
        )

    return contrib_job_datas


def get_misc_job_datas(tox_envs: list) -> list:
    misc_job_datas = []

    _tox_benchmark_env_regex = re_compile(r"benchmark.+")

    for tox_env in tox_envs:
        if (
            _tox_test_env_regex.match(tox_env) is not None
            or _tox_lint_env_regex.match(tox_env) is not None
            or _tox_contrib_env_regex.match(tox_env) is not None
            or _tox_benchmark_env_regex.match(tox_env) is not None
        ):
            continue

        misc_job_datas.append(tox_env)

    return misc_job_datas


def _batch_job_datas(
    job_datas: list,
    batch_size: int,
    parallelism: int,
    group_keys=(),
) -> list:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if parallelism < 1:
        raise ValueError("parallelism must be >= 1")

    grouped_job_datas = {}
    for job_data in job_datas:
        key = tuple(job_data.get(group_key) for group_key in group_keys)
        grouped_job_datas.setdefault(key, []).append(job_data)

    batched_job_datas = []
    batch_index = 0

    for group in grouped_job_datas.values():
        for chunk_start in range(0, len(group), batch_size):
            chunk = group[chunk_start : chunk_start + batch_size]
            first_job_data = chunk[0]
            tox_envs = [job_data["tox_env"] for job_data in chunk]

            if len(chunk) == 1:
                batched_job_data = dict(first_job_data)
                batched_job_data["tox_envs"] = tox_envs
                batched_job_data["parallelism"] = 1
                batched_job_data["env_count"] = 1
                batched_job_datas.append(batched_job_data)
                continue

            batch_index += 1
            name_parts = [
                _slug(first_job_data[group_key])
                for group_key in group_keys
                if group_key in first_job_data
            ]
            name_suffix = "_".join(name_parts) if name_parts else "tox"

            if "python_version" in first_job_data and "os" in first_job_data:
                ui_prefix = (
                    f"{first_job_data['python_version']} "
                    f"{_OS_ALIASES.get(first_job_data['os'], first_job_data['os'])}"
                )
            else:
                ui_prefix = "tox"

            batched_job_data = dict(first_job_data)
            batched_job_data["name"] = f"batch_{batch_index:03d}_{name_suffix}"
            batched_job_data["ui_name"] = (
                f"{ui_prefix} batch {batch_index:03d} ({len(chunk)} envs)"
            )
            batched_job_data["tox_env"] = tox_envs[0]
            batched_job_data["tox_envs"] = tox_envs
            batched_job_data["parallelism"] = min(parallelism, len(chunk))
            batched_job_data["env_count"] = len(chunk)
            batched_job_datas.append(batched_job_data)

    return batched_job_datas


def _generate_workflow(
    job_datas: list,
    name: str,
    workflow_directory_path: Path,
    max_jobs=250,
    runner="ubuntu-latest",
):
    for stale_workflow_path in workflow_directory_path.glob(f"{name}_*.yml"):
        stale_workflow_path.unlink()

    # Github seems to limit the amount of jobs in a workflow file, that is why
    # they are split in groups of 250 per workflow file.
    for file_number, job_datas in enumerate(
        [
            job_datas[index : index + max_jobs]
            for index in range(0, len(job_datas), max_jobs)
        ]
    ):
        with open(
            workflow_directory_path.joinpath(f"{name}_{file_number}.yml"), "w"
        ) as test_yml_file:
            test_yml_file.write(
                Environment(loader=FileSystemLoader(Path(__file__).parent))
                .get_template(f"{name}.yml.j2")
                .render(
                    job_datas=job_datas,
                    file_number=file_number,
                    runner=runner,
                )
            )
            test_yml_file.write("\n")


def generate_test_workflow(
    tox_ini_path: Path,
    workflow_directory_path: Path,
    *operating_systems,
    batch_size=1,
    parallelism=1,
) -> None:
    job_datas = get_test_job_datas(
        get_tox_envs(tox_ini_path), operating_systems
    )
    _generate_workflow(
        _batch_job_datas(
            job_datas,
            batch_size=batch_size,
            parallelism=parallelism,
            group_keys=("os", "python_version"),
        ),
        "test",
        workflow_directory_path,
    )


def generate_lint_workflow(
    tox_ini_path: Path,
    workflow_directory_path: Path,
    runner="ubuntu-latest",
    batch_size=1,
    parallelism=1,
) -> None:
    job_datas = get_lint_job_datas(get_tox_envs(tox_ini_path))
    _generate_workflow(
        _batch_job_datas(
            job_datas,
            batch_size=batch_size,
            parallelism=parallelism,
        ),
        "lint",
        workflow_directory_path,
        runner=runner,
    )


def generate_contrib_workflow(
    workflow_directory_path: Path,
) -> None:
    _generate_workflow(
        get_contrib_job_datas(
            get_tox_envs(Path(__file__).parent.joinpath("tox.ini"))
        ),
        "core_contrib_test",
        workflow_directory_path,
    )


def generate_misc_workflow(
    tox_ini_path: Path,
    workflow_directory_path: Path,
) -> None:
    _generate_workflow(
        get_misc_job_datas(get_tox_envs(tox_ini_path)),
        "misc",
        workflow_directory_path,
    )


# LoongSuite Extension
def get_loongsuite_tox_envs(additional_config_path: Path) -> list:
    if not additional_config_path or not additional_config_path.exists():
        return []

    additional_tox_ini = ToxIni(additional_config_path)
    additional_conf = State(get_options(), []).conf
    additional_section = next(additional_tox_ini.sections())
    additional_config_set = CoreConfigSet(
        additional_conf,
        additional_section,
        additional_config_path.parent,
        additional_config_path,
    )
    (
        additional_config_set.loaders.extend(
            additional_tox_ini.get_loaders(
                additional_section,
                base=[],
                override_map=defaultdict(list, {}),
                conf=additional_config_set,
            )
        )
    )
    additional_env_list = additional_config_set.load("env_list")
    # Convert EnvList to list if needed
    return list(additional_env_list) if additional_env_list else []


def generate_extension_test_workflow(
    tox_ini_path: Path,
    workflow_directory_path: Path,
    additional_config_path: Path,
    *operating_systems,
) -> None:
    loongsuite_envs = get_loongsuite_tox_envs(additional_config_path)
    if not loongsuite_envs:
        return
    loongsuite_package_names = [
        job_data["package"] for job_data in get_lint_job_datas(loongsuite_envs)
    ]

    _generate_workflow_with_template(
        get_test_job_datas(
            loongsuite_envs,
            list(operating_systems),
            loongsuite_package_names,
        ),
        "loongsuite_test",
        "loongsuite_test",
        workflow_directory_path,
    )


def generate_extension_lint_workflow(
    tox_ini_path: Path,
    workflow_directory_path: Path,
    additional_config_path: Path,
    runner="ubuntu-latest",
) -> None:
    loongsuite_envs = get_loongsuite_tox_envs(additional_config_path)
    if not loongsuite_envs:
        return

    _generate_workflow_with_template(
        get_lint_job_datas(loongsuite_envs),
        "loongsuite_lint",
        "loongsuite_lint",
        workflow_directory_path,
        runner=runner,
    )


def generate_extension_misc_workflow(
    tox_ini_path: Path,
    workflow_directory_path: Path,
    additional_config_path: Path,
) -> None:
    loongsuite_envs = get_loongsuite_tox_envs(additional_config_path)
    if not loongsuite_envs:
        return

    _generate_workflow_with_template(
        get_misc_job_datas(loongsuite_envs),
        "loongsuite_misc",
        "loongsuite_misc",
        workflow_directory_path,
    )


def _generate_workflow_with_template(
    job_datas: list,
    name: str,
    template_name: str,
    workflow_directory_path: Path,
    max_jobs=250,
    runner="ubuntu-latest",
):
    if (
        name in {"loongsuite_lint", "loongsuite_test"}
        and len(job_datas) > max_jobs
    ):
        raise RuntimeError(
            f"{name} dynamic matrix has {len(job_datas)} jobs, which exceeds "
            f"the {max_jobs}-job single-workflow limit. Add an explicit "
            "cross-workflow aggregate before allowing generated chunks."
        )

    for stale_workflow_path in workflow_directory_path.glob(f"{name}_*.yml"):
        stale_workflow_path.unlink()

    # Github seems to limit the amount of jobs in a workflow file, that is why
    # they are split in groups of 250 per workflow file.
    for file_number, job_datas in enumerate(
        [
            job_datas[index : index + max_jobs]
            for index in range(0, len(job_datas), max_jobs)
        ]
    ):
        with open(
            workflow_directory_path.joinpath(f"{name}_{file_number}.yml"), "w"
        ) as test_yml_file:
            test_yml_file.write(
                Environment(loader=FileSystemLoader(Path(__file__).parent))
                .get_template(f"{template_name}.yml.j2")
                .render(
                    job_datas=job_datas,
                    file_number=file_number,
                    runner=runner,
                )
            )
            test_yml_file.write("\n")
