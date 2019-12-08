"""
Module for using the comet-ml experiment framework
"""
import os
import time
from typing import Sequence
from types import SimpleNamespace

from comet_ml import Experiment, ExistingExperiment

from utils import get_version_string


def initialize_experiment(
    args: SimpleNamespace, params: Sequence[str] = tuple(),
) -> Experiment:
    """
    Initialize experiment tracking if specified
    """
    track = args.track
    version = get_version_string()
    if track and "-dirty" in version:
        raise RuntimeError(
            "Trying to track an experiment, but the workspace is dirty! "
            "Commit your changes first, then try again."
        )

    api_key = None if track else ""
    experiment_type = Experiment
    experiment_args = [api_key]
    if isinstance(track, str):
        experiment_type = ExistingExperiment
        if track.endswith(".guid"):
            wait_count = 0
            while not os.path.exists(track):
                wait_string = "..."[: wait_count % 4]
                wait_count += 1

                print(
                    f"\r\033[KWaiting for experiment: {track} {wait_string}", end="",
                )
                time.sleep(1)

            print(f"\r\033[KLoading experiment: {track}")
            with open(track, "rt") as guid_file:
                experiment_args.append(guid_file.readline().strip())
        else:
            experiment_args.append(track)

    experiment = experiment_type(
        *experiment_args,
        project_name="storium-gpt2-baselines",
        workspace="umass-nlp",
        disabled=not track,
        auto_metric_logging=False,
        auto_output_logging=False,
        auto_param_logging=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        log_graph=False,
        log_code=False,
        parse_args=False,
    )

    if track and experiment_type == Experiment:
        with open(os.path.join(args.output_dir, "experiment.guid"), "wt") as guid_file:
            guid_file.write(experiment.id)

    # This needs to be called separately to disable monkey patching of the
    # ML frameworks which is on by default :(
    experiment.disable_mp()

    if experiment_type is Experiment:
        experiment.log_parameter("version", version)
        for name in params:
            experiment.log_parameters(
                getattr(getattr(args, name, None), "__dict__", {}), prefix=name
            )

    return experiment
