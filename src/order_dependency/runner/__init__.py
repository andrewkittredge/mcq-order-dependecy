"""Running an experiment: its settings, its execution, and its record.

* ``llm_config.LLMConfig`` - model/backend settings; ``LLMConfig.answerer`` builds the backend.
* ``run_experiment`` - enumerate every question x ordering x sample and ``run_experiment`` them concurrently.
* ``experiment_run.ExperimentRun`` - the self-contained record a run produces.
"""
