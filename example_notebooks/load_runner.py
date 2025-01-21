from pcntoolkit.util.runner import Runner

runner = Runner.load("/project/3022000.05/projects/stijdboe/Projects/PCNtoolkit/example_notebooks/resources/hbr_runner_sandbox/temp_dir")
runner.re_submit_failed_jobs(observe=False)
