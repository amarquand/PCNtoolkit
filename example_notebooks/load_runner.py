from pcntoolkit.util.runner import Runner

runner = Runner.load("/Users/stijndeboer/Projects/PCN/PCNtoolkit/example_notebooks/resources/hbr_runner_sandbox/temp_dir")
runner.re_submit_failed_jobs()
