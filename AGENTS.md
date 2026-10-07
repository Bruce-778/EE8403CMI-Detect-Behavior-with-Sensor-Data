# Project environment

Use the existing Conda environment `cmi` for all Python commands in this project.
Its interpreter on this machine is `D:\anaconda\envs\cmi\python.exe`.
Prefer `conda run -n cmi python ...` to activate the environment for each command.
In PowerShell, an alternative is `& 'D:\anaconda\envs\cmi\python.exe' -s ...`.
Do not use the project `.venv` or Conda `base`.
The environment sets `PYTHONNOUSERSITE=1` to exclude unrelated user-installed packages;
keep this isolation when invoking the interpreter directly by using `-s`.

`environment.yml` records the core dependency versions used for the Conda migration.
`requirements.txt` records the project's general dependency requirements.

Run the existing checks with:

```powershell
conda run --no-capture-output -n cmi python -m unittest discover -s tests -v
```
