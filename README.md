# optimistix-migrad
Trying to implement MIGRAD in optimistix...


## Run examples:
```bash
uv sync

uv run test.py
uv run pyhf_test.py
```

`test.py` is a general suite of comparisons between `iminuit` and _this_ MIGRAD implementation for various [test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

`pyhf_test.py` is a comparison between `iminuit` and _this_ MIGRAD implementation for a `pyhf` likelihood function based on [this example tutorial](https://github.com/cabinetry/cabinetry-tutorials/blob/master/HEPData_workspace.ipynb).
