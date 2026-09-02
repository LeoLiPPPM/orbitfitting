# OrbitFitting

OrbitFitting is a small astrodynamics toolkit that propagates spacecraft states,
composes force models, detects terminal events, and checks numerical accuracy
against conserved quantities. The included demo designs and propagates a
low-Earth-orbit Hohmann transfer.

## Run

From the portfolio root:

```bash
PYTHONPATH=orbitfitting python -m orbitfitting.demo --output orbitfitting/outputs
python -m unittest discover -s orbitfitting/tests -v
```

The source uses kilometers, seconds, and kilograms, which is the common unit
choice in preliminary astrodynamics work. Public functions include type hints
and docstrings. The force-model interface makes it possible to add thrust,
third-body gravity, or higher-order geopotential terms without changing the
integrator.
