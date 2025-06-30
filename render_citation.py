from datetime import date

import toml
from jinja2 import Template

# Load version from pyproject.toml
pyproject = toml.load("pyproject.toml")
meta = pyproject["project"]
version = meta["version"]

# Fill in other metadata
context = {
    "version": version,
    "date": date.today().isoformat(),
    "doi": "10.5281/zenodo.5207839",
}

# Load and render template
with open("CITATION.cff.in") as f:
    template = Template(f.read())

with open("CITATION.cff", "w") as f:
    f.write(template.render(**context))

print("CITATION.cff generated.")
