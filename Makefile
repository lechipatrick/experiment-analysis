SHELL:=/usr/bin/env bash
PROJECT=experiment-analysis
PYTHON_VERSION=3.9.13

SOURCE_OBJECTS=experiment_analysis tests

setup: setup.sysdeps setup.python setup.project
setup.uninstall:
	poetry env remove ${PYTHON_VERSION} || true
setup.ci: setup.ci.poetry setup.project
setup.ci.poetry:
	pip install --user poetry
setup.project:
	@poetry env use $$(python -c "import sys; print(sys.executable)")
	@echo "interpreter path: $$(poetry env info --path)/bin/python"
	poetry install
setup.python.activation:
	@asdf local python ${PYTHON_VERSION} >/dev/null 2>&1 /dev/null || true
setup.python: setup.python.activation
	@echo "current python version: $$(python --version)"
	@echo "interpreter path: $$(python -c 'import sys; print(sys.executable)')"
	@test "$$(python --version | cut -d' ' -f2)" = "${PYTHON_VERSION}" \
        || (echo "need to activate python ${PYTHON_VERSION}" && exit 1)
setup.sysdeps:
	asdf plugin update --all
	@for p in $$(cut -d" " -f1 .tool-versions | sort | tr '\n' ' '); do \
        asdf plugin add $$p || true; \
        done;
	asdf install

format.dev:
	poetry run black ${DEV_OBJECTS}
	poetry run isort --atomic ${DEV_OBJECTS}

format: format.isort format.black
format.black:
	poetry run black ${SOURCE_OBJECTS}
format.isort:
	poetry run isort --atomic ${SOURCE_OBJECTS}

lints: lints.flake8 lints.mypy lints.pylint
lints.format.check: lints.flake8
	poetry run black --check ${SOURCE_OBJECTS}
	poetry run isort --check-only ${SOURCE_OBJECTS}
lints.flake8:
	poetry run flake8 ${SOURCE_OBJECTS}
lints.mypy:
	poetry run mypy ${SOURCE_OBJECTS}
lints.pylint:
	poetry run pylint --rcfile pyproject.toml ${SOURCE_OBJECTS}

test:
	pytest

