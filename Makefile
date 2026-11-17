PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help install-dev test test-core test-integration test-quantization lint lint-all build benchmark-help clean

help:
	@printf '%s\n' \
	  'install-dev       Install project plus development, quantization, and benchmark extras' \
	  'test              Run the full CPU pytest suite' \
	  'test-core         Run CLI, config, pipeline, and registry tests' \
	  'test-integration  Run model load/edit/export/reload integration tests' \
	  'test-quantization Run quantization compatibility and integration tests' \
	  'lint              Lint run-plan, quantization, and advanced stage code/tests' \
	  'lint-all          Run Ruff over all active source, tests, and benchmarks' \
	  'build             Build wheel and source distribution' \
	  'benchmark-help    Validate benchmark CLI argument wiring without downloading models' \
	  'clean             Remove build and test cache products'

install-dev:
	$(PIP) install -e '.[dev,quantization,benchmark]'

test:
	PYTHONPATH=src $(PYTEST) -q

test-core:
	PYTHONPATH=src $(PYTEST) -q tests/test_cli_validation.py tests/test_run_config.py tests/test_pipeline_edits_and_export.py tests/test_toggle_registry.py tests/test_orchestrator.py tests/test_baseline_regression_audit.py

test-integration:
	PYTHONPATH=src $(PYTEST) -q tests/test_quantization_integration.py

test-quantization:
	PYTHONPATH=src $(PYTEST) -q tests/test_quantization_compat.py tests/test_quantization_integration.py

lint:
	$(PYTHON) -m ruff check src/autobliteratus/run_config.py src/autobliteratus/quantization_compat.py src/autobliteratus/pipeline.py src/autobliteratus/__main__.py src/autobliteratus/tuning/_07_baseline_auto_tuning_pre_post.py src/autobliteratus/tuning/_09_multi_objective_auto_tuning.py src/autobliteratus/tuning/_10_auto_security_scanning.py src/autobliteratus/tuning/_16_anti_router_collapse_auto_tuning.py src/autobliteratus/tuning/_81_benchmark_regression_auto_tuning.py src/autobliteratus/tuning/_89_audit_trail_auto_tuning.py src/autobliteratus/tuning/evaluation_metrics.py src/autobliteratus/tuning/tuning_orchestrator.py tests/test_run_config.py tests/test_quantization_compat.py tests/test_callable_advanced_stages.py tests/test_baseline_regression_audit.py tests/test_toggle_registry.py tests/test_pipeline_edits_and_export.py

lint-all:
	$(PYTHON) -m ruff check src tests benchmarks

build:
	$(PYTHON) -m build

benchmark-help:
	PYTHONPATH=src $(PYTHON) benchmarks/quantization_quality_matrix.py --help

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find src tests benchmarks -type d -name __pycache__ -prune -exec rm -rf {} +
