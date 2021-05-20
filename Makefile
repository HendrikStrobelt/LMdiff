
test:
	python -m pytest tests

clean:
	rm -rf data/ backend/*.egg-info .pytest_cache