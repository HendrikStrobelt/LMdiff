
test:
	python -m pytest tests

tt:
	python -m pytest tests -m now

clean:
	rm -rf data/ backend/*.egg-info .pytest_cache