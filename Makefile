build:
	python setup.py build

clean-pyext:
	rm -rf build

clean-pycache:
	rm -rf mytensor_pkg/__pycache__

clean: clean-pyext clean-pycache

install:
	python setup.py install

.PHONY: build clean install
