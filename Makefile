# make sure this is executed with bash
SHELL := /bin/bash


YELLOW := "\e[1;33m"
NC := "\e[0m"

# Logger function
INFO := @bash -c '\
  printf $(YELLOW); \
  echo "=> $$1"; \
  printf $(NC)' SOME_VALUE

.venv:  # creates .venv folder if does not exist
	python3.10 -m venv .venv


.venv/bin/uv: .venv # installs latest pip
	.venv/bin/pip install -U uv

install: .venv/bin/uv
	# before running install cmake
	.venv/bin/python3 -m uv pip install -r requirements.txt
	# after installing source .venv/bin/activate in your shell

download_data_from_s3:
	.venv/bin/python3 -m download_data