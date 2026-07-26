# SPDX-License-Identifier: MIT
# Ornith Flight — Top-Level Makefile
#
# Targets:
#   make            — build ornith CLI (Metal on macOS, CPU-fallback elsewhere)
#   make cpu        — build with CPU-fallback GPU backend
#   make metal      — build with Metal GPU backend (macOS only)
#   make test       — build and run all tests
#   make test-metal — build and run Metal shader tests (macOS only)
#   make clean      — remove build artifacts
#   make format     — run clang-format on C sources

.PHONY: all cpu metal test test-metal clean format

all:
	$(MAKE) -C engine all

cpu:
	$(MAKE) -C engine cpu

metal:
	$(MAKE) -C engine metal

test:
	$(MAKE) -C engine test

test-metal:
	$(MAKE) -C engine test-metal

clean:
	$(MAKE) -C engine clean

format:
	$(MAKE) -C engine format
