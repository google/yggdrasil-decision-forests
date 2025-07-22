
# Makefile for Yggdrasil Decision Forests
#
# This Makefile provides a simplified interface for building and testing the project
# using Bazel. It is intended for beginners and for common development tasks.

# Default Bazel command
BAZEL ?= bazel

# Default Bazel options
# For maximum performance, uncomment the following line:
# BAZEL_OPTS = --copt=-mfma --copt=-mavx2 --copt=-mavx
BAZEL_OPTS ?= --copt=-mavx2

# --- Build Targets ---

.PHONY: all
all: build

# Build the C++ library and examples
.PHONY: build
build:
	@echo "Building Yggdrasil Decision Forests..."
	$(BAZEL) build $(BAZEL_OPTS) //yggdrasil_decision_forests/...

# --- Testing ---

# Run all tests
.PHONY: test
test:
	@echo "Running all tests..."
	$(BAZEL) test $(BAZEL_OPTS) --test_output=errors //yggdrasil_decision_forests/...

# --- Examples ---

# Build and run the main beginner example
.PHONY: run_example
run_example:
	@echo "Building and running the beginner example..."
	$(BAZEL) run $(BAZEL_OPTS) //yggdrasil_decision_forests/examples:beginner

# Build and run the standalone beginner example
.PHONY: run_standalone_example
run_standalone_example:
	@echo "Building and running the standalone beginner example..."
	$(BAZEL) run $(BAZEL_OPTS) //examples/standalone:beginner

# --- Release ---

# Build the binary release
.PHONY: release
release:
	@echo "Building the binary release..."
	./tools/build_binary_release.sh

# --- Cleaning ---

# Clean the Bazel cache
.PHONY: clean
clean:
	@echo "Cleaning the Bazel cache..."
	$(BAZEL) clean

# --- Help ---

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all                      Build the C++ library and examples (default)"
	@echo "  build                    Alias for 'all'"
	@echo "  test                     Run all tests"
	@echo "  run_example              Build and run the main beginner example"
	@echo "  run_standalone_example   Build and run the standalone beginner example"
	@echo "  release                  Build the binary release"
	@echo "  clean                    Clean the Bazel cache"
	@echo "  help                     Show this help message"

