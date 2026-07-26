# SPDX-License-Identifier: MIT
# Makefile for Ornith 35B MoE Inference Engine (Phase 1 — C Implementation)
#
# Targets:
#   make              — build 'ornith' CLI (Metal on macOS, CPU-fallback elsewhere)
#   make metal        — build 'ornith' CLI (Metal GPU backend on macOS)
#   make cpu          — build 'ornith' CLI (CPU-fallback GPU backend)
#   make test         — build and run all test executables (CPU fallback)
#   make test/TEST    — run a single test (e.g. make test/t_tokenizer)
#   make test-metal   — build and run Metal shader tests
#   make clean        — remove build artifacts
#   make format       — run clang-format on all C sources (if available)

CC       = clang
CFLAGS   = -Wall -Wextra -O2 -g -Isrc -Itest
LDFLAGS  = -lm

SRC_DIR  = src
TEST_DIR = test
BLD_DIR  = build

# Ensure build directory exists at Makefile parse time
$(shell mkdir -p $(BLD_DIR))

# ── GPU backend selection ─────────────────────────────────────────────────
# Phase G3: Auto-detect macOS → default to Metal backend.
# On macOS: `make` uses Metal. `make cpu` forces CPU-fallback.
# On Linux/other: `make` uses CPU-fallback (Metal unavailable).

UNAME_S := $(shell uname -s 2>/dev/null || echo "unknown")

# Metal backend (macOS, requires Xcode)
# gpu_metal.m uses runtime MSL compilation via newLibraryWithSource:.
# No Xcode metal toolchain needed for development — shaders compiled at init.

METAL_OBJ = $(BLD_DIR)/gpu_metal.o
METAL_CFLAGS = -ObjC -fobjc-arc
METAL_LDFLAGS = -framework Metal -framework MetalPerformanceShaders -lobjc

# ── Source modules ────────────────────────────────────────────────────────

CORE_SRCS = $(SRC_DIR)/cli.c \
            $(SRC_DIR)/main.c \
            $(SRC_DIR)/model.c \
            $(SRC_DIR)/inference.c \
            $(SRC_DIR)/gguf.c \
            $(SRC_DIR)/memory.c \
            $(SRC_DIR)/tokenizer.c

CORE_OBJS = $(CORE_SRCS:$(SRC_DIR)/%.c=$(BLD_DIR)/%.o)

# Main binary
TARGET   = ornith

# Test sources — each defines its own main() via the RUN_TESTS macro
TEST_SRCS = $(wildcard $(TEST_DIR)/t_*.c)
TEST_BINS = $(TEST_SRCS:$(TEST_DIR)/%.c=$(BLD_DIR)/%)

.PHONY: all metal cpu test test-metal clean format

# ── Default target: Metal on macOS, CPU-fallback elsewhere ────────────────

# On macOS: default to Metal backend
ifeq ($(UNAME_S),Darwin)
all: | $(BLD_DIR)
all: $(CORE_OBJS) $(METAL_OBJ)
	$(CC) $(CFLAGS) $(METAL_CFLAGS) -DMETAL_ENABLED -o $(TARGET) $(CORE_OBJS) $(METAL_OBJ) $(LDFLAGS) $(METAL_LDFLAGS)
	@echo "  → Built $(TARGET) (Metal backend — default on macOS)"
else
all: $(BLD_DIR)/gpu.o $(TARGET)
endif

# ── Explicit CPU-fallback target ─────────────────────────────────────────

cpu: $(BLD_DIR)/gpu.o $(TARGET)
	@echo "  → Built $(TARGET) (CPU-fallback backend, explicit)"

# ── Explicit Metal target (works on macOS only) ──────────────────────────

metal: CFLAGS += -DMETAL_ENABLED $(METAL_CFLAGS)
metal: LDFLAGS += $(METAL_LDFLAGS)
metal: $(CORE_OBJS) $(METAL_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(CORE_OBJS) $(METAL_OBJ) $(LDFLAGS)
	@echo "  → Built $(TARGET) (Metal backend)"

# ── Metal GPU shader test target ──────────────────────────────────────────
# Builds and runs t_gpu_metal using the Metal backend directly.

test-metal: CFLAGS += -DMETAL_ENABLED $(METAL_CFLAGS)
test-metal: LDFLAGS += $(METAL_LDFLAGS)
test-metal: $(BLD_DIR) $(METAL_OBJ)
	@echo ""
	@echo "═══ Building Metal GPU shader test ═══"
	# Only need: test framework, t_gpu_metal.c, gpu_metal.m, and math lib
	$(CC) $(CFLAGS) -I$(TEST_DIR) -o $(BLD_DIR)/t_gpu_metal \
	    $(TEST_DIR)/t_gpu_metal.c $(METAL_OBJ) $(LDFLAGS)
	@echo ""
	@echo "═══ Running Metal GPU shader test ═══"
	@$(BLD_DIR)/t_gpu_metal
	@echo ""
	@echo "═══ Done ═══"

# Must include all .metal files as dependencies for gpu_metal.m
METAL_SHADERS = $(wildcard $(SRC_DIR)/metal/*.metal)

$(METAL_OBJ): $(SRC_DIR)/gpu_metal.m $(METAL_SHADERS)
	@mkdir -p $(BLD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Link ──────────────────────────────────────────────────────────────────

# CPU build: links gpu.o; Metal build: links gpu_metal.o instead
$(TARGET): $(CORE_OBJS) | $(BLD_DIR)
ifeq ($(MAKECMDGOALS),metal)
	$(CC) $(CFLAGS) -o $@ $(CORE_OBJS) $(METAL_OBJ) $(LDFLAGS)
	@echo "  → Built $(TARGET) (Metal backend)"
else
	$(CC) $(CFLAGS) -o $@ $(CORE_OBJS) $(BLD_DIR)/gpu.o $(LDFLAGS)
	@echo "  → Built $(TARGET) (CPU-fallback backend)"
endif

# Note: For Metal builds, model.c still needs gpu context which is now in gpu_metal.o,
# but model.c only uses the gpu.h API, so it links fine against either backend.

# ── Compilation ───────────────────────────────────────────────────────────

$(BLD_DIR)/%.o: $(SRC_DIR)/%.c | $(BLD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BLD_DIR)/gpu.o: $(SRC_DIR)/gpu.c | $(BLD_DIR)
	$(CC) $(filter-out -DMETAL_ENABLED, $(CFLAGS)) -c -o $@ $<

$(BLD_DIR):
	mkdir -p $(BLD_DIR)

# ── Test targets ──────────────────────────────────────────────────────────

# Tests always use the CPU-fallback GPU backend (gpu.c)
test: CFLAGS += $(GPU_CFLAGS)
test: LDFLAGS += $(GPU_LDFLAGS)
test: $(BLD_DIR)/gpu.o $(TEST_BINS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Running CPU-backend tests…"
	@echo "═══════════════════════════════════════════════════════════════"
	@total=0; passed=0; failed=0; \
	for t in $(TEST_BINS); do \
		total=$$((total + 1)); \
		name=$$(basename $$t); \
		echo ""; \
		echo "─── $$name ───"; \
		$$t; \
		if [ $$? -eq 0 ]; then passed=$$((passed + 1)); else failed=$$((failed + 1)); fi; \
	done; \
	echo ""; \
	echo "═══════════════════════════════════════════════════════════════"; \
	echo "  CPU: $$passed/$$total passed, $$failed failed"; \
	echo "═══════════════════════════════════════════════════════════════"
	@if [ $$failed -eq 0 ]; then \
		echo ""; \
		echo "═══════════════════════════════════════════════════════════════"; \
		echo "  Attempting Metal GPU backend tests…"; \
		echo "═══════════════════════════════════════════════════════════════"; \
		$(MAKE) test-metal 2>/dev/null || \
		echo "  → Metal backend not available (requires macOS + Xcode); skipping"; \
	fi

# Build each test executable — link all core .o files EXCEPT main.o (tests have own main)
$(BLD_DIR)/t_%: $(TEST_DIR)/t_%.c $(filter-out $(BLD_DIR)/main.o, $(CORE_OBJS)) $(BLD_DIR)/gpu.o | $(BLD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(filter-out $(BLD_DIR)/main.o, $(CORE_OBJS)) $(BLD_DIR)/gpu.o $(LDFLAGS)

# Convenience: run a single test by name (e.g. make test/t_tokenizer)
test/%: $(BLD_DIR)/%
	@$<

# ── Clean ─────────────────────────────────────────────────────────────────

clean:
	rm -rf $(BLD_DIR) $(TARGET)
	@echo "  Cleaned build artifacts"

# ── Format (optional) ──────────────────────────────────────────────────────

format:
	clang-format -i $(SRC_DIR)/*.c $(SRC_DIR)/*.h $(TEST_DIR)/*.c $(TEST_DIR)/*.h 2>/dev/null || \
	  echo "clang-format not available; skipping"
