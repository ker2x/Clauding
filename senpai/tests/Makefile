PYTHON   := ../../.venv/bin/python
RUN      := $(PYTHON) ../scripts/run.py
BUILD    := .build
CASES    := cases

# Discover tests
E2E_SEN    := $(wildcard $(CASES)/e2e/*.sen)
E2E_TESTS  := $(patsubst $(CASES)/e2e/%.sen,%,$(E2E_SEN))

ERROR_SEN  := $(wildcard $(CASES)/error/*.sen)
ERROR_TESTS := $(patsubst $(CASES)/error/%.sen,%,$(ERROR_SEN))

# Module tests: directories under cases/modules/ that contain main.sen
MOD_DIRS    := $(wildcard $(CASES)/modules/*/main.sen)
MOD_TESTS   := $(patsubst $(CASES)/modules/%/main.sen,%,$(MOD_DIRS))

# Result files
E2E_RESULTS   := $(patsubst %,$(BUILD)/e2e/%.result,$(E2E_TESTS))
ERROR_RESULTS := $(patsubst %,$(BUILD)/error/%.result,$(ERROR_TESTS))
MOD_RESULTS   := $(patsubst %,$(BUILD)/module/%.result,$(MOD_TESTS))
ALL_RESULTS   := $(E2E_RESULTS) $(ERROR_RESULTS) $(MOD_RESULTS)

.PHONY: test clean

test: $(ALL_RESULTS)
	@passed=0; failed=0; \
	for f in $(ALL_RESULTS); do \
		status=$$(cat "$$f"); \
		name=$$(basename "$$f" .result); \
		kind=$$(basename $$(dirname "$$f")); \
		if [ "$$status" = "PASS" ]; then \
			printf "  PASS  %s: %s\n" "$$kind" "$$name"; \
			passed=$$((passed + 1)); \
		else \
			printf "  FAIL  %s: %s: %s\n" "$$kind" "$$name" "$$status"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo; echo "========================================"; \
	printf "  %d passed, %d failed\n" "$$passed" "$$failed"; \
	echo "========================================"; \
	[ "$$failed" -eq 0 ]

# --- e2e tests: compile, run, compare output ---
$(BUILD)/e2e/%.result: $(CASES)/e2e/%.sen $(CASES)/e2e/%.expected
	@mkdir -p $(dir $@)
	@expected=$$(cat $(word 2,$^)); \
	if actual=$$($(RUN) $< 2>&1) && actual=$$(printf '%s' "$$actual"); then \
		if [ "$$actual" = "$$expected" ]; then \
			echo "PASS" > $@; \
		else \
			printf 'expected:\n%s\ngot:\n%s' "$$expected" "$$actual" > $@; \
		fi; \
	else \
		printf 'runtime error: %s' "$$actual" > $@; \
	fi

# --- error tests: compile must fail, stderr must contain fragment ---
$(BUILD)/error/%.result: $(CASES)/error/%.sen $(CASES)/error/%.error
	@mkdir -p $(dir $@)
	@fragment=$$(cat $(word 2,$^)); \
	if output=$$($(RUN) $< 2>&1); then \
		printf 'expected compile error containing "%s", but compilation succeeded' "$$fragment" > $@; \
	else \
		if printf '%s' "$$output" | grep -q "$$fragment"; then \
			echo "PASS" > $@; \
		else \
			printf 'expected "%s" in error, got: %s' "$$fragment" "$$output" > $@; \
		fi; \
	fi

# --- module tests (with .expected): compile, run, compare ---
define MODULE_E2E_RULE
$(BUILD)/module/$(1).result: $(CASES)/modules/$(1)/main.sen $(CASES)/modules/$(1)/main.expected
	@mkdir -p $$(dir $$@)
	@expected=$$$$(cat $$(word 2,$$^)); \
	if actual=$$$$($(RUN) $$< 2>&1) && actual=$$$$(printf '%s' "$$$$actual"); then \
		if [ "$$$$actual" = "$$$$expected" ]; then \
			echo "PASS" > $$@; \
		else \
			printf 'expected:\n%s\ngot:\n%s' "$$$$expected" "$$$$actual" > $$@; \
		fi; \
	else \
		printf 'runtime error: %s' "$$$$actual" > $$@; \
	fi
endef

# --- module tests (with .error): compile must fail ---
define MODULE_ERROR_RULE
$(BUILD)/module/$(1).result: $(CASES)/modules/$(1)/main.sen $(CASES)/modules/$(1)/main.error
	@mkdir -p $$(dir $$@)
	@fragment=$$$$(cat $$(word 2,$$^)); \
	if output=$$$$($(RUN) $$< 2>&1); then \
		printf 'expected compile error containing "%s", but compilation succeeded' "$$$$fragment" > $$@; \
	else \
		if printf '%s' "$$$$output" | grep -q "$$$$fragment"; then \
			echo "PASS" > $$@; \
		else \
			printf 'expected "%s" in error, got: %s' "$$$$fragment" "$$$$output" > $$@; \
		fi; \
	fi
endef

# Generate rules for each module test
$(foreach m,$(MOD_TESTS),\
  $(if $(wildcard $(CASES)/modules/$(m)/main.error),\
    $(eval $(call MODULE_ERROR_RULE,$(m))),\
    $(eval $(call MODULE_E2E_RULE,$(m)))))

clean:
	rm -rf $(BUILD)
