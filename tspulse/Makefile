# Builds the program and the weight check for the host (`make`), for rv64gc (`make TARGET=rv64`) or for rv32gc (`make TARGET=rv32`), both scalar
# `tspulse-dump` is the program with the code that writes inner outputs, which `check-model` runs, and `tspulse` carries none of it
# `tspulse-profile`, built for rv64 only, reads the instruction counter at the same hooks to count instructions per module
# `make stack` compiles the program with `-fcallgraph-info=su`, which leaves a `.ci` file of stack frame sizes beside each object, and links nothing; `make TARGET=rv64` also builds them
# `make check` runs the weight check and compares its dump with the one export-weights wrote from the safetensors file, weights_int8.tdump on a QUANT build
# `make check-qlinear` (QUANT builds only) builds `check-qlinear` and runs it, on the RISC-V builds under Spike: the build's quantized Linear against a reference on the same weights, bit for bit
# `make approx-errors` (host only) builds `check-approx` and writes approx-errors.csv in the build's folder, as in build/host/approx-errors.csv, each exp and GELU approximation's error against double precision
# Run inside the dev shell, after `download-weights`
# Settings are Make variables listed in variants.mk, and `make settings` prints them with their values and defaults; `make LINEAR=blocked` would pick one
# A build that sets nothing is the baseline, in build/<target>; any other lands in build/<name>/<target>, named after its non-default settings, as in build/linear-blocked+unroll-on/host
# Every build writes a build.json beside its binaries, recording its settings, compiler, flags, files, commit and the binaries' hashes

# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀
# ░█░░░█░█░█░█░█▀▀░░█░░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀

TARGET ?= host
include variants.mk
BUILD := $(if $(filter baseline,$(VARIANT_NAME)),build/$(TARGET),build/$(VARIANT_NAME)/$(TARGET))
GENERATED := build/generated

# Fusing, vectorizing and the C standard change float results, so the host build is pinned to what rv64 does
# -std=c11 turns fusing off silently, so no -std flag is given
ifeq ($(TARGET),host)
CC := gcc
ARCH_FLAGS := -O3 -mfma --param avoid-fma-max-bits=0 -fno-tree-vectorize
else ifeq ($(TARGET),rv64)
CC := riscv64-none-elf-gcc
ARCH_FLAGS := -march=rv64gc -mabi=lp64d -O3
SPIKE_ISA := rv64gc_zicntr
SPIKE_PK := $$RISCV_PK
else ifeq ($(TARGET),rv32)
CC := riscv32-none-elf-gcc
# Same as rv64gc with only the width changed, the one library set the rv32 toolchain ships
ARCH_FLAGS := -march=rv32gc -mabi=ilp32d -O3
SPIKE_ISA := rv32gc_zicntr
SPIKE_PK := $$RISCV32_PK
else
$(error unknown TARGET '$(TARGET)', expected host, rv64 or rv32)
endif

# A command-line CFLAGS replaces the settings' flags and revin's -ffp-contract=off, so build.json would record flags the build never used
$(if $(filter command line,$(origin CFLAGS)),$(error Set ARCH_FLAGS instead of CFLAGS, so the settings' flags and revin's flags still apply))

CPPFLAGS += -I. -MMD -MP $(VARIANT_DEFINES)
CFLAGS += $(ARCH_FLAGS) -Wall -Wextra $(VARIANT_FLAGS)
LDLIBS := -lm

# PyTorch's RevIN does each step as its own tensor operation, in both directions, rounding after each, so the whole file is built without fused multiply-add
# Fused, more scaled values land one ulp away from PyTorch's, and the alternating sum behind the FFT's base component (bin 256) magnifies those differences past its allowance on YAHOO w908
REVIN_FILE := kernels/revin.c
REVIN_FLAGS := -ffp-contract=off

# The host build always compiles newlib's erff and expf, because glibc's round differently and the host must match rv64 bit for bit
# The RISC-V builds compile them when a setting asks, to take fewer instructions per call than the toolchain's libm.a, which holds the same code
# Without __OBSOLETE_MATH=0, x86 takes newlib's old code path: sf_exp.c compiles to nothing and sf_erf.o needs __ieee754_expf
# Only the machine/ folder is added to the include path, since newlib's whole libc/include would replace glibc's math.h
# The RISC-V builds get the same flags, which change nothing there: that machine/ieeefp.h is byte-identical to both cross toolchains', and rv64gc and rv32gc already default to __OBSOLETE_MATH=0
NEWLIB_FILES := libm/math/sf_erf.c libm/common/sf_exp.c libm/common/sf_exp2_data.c libm/common/math_errf.c libm/common/math_err.c
NEWLIB_FLAGS = -D__OBSOLETE_MATH=0 -I$(NEWLIB_HOST_INCLUDE) -I$(NEWLIB_SRC)/newlib/libm/common

# Non-empty when this build compiles newlib's math: always on the host, and on the RISC-V builds when a setting asks
NEWLIB_LINKED := $(strip $(filter host,$(TARGET)) $(VARIANT_NEWLIB))

# ░█▀▀░█▀█░█░█░█▀▄░█▀▀░█▀▀░█▀▀
# ░▀▀█░█░█░█░█░█▀▄░█░░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

SOURCES := io/tdump.c checks/weights.c
OBJECTS := $(SOURCES:%.c=$(BUILD)/%.o)

# The dump and profile builds define TSPULSE_DUMP or TSPULSE_PROFILE and keep their objects apart, so they never mix with the clean one
KERNEL_SOURCES := kernels/add.c kernels/fft.c kernels/gelu.c kernels/layer_norm.c kernels/linear/plain.c kernels/multiply.c kernels/revin.c kernels/softmax.c kernels/sum.c kernels/transpose.c
MODULE_SOURCES := modules/backbone.c modules/block.c modules/decoder.c modules/embedding.c modules/heads.c modules/mixer.c
PROGRAM_SOURCES := $(call variant_sources,tspulse.c io/rows.c model/history.c model/pass.c model/window.c $(KERNEL_SOURCES) $(MODULE_SOURCES))
PROGRAM_OBJECTS := $(PROGRAM_SOURCES:%.c=$(BUILD)/%.o)
DUMP_OBJECTS := $(PROGRAM_SOURCES:%.c=$(BUILD)/dump/%.o) $(BUILD)/dump/io/tdump.o $(BUILD)/dump/io/dump.o

# The profile build keeps its own objects too, and has no host form, because it reads a RISC-V counter
PROFILE_OBJECTS := $(PROGRAM_SOURCES:%.c=$(BUILD)/profile/%.o) $(BUILD)/profile/io/profile.o

# The stack build is the clean program's objects with the call graph written beside them, so the frame sizes are those of the real build
STACK_OBJECTS := $(PROGRAM_SOURCES:%.c=$(BUILD)/stack/%.o)

# The approximation check holds every value's header and both tables, whatever the settings, and is no part of BINARIES or all
# Its objects leave out those the program already holds, so a table build doesn't name kernels/*/table.o twice
APPROX_SOURCES := checks/approx.c kernels/exp/table.c kernels/gelu/table.c
APPROX_OBJECTS := $(APPROX_SOURCES:%.c=$(BUILD)/%.o)
APPROX_NEW_OBJECTS := $(filter-out $(PROGRAM_OBJECTS),$(APPROX_OBJECTS))

# The quantized Linear check links the build's QUANT kernel and blocked.c, never plain.c, whatever LINEAR says, since the w8a32 check compares against blocked.c's result
# Like the approximation check, it is no part of BINARIES or all, and leaves out the objects the program already holds
QLINEAR_SOURCES := checks/qlinear.c $(QUANT_SOURCES_$(QUANT)) kernels/linear/blocked.c
QLINEAR_OBJECTS := $(QLINEAR_SOURCES:%.c=$(BUILD)/%.o)
QLINEAR_NEW_OBJECTS := $(filter-out $(PROGRAM_OBJECTS),$(QLINEAR_OBJECTS))

ifneq ($(NEWLIB_LINKED),)
NEWLIB_OBJECTS := $(NEWLIB_FILES:%.c=$(BUILD)/newlib/%.o)
endif
ifneq ($(TARGET),host)
# Under pk the program carries its own _open, because pk reads open flags as Linux numbers
PLATFORM_OBJECTS := $(BUILD)/io/pk_open.o
endif
# A setting's WEIGHTS entry picks the generated set, as in weights_int8.c, weights_int8_table.c and weights_int8.tdump
WEIGHTS_STEM := weights$(if $(VARIANT_WEIGHTS),_$(VARIANT_WEIGHTS))
GENERATED_SOURCES := $(GENERATED)/$(WEIGHTS_STEM).c $(GENERATED)/$(WEIGHTS_STEM)_table.c
GENERATED_OBJECTS := $(GENERATED_SOURCES:$(GENERATED)/%.c=$(BUILD)/generated/%.o)
GENERATED_DUMP := $(GENERATED)/$(WEIGHTS_STEM).tdump

# The program needs only the weights, while the weight check also needs their table of names
WEIGHTS_OBJECT := $(BUILD)/generated/$(WEIGHTS_STEM).o

# What the build compiles and links, fixed when the Makefile is read
# A recipe's $(CFLAGS) can carry the target-specific -ffp-contract=off of the object it belongs to, so the stamp and build.json never read it there
# The compiler's resolved path is in the record, so a new compiler behind the same name rebuilds the objects
# The newlib files are in it because a setting can add them without changing any flag; without them, one folder built with and then without them keeps the old binary
# It is exported so the recipe reads it from the environment and quotes it, since the record's paths can hold spaces
export FLAGS_RECORD := $(CC) $(shell command -v $(CC)) | $(CPPFLAGS) | $(CFLAGS) | $(LDLIBS) | $(REVIN_FILE) $(REVIN_FLAGS) | $(NEWLIB_FLAGS) | $(PROGRAM_SOURCES) | $(NEWLIB_OBJECTS:$(BUILD)/newlib/%.o=%.c)
RECORDED_CPPFLAGS := $(strip $(filter-out -I%,$(CPPFLAGS)))
RECORDED_CFLAGS := $(strip $(CFLAGS))
RECORDED_LDLIBS := $(LDLIBS)
OVERRIDDEN := $(foreach v,CC ARCH_FLAGS BUILD GENERATED,$(if $(filter command line,$(origin $(v))),$(v)))

# ░█▀▀░█▀▀░█▀█░█▀▀░█▀▄░█▀█░▀█▀░█▀█░█▀▄
# ░█░█░█▀▀░█░█░█▀▀░█▀▄░█▀█░░█░░█░█░█▀▄
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀░▀

ifneq ($(MAKECMDGOALS),clean)

$(if $(TSPULSE_DATA),,$(error TSPULSE_DATA is not set; run make inside nix develop))
SAFETENSORS := $(TSPULSE_DATA)/models/tspulse-r1/model.safetensors

# GNU Make splits prerequisites on spaces, and the data folder's path may have some
space := $(subst ,, )
SAFETENSORS_PREREQ := $(subst $(space),\ ,$(SAFETENSORS))

GENERATOR := $(shell command -v export-weights)
$(if $(GENERATOR),,$(error export-weights not found; run make inside nix develop))

ifneq ($(NEWLIB_LINKED),)
$(if $(NEWLIB_SRC),,$(error NEWLIB_SRC is not set; run make inside nix develop))
$(if $(NEWLIB_HOST_INCLUDE),,$(error NEWLIB_HOST_INCLUDE is not set; run make inside nix develop))
endif

endif

# ░▀█▀░█▀█░█▀▄░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀█░█▀▄░█░█░█▀▀░░█░░▀▀█
# ░░▀░░▀░▀░▀░▀░▀▀▀░▀▀▀░░▀░░▀▀▀

.PHONY: all approx-errors check check-qlinear clean settings stack FORCE

# The rv64 build keeps the call-graph objects fresh too, so the stack frames `benchmark profile` reads never lag behind the sources
ifeq ($(TARGET),rv64)
PROFILE_PROGRAM := $(BUILD)/tspulse-profile
STACK_TARGETS := $(STACK_OBJECTS)
endif

BINARIES := $(BUILD)/tspulse $(BUILD)/tspulse-dump $(PROFILE_PROGRAM) $(BUILD)/check-weights

all: $(BINARIES) $(STACK_TARGETS) $(BUILD)/build.json

# Each setting with its allowed values and its default
settings:
	@printf '%-12s %-28s %s\n' SETTING VALUES DEFAULT; $(foreach s,$(SETTINGS),printf '%-12s %-28s %s\n' '$(s)' '$($(s)_VALUES)' '$($(s)_DEFAULT)';)

stack: $(STACK_OBJECTS)

# Run the check program: on the host directly, on the RISC-V builds under Spike
check: $(BUILD)/check-weights $(GENERATED_DUMP)
ifeq ($(TARGET),host)
	$(BUILD)/check-weights $(BUILD)/weights.tdump
else
	spike --isa=$(SPIKE_ISA) "$(SPIKE_PK)" $(BUILD)/check-weights $(BUILD)/weights.tdump
endif
# Pass only on 0 fail, 0 missing and 0 extra, because compare-reference exits 0 on missing records
	status=0; compare-reference $(GENERATED_DUMP) $(BUILD)/weights.tdump --exact > $(BUILD)/compare.txt || status=$$?; \
	grep -E '^[0-9]+ pass, ' $(BUILD)/compare.txt; \
	[ $$status -eq 0 ] && grep -Eq '^[0-9]+ pass, 0 fail, 0 missing, 0 extra$$' $(BUILD)/compare.txt

# Refused before anything builds: the check runs the host's binary directly
ifneq ($(filter approx-errors,$(MAKECMDGOALS)),)
$(if $(filter host,$(TARGET)),,$(error approx-errors runs on the host only, got TARGET=$(TARGET)))
endif

approx-errors: $(BUILD)/check-approx
	$(BUILD)/check-approx $(BUILD)/approx-errors.csv

# Refused before anything builds: QUANT=off has no quantized kernel to check
ifneq ($(filter check-qlinear,$(MAKECMDGOALS)),)
$(if $(filter $(QUANT_DEFAULT),$(QUANT)),$(error check-qlinear needs a QUANT build, got QUANT=$(QUANT)))
endif

# Run the quantized Linear check: on the host directly, on the RISC-V builds under Spike
check-qlinear: $(BUILD)/check-qlinear
ifeq ($(TARGET),host)
	$(BUILD)/check-qlinear
else
	spike --isa=$(SPIKE_ISA) "$(SPIKE_PK)" $(BUILD)/check-qlinear
endif

clean:
	rm -rf build

# ░█▀▄░█░█░▀█▀░█░░░█▀▄░█▀▀
# ░█▀▄░█░█░░█░░█░░░█░█░▀▀█
# ░▀▀░░▀▀▀░▀▀▀░▀▀▀░▀▀░░▀▀▀

$(BUILD)/check-weights: $(OBJECTS) $(GENERATED_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

# The newlib objects come before -lm, so the tables and the libm rows use the program's erff and expf, not glibc's
$(BUILD)/check-approx: $(APPROX_OBJECTS) $(NEWLIB_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Linked with $(LDLIBS) like the program, so any libm call a QUANT kernel makes resolves on the host
$(BUILD)/check-qlinear: $(QLINEAR_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# The newlib objects come before -lm, so the linker finds erff and expf in them first
$(BUILD)/tspulse: $(PROGRAM_OBJECTS) $(WEIGHTS_OBJECT) $(NEWLIB_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/tspulse-dump: $(DUMP_OBJECTS) $(WEIGHTS_OBJECT) $(NEWLIB_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/tspulse-profile: $(PROFILE_OBJECTS) $(WEIGHTS_OBJECT) $(NEWLIB_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# A member with transposed dimensions has the right size but only draws an excess-elements warning, so generated objects build with -Werror
$(GENERATED_OBJECTS): $(BUILD)/generated/%.o: $(GENERATED)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Werror -c $< -o $@

$(OBJECTS) $(APPROX_NEW_OBJECTS) $(QLINEAR_NEW_OBJECTS) $(PROGRAM_OBJECTS) $(PLATFORM_OBJECTS): $(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(foreach d,$(BUILD) $(BUILD)/dump $(BUILD)/profile $(BUILD)/stack,$(d)/$(REVIN_FILE:.c=.o)): CFLAGS += $(REVIN_FLAGS)

# Every object is rebuilt when the stamp of the build's flags and files changes
$(OBJECTS) $(APPROX_NEW_OBJECTS) $(QLINEAR_NEW_OBJECTS) $(PROGRAM_OBJECTS) $(DUMP_OBJECTS) $(PROFILE_OBJECTS) $(STACK_OBJECTS) $(NEWLIB_OBJECTS) $(PLATFORM_OBJECTS) $(GENERATED_OBJECTS): $(BUILD)/flags

$(DUMP_OBJECTS): $(BUILD)/dump/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DTSPULSE_DUMP -c $< -o $@

$(PROFILE_OBJECTS): $(BUILD)/profile/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DTSPULSE_PROFILE -c $< -o $@

$(STACK_OBJECTS): $(BUILD)/stack/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fcallgraph-info=su -c $< -o $@

$(NEWLIB_OBJECTS): $(BUILD)/newlib/%.o: $(NEWLIB_SRC)/newlib/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(NEWLIB_FLAGS) -c $< -o $@

# build.json is written by make itself, so no tool beyond the shell's sha256sum and git is needed
# A string is escaped for JSON by doubling backslashes and escaping quotes, which covers every character a flag or path can hold here
comma := ,
json_var = "$(subst ",\",$(subst \,\\,$($(1))))"

# Nested objects and lists are laid out like Python's `json.dumps(..., indent=2)`, one item per line, so the tools' copies of the record look the same
space := $(subst ,, )
indent := $(space)$(space)
define newline


endef

# `foreach` joins its results with a space, so each item opens with a marker: a space before a marker becomes the comma between items, and the first marker is dropped
json_marker := @@
json_item = $(json_marker)$(newline)$(indent)$(indent)
json_body = $(if $(strip $(1)),$(subst $(json_marker),,$(subst $(space)$(json_marker),$(comma),$(1)))$(newline)$(indent))

# The git fields are null outside a repository, and the dirty check leaves out the benchmark tables, which the tools write themselves
CC_VERSION = $(shell $(CC) --version 2>/dev/null | head -n 1)
GIT_COMMIT = $(shell git rev-parse HEAD 2>/dev/null)
GIT_TREE = $(shell git rev-parse 'HEAD^{tree}' 2>/dev/null)
GIT_DIRTY = $(shell git status --porcelain -- ':(top)' ':(top,exclude)benchmark/results' 2>/dev/null)

define BUILD_JSON
{
  "name": $(call json_var,VARIANT_NAME),
  "target": $(call json_var,TARGET),
  "settings": {$(call json_body,$(foreach s,$(SETTINGS),$(json_item)"$($(s)_NAME)": $(call json_var,$(s))))},
  "cc": $(call json_var,CC),
  "cc_version": $(call json_var,CC_VERSION),
  "cppflags": $(call json_var,RECORDED_CPPFLAGS),
  "cflags": $(call json_var,RECORDED_CFLAGS),
  "ldlibs": $(call json_var,RECORDED_LDLIBS),
  "file_flags": {$(call json_body,$(json_item)"$(REVIN_FILE)": $(call json_var,REVIN_FLAGS))},
  "sources": [$(call json_body,$(foreach f,$(PROGRAM_SOURCES),$(json_item)"$(f)"))],
  "overrides": {$(call json_body,$(foreach v,$(OVERRIDDEN),$(json_item)"$(v)": $(call json_var,$(v))))},
  "commit": $(if $(GIT_COMMIT),"$(GIT_COMMIT)",null),
  "tree": $(if $(GIT_TREE),"$(GIT_TREE)",null),
  "dirty": $(if $(GIT_COMMIT),$(if $(GIT_DIRTY),true,false),null),
  "sha256": {$(call json_body,$(foreach b,$(BINARIES),$(json_item)"$(notdir $(b))": "$(firstword $(shell sha256sum $(b)))"))}
}
endef

# Always rewritten, so the commit and the dirty flag are those of the moment the binaries were last checked
# The temporary file is renamed over it, so a half-written record never exists
$(BUILD)/build.json: $(BINARIES) $(BUILD)/flags FORCE
	@$(file >$@.tmp,$(BUILD_JSON))
	@mv $@.tmp $@

# ░█▀▀░█▀▀░█▀█░█▀▀░█▀▄░█▀█░▀█▀░█▀▀░█▀▄
# ░█░█░█▀▀░█░█░█▀▀░█▀▄░█▀█░░█░░█▀▀░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀░

# The store path of the generator changes with the package, but store files all have the same timestamp
# The path is recorded in a file that changes only with it, so the generator can be a prerequisite
$(GENERATED)/generator: FORCE
	@mkdir -p $(@D); [ "$$(cat $@ 2>/dev/null)" = "$(GENERATOR)" ] || printf '%s\n' "$(GENERATOR)" > $@

FORCE:

# The flags and file list are recorded in a file that changes only with them, so the objects can depend on it
# FORCE makes make compare on every run, and the file keeps its timestamp when nothing changed
$(BUILD)/flags: FORCE
	@mkdir -p $(@D); [ "$$(cat $@ 2>/dev/null)" = "$$FLAGS_RECORD" ] || printf '%s\n' "$$FLAGS_RECORD" > $@

# A stamp rather than a grouped target, because export-weights leaves unchanged outputs alone
# A grouped target would rerun it on every make once the safetensors file is touched
$(GENERATED)/stamp: $(SAFETENSORS_PREREQ) $(GENERATED)/generator
	export-weights "$<" $(GENERATED)
	touch $@

$(foreach w,weights weights_int8,$(GENERATED)/$(w).c $(GENERATED)/$(w)_table.c $(GENERATED)/$(w).tdump): $(GENERATED)/stamp ;

-include $(OBJECTS:.o=.d) $(APPROX_NEW_OBJECTS:.o=.d) $(QLINEAR_NEW_OBJECTS:.o=.d) $(PROGRAM_OBJECTS:.o=.d) $(DUMP_OBJECTS:.o=.d) $(PROFILE_OBJECTS:.o=.d) $(STACK_OBJECTS:.o=.d) $(NEWLIB_OBJECTS:.o=.d) $(PLATFORM_OBJECTS:.o=.d) $(GENERATED_OBJECTS:.o=.d)
