# Builds the program and the weight check for the host (`make`) or for rv64gc, scalar (`make TARGET=rv64`)
# `tspulse-dump` is the program with the code that writes inner outputs, which `check-model` runs, and `tspulse` carries none of it
# `make check` runs the weight check and compares its dump with the one export-weights wrote from the safetensors file
# Run inside the dev shell, after `download-weights`

# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀
# ░█░░░█░█░█░█░█▀▀░░█░░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀

TARGET ?= host
BUILD := build/$(TARGET)
GENERATED := build/generated

# Fusing, vectorizing and the C standard change float results, so the host build is pinned to what rv64 does
# -std=c11 turns fusing off silently, so no -std flag is given
ifeq ($(TARGET),host)
CC := gcc
ARCH_FLAGS := -O2 -mfma --param avoid-fma-max-bits=0 -fno-tree-vectorize
else ifeq ($(TARGET),rv64)
CC := riscv64-none-elf-gcc
ARCH_FLAGS := -march=rv64gc -mabi=lp64d -O2
else
$(error unknown TARGET '$(TARGET)', expected host or rv64)
endif

CPPFLAGS += -I. -MMD -MP
CFLAGS += $(ARCH_FLAGS) -Wall -Wextra
LDLIBS := -lm

# The host build compiles newlib's erff and expf, because glibc's round differently and the host must match rv64 bit for bit
# Without __OBSOLETE_MATH=0, x86 takes newlib's old code path: sf_exp.c compiles to nothing and sf_erf.o needs __ieee754_expf
# Only the machine/ folder is added to the include path, since newlib's whole libc/include would replace glibc's math.h
NEWLIB_FILES := libm/math/sf_erf.c libm/common/sf_exp.c libm/common/sf_exp2_data.c libm/common/math_errf.c libm/common/math_err.c
NEWLIB_FLAGS = -D__OBSOLETE_MATH=0 -I$(NEWLIB_HOST_INCLUDE) -I$(NEWLIB_SRC)/newlib/libm/common

# ░█▀▀░█▀█░█░█░█▀▄░█▀▀░█▀▀░█▀▀
# ░▀▀█░█░█░█░█░█▀▄░█░░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

SOURCES := io/tdump.c checks/weights.c
OBJECTS := $(SOURCES:%.c=$(BUILD)/%.o)

# The program is built twice: the dump build defines TSPULSE_DUMP and keeps its objects apart, so it never mixes with the clean one
KERNEL_SOURCES := kernels/add.c kernels/fft.c kernels/gelu.c kernels/layer_norm.c kernels/linear.c kernels/multiply.c kernels/revin.c kernels/softmax.c kernels/sum.c kernels/transpose.c
MODULE_SOURCES := modules/backbone.c modules/block.c modules/decoder.c modules/embedding.c modules/heads.c modules/mixer.c
PROGRAM_SOURCES := tspulse.c io/rows.c model/history.c model/pass.c model/window.c $(KERNEL_SOURCES) $(MODULE_SOURCES)
PROGRAM_OBJECTS := $(PROGRAM_SOURCES:%.c=$(BUILD)/%.o)
DUMP_OBJECTS := $(PROGRAM_SOURCES:%.c=$(BUILD)/dump/%.o) $(BUILD)/dump/io/tdump.o $(BUILD)/dump/io/dump.o

ifeq ($(TARGET),host)
NEWLIB_OBJECTS := $(NEWLIB_FILES:%.c=$(BUILD)/newlib/%.o)
else
# Under pk the program carries its own _open, because pk reads open flags as Linux numbers
PLATFORM_OBJECTS := $(BUILD)/io/pk_open.o
endif
GENERATED_SOURCES := $(GENERATED)/weights.c $(GENERATED)/weights_table.c
GENERATED_OBJECTS := $(GENERATED_SOURCES:$(GENERATED)/%.c=$(BUILD)/generated/%.o)

# The program needs only the weights, while the weight check also needs their table of names
WEIGHTS_OBJECT := $(BUILD)/generated/weights.o

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

ifeq ($(TARGET),host)
$(if $(NEWLIB_SRC),,$(error NEWLIB_SRC is not set; run make inside nix develop))
$(if $(NEWLIB_HOST_INCLUDE),,$(error NEWLIB_HOST_INCLUDE is not set; run make inside nix develop))
endif

endif

# ░▀█▀░█▀█░█▀▄░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀█░█▀▄░█░█░█▀▀░░█░░▀▀█
# ░░▀░░▀░▀░▀░▀░▀▀▀░▀▀▀░░▀░░▀▀▀

.PHONY: all check clean FORCE

all: $(BUILD)/tspulse $(BUILD)/tspulse-dump $(BUILD)/check-weights

# Run the check program: on the host directly, on rv64 under Spike
check: $(BUILD)/check-weights $(GENERATED)/weights.tdump
ifeq ($(TARGET),host)
	$(BUILD)/check-weights $(BUILD)/weights.tdump
else
	spike --isa=rv64gc_zicntr "$$RISCV_PK" $(BUILD)/check-weights $(BUILD)/weights.tdump
endif
# Pass only on 0 fail, 0 missing and 0 extra, because compare-reference exits 0 on missing records
	status=0; compare-reference $(GENERATED)/weights.tdump $(BUILD)/weights.tdump --exact > $(BUILD)/compare.txt || status=$$?; \
	grep -E '^[0-9]+ pass, ' $(BUILD)/compare.txt; \
	[ $$status -eq 0 ] && grep -Eq '^[0-9]+ pass, 0 fail, 0 missing, 0 extra$$' $(BUILD)/compare.txt

clean:
	rm -rf build

# ░█▀▄░█░█░▀█▀░█░░░█▀▄░█▀▀
# ░█▀▄░█░█░░█░░█░░░█░█░▀▀█
# ░▀▀░░▀▀▀░▀▀▀░▀▀▀░▀▀░░▀▀▀

$(BUILD)/check-weights: $(OBJECTS) $(GENERATED_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

# The newlib objects come before -lm, so the linker finds erff and expf in them first
$(BUILD)/tspulse: $(PROGRAM_OBJECTS) $(WEIGHTS_OBJECT) $(NEWLIB_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/tspulse-dump: $(DUMP_OBJECTS) $(WEIGHTS_OBJECT) $(NEWLIB_OBJECTS) $(PLATFORM_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# A member with transposed dimensions has the right size but only draws an excess-elements warning, so generated objects build with -Werror
$(GENERATED_OBJECTS): $(BUILD)/generated/%.o: $(GENERATED)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Werror -c $< -o $@

$(OBJECTS) $(PROGRAM_OBJECTS) $(PLATFORM_OBJECTS): $(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# PyTorch's RevIN does each step as its own tensor operation, in both directions, rounding after each, so the whole file is built without fused multiply-add
# Fused, more scaled values land one ulp away from PyTorch's, and the alternating sum behind the FFT's base component (bin 256) magnifies those differences past its allowance on YAHOO w908
$(BUILD)/kernels/revin.o $(BUILD)/dump/kernels/revin.o: CFLAGS += -ffp-contract=off

$(DUMP_OBJECTS): $(BUILD)/dump/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DTSPULSE_DUMP -c $< -o $@

$(NEWLIB_OBJECTS): $(BUILD)/newlib/%.o: $(NEWLIB_SRC)/newlib/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(NEWLIB_FLAGS) -c $< -o $@

# ░█▀▀░█▀▀░█▀█░█▀▀░█▀▄░█▀█░▀█▀░█▀▀░█▀▄
# ░█░█░█▀▀░█░█░█▀▀░█▀▄░█▀█░░█░░█▀▀░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀░

# The store path of the generator changes with the package, but store files all have the same timestamp
# The path is recorded in a file that changes only with it, so the generator can be a prerequisite
$(GENERATED)/generator: FORCE
	@mkdir -p $(@D); [ "$$(cat $@ 2>/dev/null)" = "$(GENERATOR)" ] || printf '%s\n' "$(GENERATOR)" > $@

FORCE:

# A stamp rather than a grouped target, because export-weights leaves unchanged outputs alone
# A grouped target would rerun it on every make once the safetensors file is touched
$(GENERATED)/stamp: $(SAFETENSORS_PREREQ) $(GENERATED)/generator
	export-weights "$<" $(GENERATED)
	touch $@

$(GENERATED)/weights.c $(GENERATED)/weights_table.c $(GENERATED)/weights.tdump: $(GENERATED)/stamp ;

-include $(OBJECTS:.o=.d) $(PROGRAM_OBJECTS:.o=.d) $(DUMP_OBJECTS:.o=.d) $(NEWLIB_OBJECTS:.o=.d) $(PLATFORM_OBJECTS:.o=.d) $(GENERATED_OBJECTS:.o=.d)
