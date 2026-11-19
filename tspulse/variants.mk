# The settings table and what a choice of values turns into, included by the Makefile before it picks the build folder
# Variables only, so the Makefile's default goal stays `all`
# A setting is a Make variable with a fixed list of values, set on the command line (`make LINEAR=blocked`); a build that sets nothing is the baseline
# To add a setting, append its name to SETTINGS and give it the entries below, where S stands for its name and V for one of its values:
#   S ?= <default value>
#   S_NAME := <lower-case name, used in the build's name and folder>
#   S_DEFAULT := <default value>
#   S_VALUES := <every allowed value, in lower case>
#   S_SOURCES_V   files compiled for V, in place of the baseline files in S_REPLACES, or added to them when there is no S_REPLACES
#   S_REPLACES    the baseline files that a value with S_SOURCES_V swaps out
#   S_DEFINES_V   -D macros for V
#   S_FLAGS_V     compiler flags for V, on both targets, so the host build keeps matching rv64 bit for bit
#   S_HOST_FLAGS_V, S_RV64_FLAGS_V   compiler flags for V on one target
#   S_NEWLIB_V    yes when V compiles newlib's math source into the rv64 program too, as the host build always does
#   S_WEIGHTS_V   the generated weight set V builds with, as in int8 for build/generated/weights_int8.c, weights_int8_table.c and weights_int8.tdump; without it, the float set

# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀
# ░█░░░█░█░█░█░█▀▀░░█░░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀

# Listed in the order they appear in a build's name
SETTINGS := LINEAR UNROLL MATH OUTPUTS EXP GELU QUANT

# The blocked kernel keeps each output's sum in the plain loop's order, so the result stays exact
LINEAR ?= plain
LINEAR_NAME := linear
LINEAR_DEFAULT := plain
LINEAR_VALUES := plain blocked
LINEAR_REPLACES := kernels/linear/plain.c
LINEAR_SOURCES_blocked := kernels/linear/blocked.c

# GCC unrolls without reordering the float sum, so the result stays exact
UNROLL ?= off
UNROLL_NAME := unroll
UNROLL_DEFAULT := off
UNROLL_VALUES := off on
UNROLL_FLAGS_on := -funroll-loops

# The same newlib source as the toolchain's libm.a, compiled with the program's flags, so it returns the same results in fewer instructions per call
MATH ?= libm
MATH_NAME := math
MATH_DEFAULT := libm
MATH_VALUES := libm bundled
MATH_NEWLIB_bundled := yes

# Unread rows are never computed and the read rows run the same code, so the result stays exact
OUTPUTS ?= all
OUTPUTS_NAME := outputs
OUTPUTS_DEFAULT := all
OUTPUTS_VALUES := all read
OUTPUTS_DEFINES_read := -DTSPULSE_OUTPUTS_READ

# Approximations trade accuracy for fewer instructions per call, and the values are inline headers so that the cheap ones don't pay a call
EXP ?= libm
EXP_NAME := exp
EXP_DEFAULT := libm
EXP_VALUES := libm fine coarse bits table
EXP_DEFINES_fine := -DEXP_FINE
EXP_DEFINES_coarse := -DEXP_COARSE
EXP_DEFINES_bits := -DEXP_BITS
EXP_DEFINES_table := -DEXP_TABLE
EXP_SOURCES_table := kernels/exp/table.c

# GELU's values are inline headers for the same reason as EXP's
GELU ?= libm
GELU_NAME := gelu
GELU_DEFAULT := libm
GELU_VALUES := libm fine coarse tanh table
GELU_DEFINES_fine := -DGELU_FINE
GELU_DEFINES_coarse := -DGELU_COARSE
GELU_DEFINES_tanh := -DGELU_TANH
GELU_DEFINES_table := -DGELU_TABLE
GELU_SOURCES_table := kernels/gelu/table.c

# Int8 mixer weights with one float scale per output row change the result; the kernel is added beside `linear`, which still runs every Linear outside the mixers
QUANT ?= off
QUANT_NAME := quant
QUANT_DEFAULT := off
QUANT_VALUES := off w8a32
QUANT_DEFINES_w8a32 := -DTSPULSE_QUANT_W8A32
QUANT_SOURCES_w8a32 := kernels/qlinear/w8a32.c
QUANT_WEIGHTS_w8a32 := int8

# ░█▀▀░█▀█░█░█░█▀▄░█▀▀░█▀▀░█▀▀
# ░▀▀█░█░█░█░█░█▀▄░█░░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# A value outside the list would silently build the baseline's code under another name, and a value of several words passes the filter when one of them is allowed
$(foreach s,$(SETTINGS),$(if $(and $(filter 1,$(words $($(s)))),$(filter $($(s)),$($(s)_VALUES))),,$(error $(s)=$($(s)) is not allowed, expected one of: $($(s)_VALUES))))

# A build is named after its settings that differ from the default, in table order, as `<setting>-<value>` joined with +
# `+custom` marks a compiler, architecture or flag choice made on the command line, which no setting accounts for
# A setting left at its default leaves an extra space between its neighbours, which strip removes so it never turns into a stray +
VARIANT_PARTS := $(strip $(foreach s,$(SETTINGS),$(if $(filter $($(s)),$($(s)_DEFAULT)),,$($(s)_NAME)-$($(s)))))
VARIANT_CUSTOM := $(filter command line,$(foreach v,CC ARCH_FLAGS,$(origin $(v))))
VARIANT_NAME := $(if $(VARIANT_PARTS),$(subst $() $(),+,$(VARIANT_PARTS)),baseline)$(if $(VARIANT_CUSTOM),+custom)

# A value that names replacement files swaps out the setting's baseline files, and one that doesn't leaves them in
VARIANT_REPLACING := $(foreach s,$(SETTINGS),$(if $($(s)_SOURCES_$($(s))),$(s)))
VARIANT_REPLACED := $(foreach s,$(VARIANT_REPLACING),$($(s)_REPLACES))
VARIANT_SOURCES := $(foreach s,$(VARIANT_REPLACING),$($(s)_SOURCES_$($(s))))

# The program's file list for this build, given the baseline's
variant_sources = $(filter-out $(VARIANT_REPLACED),$(1)) $(VARIANT_SOURCES)

# Any setting whose value asks for newlib's math makes the rv64 build compile it
VARIANT_NEWLIB := $(strip $(foreach s,$(SETTINGS),$($(s)_NEWLIB_$($(s)))))

# The generated weight set's suffix, empty for the float set
VARIANT_WEIGHTS := $(strip $(foreach s,$(SETTINGS),$($(s)_WEIGHTS_$($(s)))))

VARIANT_DEFINES := $(foreach s,$(SETTINGS),$($(s)_DEFINES_$($(s))))

# The target's own flags come from the table's HOST or RV64 entries
VARIANT_TARGET := $(if $(filter host,$(TARGET)),HOST,RV64)
VARIANT_FLAGS := $(foreach s,$(SETTINGS),$($(s)_FLAGS_$($(s))) $($(s)_$(VARIANT_TARGET)_FLAGS_$($(s))))
