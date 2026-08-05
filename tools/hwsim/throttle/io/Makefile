CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -pthread
# Libraries must come AFTER the source in the link line: toolchains that
# resolve archives strictly left-to-right otherwise drop libpthread symbols
# -> "undefined reference to pthread_join" (seen on the RTX PRO 6000 x86
# box, external-validation-rtx-pro-6000.md defect 1).
LDLIBS  ?= -lpthread

io_load: io_load.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f io_load

.PHONY: clean
