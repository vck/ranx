CC      ?= clang
CFLAGS  ?= -Wall -Wextra -Wpedantic -O3 -march=armv8.5-a+fp16 -std=c11 -DACCELERATE_NEW_LAPACK
LDFLAGS ?= -framework Accelerate

SRC = src/main.c src/gguf.c src/model.c src/ops.c src/tokenizer.c
OBJ = $(SRC:.c=.o)

ranx: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f ranx src/*.o

.PHONY: clean
