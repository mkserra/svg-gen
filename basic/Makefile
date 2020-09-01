OBJS = svg_gen.c

CC = c99

COMPILER_FLAGS = -O3 -Wall -Wextra -march=native -mtune=native -msse2 -mssse3 -Wno-unused-result

SDL_LIB = -L/usr/lib -lSDL -Wl,-rpath=/usr/lib

SDL_INCLUDE = -I/usr/include

LDFLAGS = $(SDL_LIB)

LINKER_FLAGS = -lSDL -lSDL_image -lSDL_gfx -lm

OBJ_NAME = svg_gen

all : $(OBJS)
		$(CC) $(OBJS) $(COMPILER_FLAGS) $(LINKER_FLAGS) -o $(OBJ_NAME)
