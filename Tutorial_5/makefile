COMPILER := gcc
OPT_FLAGS := -O3 -fopenmp
all: compile
	${COMPILER} Tutorial_5.o -o Tutorial_5.exe ${OPT_FLAGS}
compile:
	${COMPILER} Tutorial_5.c -c ${OPT_FLAGS}
clean:
	rm Tutorial_5.exe Tutorial_5.o