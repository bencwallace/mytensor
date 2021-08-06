CC=g++

SRCDIR=src

CCFLAGS=-I.
DEPS=*.h

OBJS = tensor.o test.o utils.o

all: $(OBJS)
	$(CC) $(CCFLAGS) -o test $(OBJS)

%.o: %.cpp $(DEPS)
	$(CC) $(CCFLAGS) -c -o $@ $<

.PHONY: clean test

clean:
	rm -rf *.o test

test:
	./test
