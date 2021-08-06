CC=g++

SRCDIR=src

CCFLAGS=-I.
DEPS = *.h *.tpp

OBJS = tensor_utils.o tensor.o test.o

all: $(OBJS)
	$(CC) $(CCFLAGS) -o test $(OBJS)

%.o: %.cpp $(DEPS)
	$(CC) $(CCFLAGS) -c -o $@ $<

.PHONY: clean test

clean:
	rm -rf *.o test

test:
	./test
