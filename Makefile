CC=g++

INCDIR=src
OUTDIR=build
SRCDIR=src

CCFLAGS=-I$(INCDIR)
DEPS=$(INCDIR)/*.h

OBJS = tensor.o

$(OUTDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CCFLAGS)

$(OUTDIR)/tensor.a: $(OUTDIR)/$(OBJS)
	ar rcs $@ $<

build: build/tensor.a

$(OUTDIR)/test: test.cpp $(OUTDIR)/tensor.a
	$(CC) -o $@ test.cpp $(OUTDIR)/tensor.a $(CCFLAGS)

.PHONY: clean run

clean:
	rm -rf $(OUTDIR)/*.o $(OUTDIR)/*.a

run:
	./build/test
