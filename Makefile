CC=g++

INCDIR=src
OUTDIR=build
SRCDIR=src

CCFLAGS=-I$(INCDIR)
DEPS=$(INCDIR)/*.h

$(OUTDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

build: $(OUTDIR)/tensor.o

.PHONY: clean

clean:
	rm -rf $(OUTDIR)/*.o
