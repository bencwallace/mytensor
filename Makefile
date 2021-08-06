all:
	+$(MAKE) -C src

clean:
	+$(MAKE) -C src clean

test:
	+$(MAKE) -C src test
