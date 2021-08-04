from distutils.core import setup, Extension


def main():
	setup(
		name="fputs",
		version="1.0.0",
		description="Python interface for the fputs C library function",
		packages=["mytensor", "mytensor.lib"],
		ext_modules=[
			# Extension("mytensor.lib.vector", ["mytensor/lib/src/vectormodule.c"]),
			Extension(
				"mytensor.lib.tensor",
				["mytensor/lib/src/tensormodule.cpp", "mytensor/lib/src/tensor.cpp"],
			),
		],
	)


if __name__ == "__main__":
	main()
