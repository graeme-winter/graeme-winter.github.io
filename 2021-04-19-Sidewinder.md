# Parallelism in Python

## TL;DR

Demo cases about making Python code much faster, using numba, openCL etc.

[code here](https://github.com/graeme-winter/sidewinder)

## verbose=true

Updating skills, trying to learn some more about what can be done with "modern" Python so decided to use a toy project (Mandelbrot set calculation) to investigate:

- numba
- dask
- pyopencl

... then move on to something a little more realistic.

## Mandelbrot set

[This](https://en.wikipedia.org/wiki/Mandelbrot_set) is obviously a very well known computational target and is embarassingly parallel so a good demonstration case. Basic calculation is trivial:

```python
def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < IMAX:
        z = z * z + c
        n += 1
		return n
```

if a value `c` is in the set, then this value `n >= IMAX`. This implementation is very slow however as it is all done in Python loops. In practice it is trivial to vectorise - however most Python optimisation like this is built around the concept of operations on every position of a numpy array, so in this case I implemented it as a complex field array, then compute the function above on every element of the array to generate a new array. This also made it suitable for `numba` just-in-time compilation and ufunc vectorisation, which worked reasonably well.

Much more fun was the openCL implementation: rewriting the calculation above as an openCL kernel was very straightforward, and allowed use of _massive_ parallelism to compute the set. Noteworthy when working on a GPU that there is a big gap between the single and double precision performance (both are in the repo above),

```c
__kernel void mandelbrot(const __global float2 *field, const int width,
                         const int height, const int imax,
                         __global int *image) {
  int j = get_global_id(0);
  int i = get_global_id(1);

  if (j >= width || i >= height) {
    return;
  }

  float cr = field[j + i * width].s0;
  float ci = field[j + i * width].s1;

  float zr = 0.0;
  float zi = 0.0;

  float tr, ti;

  int n = 0;

  while (((zr * zr + zi * zi) <= 4) && (n < imax)) {
    tr = zr * zr - zi * zi + cr;
    ti = 2 * zr * zi + ci;
    zr = tr;
    zi = ti;
    n += 1;
  }

  image[j + i * width] = n;
}
```

Kernel for this is nicely trivial - since there are no operations which interact between positions on the set there is no need to worry about memory management beyond there being space in the GPU to store the input and output arrays. End product:

![Mandelbrot set](./2021-04-19/mandelbrot.png)

More to follow...
