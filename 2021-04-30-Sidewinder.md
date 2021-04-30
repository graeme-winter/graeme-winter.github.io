# Parallelism in Python II: Real Work

## TL;DR

Demo cases about making Python code much faster, using numba, openCL etc.

[code here](https://github.com/graeme-winter/sidewinder)

## verbose=true

[Following on from](./2021-04-19-Sidewinder) time for some real work - stuff which requires non-trivial memory management, worker configuration etc. - finding spots on X-ray diffraction images. Following on from an implementation in [DIALS](https://doi.org/10.1107/S2059798317017235) want to try identifying signal pixels in regions of images - this requires computing the

- local mean
- local variance
- index of dispersion (variance / mean)

then thresholding the individual pixel against these quantities. Typically the calculation is performed on a 7x7 grid of pixels around the pixel in question, so whenever you have access to a pixel, you also need access to _another_ 48 pixels somehow to make this useful. CPU implementations of this focus on the use of integral images, but with a GPU why not try brute force?

## Precondition: memory management

Copying the image data in is critical, likewise the valid pixel mask. Making sure that the right pixels come out at the end also critical -> first task is to implement 1D, 2D, 3D `memcpy` functions to verify that this works as I expected. Turns out this was useful, as getting the `global_id` _right_ requires more head scratching than I would have expected. End game though;

```python
# memcpy_3d.py
#
# 3D memcpy function on a GPU, to verify correct treatment of data in local
# memory

import sys
import time

import numpy as np
import pyopencl as cl


def get_devices():
    result = []
    for pl in cl.get_platforms():
        result.extend(pl.get_devices())
    return result


def _help():
    devices = get_devices()
    print("Available devices:")
    for j, dev in enumerate(devices):
        print(f"{j} -> {dev.name}")
        print(f"Type:                 {cl.device_type.to_string(dev.type)}")
        print(f"Vendor:               {dev.vendor}")
        print(f"Available:            {bool(dev.available)}")
        print(f"Memory (global / MB): {int(dev.global_mem_size/1024**2)}")
        print(f"Memory (local / B):   {dev.local_mem_size}")
        print("")
    print(f"Please select device 0...{len(devices)-1}")


def main():
    if len(sys.argv) != 2:
        _help()
        sys.exit(1)

    device = int(sys.argv[1])

    devices = get_devices()
    context = cl.Context(devices=[devices[device]])
    queue = cl.CommandQueue(context)

    # TODO use these to decide what "shape" to make the work groups
    # and add something which allows those shapes to be replaced in the
    # openCL source code
    max_group = devices[device].max_work_group_size
    max_item = devices[device].max_work_item_sizes

    cl_text = open("memcpy_3d.cl", "r").read().replace("LOCAL_SIZE", "256")
    program = cl.Program(context, cl_text).build()
    memcpy_3d = program.memcpy_3d

    memcpy_3d.set_scalar_arg_dtypes(
        [
            None,
            np.int32,
            np.int32,
            np.int32,
            None,
        ]
    )

    shape = (32, 512, 1028)
    size = shape[0] * shape[1] * shape[2]

    mem_in = np.random.randint(0, 256, size=size, dtype=np.uint16).reshape(shape)

    _mem_in = cl.Buffer(
        context, cl.mem_flags.READ_ONLY, mem_in.size * np.dtype(mem_in.dtype).itemsize
    )
    _mem_out = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, mem_in.size * np.dtype(mem_in.dtype).itemsize
    )

    mem_out = np.zeros(shape=mem_in.shape, dtype=mem_in.dtype)

    cl.enqueue_copy(queue, _mem_in, mem_in)

    # work must be a multiple of group size
    group = (1, 12, 16)
    work = tuple(int(group[d] * np.ceil(shape[d] / group[d])) for d in (0, 1, 2))
    print(f"{shape} -> {work}")
    evt = memcpy_3d(
        queue,
        work,
        group,
        _mem_in,
        shape[0],
        shape[1],
        shape[2],
        _mem_out,
    )
    evt.wait()

    cl.enqueue_copy(queue, mem_out, _mem_out)

    assert np.array_equal(mem_in, mem_out)


main()
```

and

```c
__kernel void memcpy_3d(const __global unsigned short *mem_in, const int frames,
                        const int height, const int width,
                        __global unsigned short *mem_out) {

  __local unsigned short _mem[LOCAL_SIZE];

  int gid[3], gsz[3], ggd[3];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);
  gid[2] = get_global_id(2);

  gsz[0] = get_global_size(0);
  gsz[1] = get_global_size(1);
  gsz[2] = get_global_size(2);

  ggd[0] = get_group_id(0);
  ggd[1] = get_group_id(1);
  ggd[2] = get_group_id(2);

  int lid[3], lsz[3];

  lid[0] = get_local_id(0);
  lid[1] = get_local_id(1);
  lid[2] = get_local_id(2);

  lsz[0] = get_local_size(0);
  lsz[1] = get_local_size(1);
  lsz[2] = get_local_size(2);

  if (lid[0] == lid[1] == lid[2] == 0) {
    for (int i = 0; i < lsz[0]; i++) {
      for (int j = 0; j < lsz[1]; j++) {
        for (int k = 0; k < lsz[2]; k++) {
          if (((ggd[0] * lsz[0] + i) > frames) ||
              ((ggd[1] * lsz[1] + j) > height) ||
              ((ggd[2] * lsz[2] + k) > width)) {
            _mem[i * lsz[1] * lsz[2] + j * lsz[2] + k] = 0;
            continue;
          }
          _mem[i * lsz[1] * lsz[2] + j * lsz[2] + k] =
              mem_in[(ggd[0] * lsz[0] + i) * height * width +
                     (ggd[1] * lsz[1] + j) * width + (ggd[2] * lsz[2] + k)];
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if ((gid[0] < frames) && (gid[1] < height) && (gid[2] < width)) {
    mem_out[gid[0] * height * width + gid[1] * width + gid[2]] =
        _mem[lid[0] * lsz[1] * lsz[2] + lid[1] * lsz[2] + lid[2]];
  }

  return;
}
```

As with all this stuff, most of this is arithmatic.

## Real Work

Actually reading the image is one thing, but this data is read from modular detectors so made a choice to blit the data from one large image into a stack of 32 smaller images (around 500,000 pixels each) and perform the spot finding on them. These have a substantial gap (much more than 3 pixels) so not looking for common signal across modules.

Getting the data in place is straightforward, as is reading the mask etc. - the tricky bit was making the kernel work right, as you start to become sensitive to (i) the amount of local memory available (ii) the sizes of work groups and (iii) the amount of calculation necessary for a given amount of data transfer. The last of these is the most important measure of whether a GPU will help you... if this is not a big number, the answer is probably no.

Kernel:

```c
__kernel void spot_finder(const __global unsigned short *image,
                          const __global unsigned char *mask, const int frames,
                          const int height, const int width, const int knl,
                          const float sigma_s, const float sigma_b,
                          __global unsigned char *signal) {

  __local unsigned short _image[LOCAL_SIZE];
  __local unsigned char _mask[LOCAL_SIZE];

  int gid[3], ggd[3];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);
  gid[2] = get_global_id(2);

  ggd[0] = get_group_id(0);
  ggd[1] = get_group_id(1);
  ggd[2] = get_group_id(2);

  int lid[3], lsz[3];

  lid[0] = get_local_id(0);
  lid[1] = get_local_id(1);
  lid[2] = get_local_id(2);

  lsz[0] = get_local_size(0);
  lsz[1] = get_local_size(1);
  lsz[2] = get_local_size(2);

  int knl2 = 2 * knl + 1;
  int nj = lsz[1] + knl2;
  int nk = lsz[2] + knl2;

  if (lid[0] == lid[1] == lid[2] == 0) {
    int off[3];
    off[0] = ggd[0] * lsz[0];
    off[1] = ggd[1] * lsz[1];
    off[2] = ggd[2] * lsz[2];
    for (int j = 0; j < nj; j++) {
      for (int k = 0; k < nk; k++) {
        int _j = j - knl;
        int _k = k - knl;
        if (((off[1] + _j) < 0) || ((off[2] + _k) < 0) ||
            ((off[1] + _j) >= height) || ((off[2] + _k) >= width)) {
          _image[j * nk + k] = 0;
          _mask[j * nk + k] = 0;
        } else {
          _image[j * nk + k] = image[(off[0] * height * width) +
                                     (off[1] + _j) * width + (off[2] + _k)];
          _mask[j * nk + k] = mask[(off[0] * height * width) +
                                   (off[1] + _j) * width + (off[2] + _k)];
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if ((gid[0] >= frames) || (gid[1] >= height) || (gid[2] >= width)) {
    return;
  }

  // local and global pixel locations for this worker
  int gpxl = gid[0] * width * height + gid[1] * width + gid[2];
  int lpxl = (lid[1] + knl) * nk + lid[2] + knl;

  // if masked, cannot be signal
  if (_mask[lpxl] == 0) {
    signal[gpxl] = 0;
    return;
  }

  float sum = 0.0;
  float sum2 = 0.0;
  float n = 0.0;

  for (int j = 0; j < knl2; j++) {
    for (int k = 0; k < knl2; k++) {
      int pxl = (lid[1] + j) * nk + lid[2] + k;
      sum += _image[pxl] * _mask[pxl];
      sum2 += _image[pxl] * _image[pxl] * _mask[pxl];
      n += _mask[pxl];
    }
  }

  if ((n >= 2) && (sum >= 0)) {
    float n_disp = n * sum2 - sum * sum - sum * (n - 1);
    float t_disp = sum * sigma_b * sqrt(2 * (n - 1));
    float n_stng = n * _image[lpxl] - sum;
    float t_stng = sigma_s * sqrt(n * sum);
    if ((n_disp > t_disp) && (n_stng > t_stng)) {
      signal[gpxl] = 1;
      return;
    }
  }

  signal[gpxl] = 0;

  return;
}
```

copies data from global memory into the local memory with a `knl` margin (either masked 0's or pixel data). Mean and variance are then computed from various sums, and the pixel thresholded with the output.

Turns out that this works fine, but cannot really saturate even a laptop GPU - the amount of calculation even with a brute force algorithm is not enough to make it worthwhile... 
