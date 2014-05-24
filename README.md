Circle Map Rotation Number
==========================

There is a map named [Circle Map]. It maps a circle to itself:

![Equation](http://upload.wikimedia.org/math/c/c/c/ccc193ba8bfd2d40b2ea087ac2ae1c1f.png)

θ is an angle between 0 and 1.

Circle map has two parameters, K and Ω. We can lock them and calculate the average angle the point rotates by during iteration process. If we clip resulting value by [0, 1] and vary our parameters, we can get following picure:

![Result](https://demyanenko.github.io/Circle-Map-Rotation-Number/tiles/tile-0-0-0.png)

There red corresponds to 0, yellow to 1/2 and blue to 1. Ω varies from 0 to 1 along the x-axis, and K varies from 0 at the bottom to 2π at the top.

But if we want to produce a picture that is bigger than 256x256, we might want to use GPU power for computing. That's why I implemented this program in CUDA.

I rendered the picture in 16K×16K resolution, and that took 20+ hours on GTX480. You can view result [here](http://demyanenko.github.io/Circle-Map-Rotation-Number/) (powered by pano.js).

Prerequisites
-------------
- Nvidia GPU with Computing Capability 2.0 or higher
- `nvcc` compiler
- Python 2.x with `numpy` and `matplotlib` modules

Usage
-----
The program renders the picture of rotation numbers by square parts.

### Fetching
**WARNING**

`git clone` fetches all branches by default, so it will download hundreds of tiles from GitHub Pages.

To avoid this, use
```
$ git clone https://github.com/demyanenko/Circle-Map-Rotation-Number.git -b master --single-branch
```

### Configuring
Rendering process is fully controlled by definitions in the beginning of `rmap.cu`:
```
#define SIZE 256  // part side in pixels
#define TOTAL_SIZE 256  // whole picture side in pixels
#define X 0  // part offset from left
#define Y 0  // part offset from bottom
```
You can adjust threading settings according to your GPU:
```
#define THREADS 448
#define JOBS_PER_THREAD 384
#define ITER_PER_JOB 100000
```
And computation settings:
```
#define EPS (floattype)(1e-3)
#define ITER_TAIL 1000
#define SAMPLING_TAIL 512
```

### Compiling
`$ make`

### Rendering
`$ ./rmap`

Output consists of five files in binary format:
- `image` is the picture of rotation numbers itself;
- `aa` is number of different seeds required for each pixel to converge;
- `iter_min`, `iter` and `iter_max` are minimum, average and maximum number of iterations per pixel respectively.

The first is desired one, but others are quite pretty, too. Types, sizes and offsets of parts are coded in filenames, so please don't change them.

### Combining parts
If you decided to render all picture in one part, you can skip this section.

Otherwise, you may notice that resulting picture has horisontal symmetry.

`combine.py` takes paths to the parts of the same type (`image`, for example), merges them and copy-reflects the left half to the right, resulting in big combined file in binary format.

### Producing picture
After you have desired pictures in binary format, it's time to convert them to actual pictures.

`convert.py` takes paths to binary files and produces PNG pictures.
If there is `image` substring in filename, pixel values are clipped and color coded in [0, 1], otherwise in [min, max] of overall picture.

[Circle Map]:(http://en.wikipedia.org/wiki/Circle_map)
