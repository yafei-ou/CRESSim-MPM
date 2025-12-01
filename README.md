# CRESSim-MPM

CRESSim-MPM is a material point method (MPM) simulation library using CUDA. It is primarily focused on real-time soft body simulation for surgical applications but is also applicable to other domains.

## Features

* A light-weight library for real-time MPM simulation.
* Supports:
  * Standard MPM
  * [Moving least squares MPM (MLS-MPM)](https://yzhu.io/publication/mpmmls2018siggraph/)
  * [Position-based MPM (PB-MPM)](https://www.ea.com/seed/news/siggraph2024-pbmpm)
* Rigid body coupling and contact support (requires external solver for two-way coupling).
* Geometries tailored for surgical simulation, including cutting and suturing.

## Requirements

### Hardware

* A CUDA-enabled GPU. The code has been tested with compute capability >= 5.0.

### Core

* CUDA Toolkit (tested on version 12.6 and 13.0). Any recent version (version 12/13 or even older versions) should generally work.
* CMake (>=3.18)

### Optional

The following packages are used in examples and tests.

* [glfw](https://www.glfw.org/) for rendering.
* [Eigen](https://eigen.tuxfamily.org/).

## Build Instructions

### Basic Build

Clone this repository. Inside the repository, use the following commands:

```bash
mkdir build
cd build
cmake .. -DENABLE_DEBUG_LOGGER=ON -DENGINE_STATIC=ON -DENABLE_EXAMPLES=OFF -DENABLE_TESTS=OFF
cmake --build . --config Release
```

The binaries will be located in `build/bin`.

### Enable Examples and Tests

To enable examples and tests, set the CMake options `ENABLE_EXAMPLES`  and/or `ENABLE_TESTS` to `ON`. For example:

```bash
cmake .. -DENABLE_DEBUG_LOGGER=ON -DENGINE_STATIC=ON -DENABLE_EXAMPLES=ON -DENABLE_TESTS=ON
```

To specify the location of Eigen, use `EIGEN3_INCLUDE_DIR` (absolute path to the directory that contains the folder Eigen). For example:

```bash
cmake .. -DENABLE_DEBUG_LOGGER=ON -DENGINE_STATIC=ON -DENABLE_EXAMPLES=ON -DENABLE_TESTS=ON -DEIGEN3_INCLUDE_DIR="path/to/Eigen"
```

On Windows, the GLFW library path has to be specified with `GLFW_LIBRARY_DIR`. For example:
```bash
cmake .. -DENABLE_DEBUG_LOGGER=ON -DENGINE_STATIC=ON -DENABLE_EXAMPLES=ON -DENABLE_TESTS=ON -DEIGEN3_INCLUDE_DIR="path/to/Eigen" -DGLFW_LIBRARY_DIR="path/to/GLFW/lib-vc2022"
```

Optionally, the engine can be built as a shared library by changing the CMake option `ENGINE_STATIC` to `OFF`.

To change the target CUDA architecture(s), modify the `CMAKE_CUDA_ARCHITECTURES` value in the CMake cache.

## Getting Started

Below is a simple example of creating a simulation scene without soft body particles. For more practical usage, please refer to the code in `examples`.

```cpp
#include "simulation_factory.h"

int main()
{
    // Create factory
    crmpm::SimulationFactory *simFactory = crmpm::createFactory(10000, 4, 4, 2000, true);

    // Scene description
    crmpm::SceneDesc sceneDesc;
    sceneDesc.solverType = crmpm::MpmSolverType::eGpuMlsMpmSolver;
    sceneDesc.numMaxParticles = 2000;
    sceneDesc.gravity = crmpm::Vec3f(0, -9.81f, 0);
    sceneDesc.gridBounds = crmpm::Bounds3(crmpm::Vec3f(-5.0f, -0.5f, -5.0f), crmpm::Vec3f(5.0f, 5.0f, 5.0f));
    sceneDesc.gridCellSize = 0.2f;
    sceneDesc.solverIntegrationStepSize = 0.002f;
    sceneDesc.solverIterations = 20;

    // Create scene
    crmpm::Scene *scene = simFactory->createScene(sceneDesc);

    // Scene advance
    scene->advance(0.02);

    // Fetch results
    scene->fetchResults();

    // Release scene
    scene->release();
    crmpm::releaseFactory(simFactory);
    return 0;
}
```

## C APIs

To use the C APIs, include the header located in `c_api/include`. Link against the shared library target `crmpm_c_api`. As an example, please refer to the code in `examples/example_c_api`.

## Known Issues

* The library is not thread-safe.
* The library does not throw exceptions on error. Instead, an error message will be logged (if built with `-DENABLE_DEBUG_LOGGER=ON`).
* Triangle mesh processing and collision is not ideal. It may not work for certain types of meshes, especially if the mesh is not watertight.
* CMakeLists.txt files are badly written.

## Citation

If you find this project helpful for your research, please consider citing the following paper.

```bibtex
@article{ou2025cressim,
  title={CRESSim-MPM: A Material Point Method Library for Surgical Soft Body Simulation with Cutting and Suturing},
  author={Ou, Yafei and Tavakoli, Mahdi},
  journal={arXiv preprint arXiv:2502.18437},
  year={2025}
}
```