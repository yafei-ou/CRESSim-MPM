# v2.0.2
December 3, 2025

## Fixes
* Fix a math error in calculating node velocities relative to a rigid body.


# v2.0.1
December 1, 2025

## General
* Add version information for the project.

## Fixes
* Fix a missing macro import for INT_MAX.
* Fix a cmake logic error when glad is not added when only `ENABLE_EXAMPLES` is set to `ON`.

## Documentation
* Add CHANGELOG.md.
* Update README.md to match API changes in `v2.0.0`.


# v2.0.0
November 28, 2025

**This version introduces feature-breaking API changes.**

## General

* Particle position/mass and velocity data are unified for all scenes in one memory block. This allows reading all particle data at once.
* Multiple CUDA streams can be used. Each scene/solver will use one stream. The maximum number of CUDA streams can be set. See [API](#API) changes.
* All future GPU solvers must inherit from `MpmSolverGpu`.

## API
* Changed `createFactory` and `CrInitializeEngine()`. The maximum number of particles in the `SimulationFactory` must be given. The number of CUDA streams to be used can be set (default: 1).
* Added `SimulationFactory.advanceAll()` / `CrAdvanceAll()` and `SimulationFactory.fetchResultsAll()` `CrFetchResultsAll()` to allow advancing all scenes at once.
* Added `SimulationFactory.getParticleDataAll()` / `CrGetParticleDataAll()` for getting all particle data in the `SimulationFactory`.
* Added `Scene.getParticleDataGlobalOffset()` / `CrGetSceneParticleDataGlobalOffset()` for reading the offset of one scene's particle data in the global particle data array.

## Fixes
* Removed duplicated logic across multiple GPU solvers and consolidated it into `MpmSolverGpu`.


# v1.2.0
November 17, 2025

## APIs
* Add C API function `CrSceneMarkDirty` and enum `CrSceneDataDirtyFlags` for marking Scene data dirty.


# v1.1.1
October 29, 2025

## Fixes
* Fix an error when computing the internal force.


# v1.1.0
October 8, 2025

## APIs
* Add C API function `CrResetParticleObject` for resetting the ParticleObject to its initial state.


# v1.0.1
July 3, 2025

## General
* Add an example about cutting using `QuadSlicer`.

## Fixes
* Remove some weird casts and pointer arithmetics.


# v1.0.0
May 3, 2025

## General

* Initial public release.