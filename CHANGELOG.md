# v2.2.0
December 24, 2025

**This version introduces feature-breaking API changes.**

## General
* Added a different way of computing contact impulse between dynamic shapes and MPM nodes. Particle push-out does not lead to impulse any more.
This approach is mostly LLM-writen and not well-tested.
* Added `ShapeContactModel` for controlling which contact method to use for each `Scene`. Use `SceneDesc::contactModel` (default: `ShapeContactModel::eKinematic`).
This does not affect `ShapeType::eKinematic` and `ShapeType::eStatic` shapes.
* For all dynamic shapes, `ShapeDesc::comInvMass`, `ShapeDesc::inertiaInv0` and `ShapeDesc::inertiaInv1` should be specified,
no matter which `ShapeContactModel` is used. if `ShapeContactModel::eKinematic` is used, these added data will not be used.

## API
* Changed C API functions `CrCreateScene()` and `CrCreateShape()` for setting the contact model and dynamic shape data. Added C API enum `CrShapeContactModel`.
* Added `MpmSolver::setShapeContactModel()` for setting solver dynamic shape contact model.

# v2.1.0
December 23, 2025

## API
* Added methods for setting shape pose and velocity directly using `setPose()` and `setVelocity()`, in addition to `setKinematicTarget()`.
* Added corresponding C API functions `CrSetShapePose` and `CrSetShapeVelocity`.

## Fixes
* Fixed an error when deformation gradient tensor is not correctly (re)initialized to identity.

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