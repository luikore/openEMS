Metal engine
============

``openEMS model.xml --engine=metal`` runs the FDTD field updates on Apple GPUs
via Metal. Build with ``-DWITH_METAL=ON``; the option is off by default. It
enables every Metal feature; there are no per-feature switches. The same binary
keeps the SSE and multithreaded engines for comparison.

Operator construction, mesh grading, material EC sampling and the coefficient
build stay on the CPU. For ordinary Cartesian and PEC models, the GPU advances
an in-place space-time diamond wavefront. It uses one E/H field pair: no
full-grid ping-pong copy is allocated.

The GPU also executes the PEC / ``MATERIAL|METAL`` geometry pass. Its winners
remain available to the conducting-sheet and dispersive-material setup code.

Cylindrical and MPI operators are not supported by the Metal engine.

Field updates
-------------

The X and Y axes are split into alternating mountain and valley tiles. Their
ranges contract or expand at each E/H half-step according to stencil
dependencies. The Cartesian product gives four independent phases. Every
threadgroup owns one XY diamond, spans all packed-Z slots, and advances up to
four complete timesteps in place. Threads stride over the tile volume and use
device-memory threadgroup barriers between E, voltage-source, H, and
current-source work. Separate command encoders provide the global barrier
between diamond phases; the kernel never uses a grid-wide spin barrier.

The shortest XY span is two cells. This leaves enough independent threadgroups
for large grids while keeping four timesteps of temporal locality. Depths one
through four are precomputed, so a partial final block remains on the diamond
path rather than falling back. Schedule and source tables are small auxiliary
buffers; field storage remains one E/H pair.

Voltage and current sources are assigned to their unique owning tile for every
local timestep. Source order, overlapping sources, packed-Z lanes, partial edge
tiles, and boundary values are bit-identical to the explicit two-dispatch Metal
path.

UPML, ADE and arbitrary CPU extension hooks have not yet migrated inside the
diamond wavefront. Such a model aborts before timestep 0 rather than silently
running the old update path. ``OPENEMS_METAL_FUSED_PIPELINE=0`` is retained as
an explicit legacy diagnostic override. ``OPENEMS_METAL_FP64_REFERENCE`` also
requests that legacy diagnostic path. Fast math is always off.

PEC and geometry mapping
------------------------

The Metal pass resolves the winning ``MATERIAL|METAL`` primitive per Yee
component and records it via ``Operator::GetGeometryWinners()``;
``Operator_Ext_ConductingSheet`` and ``Operator_Ext_LorentzMaterial`` consume
those winners instead of re-querying CSXCAD per cell. Unsupported primitives and
genuine near-boundary ties fall back to the original CPU query, so no
approximate decision is used at an edge.

Supported primitives (untransformed, Cartesian only):

* ``BOX`` (Cartesian box)
* ``POLYGON``
* ``LINPOLY`` (linearly extruded polygon)
* ``CYLINDER``
* ``CYLINDRICALSHELL`` (via-like annuli)

Unsupported primitives, resolved per query via CSXCAD FP64:

* ``POINT``, ``MULTIBOX``
* ``SPHERE``, ``SPHERICALSHELL``
* ``ROTPOLY``, ``POLYHEDRON``, ``POLYHEDRONREADER``
* ``CURVE``, ``WIRE``, ``USERDEFINED``
* any primitive with a transform, a non-Cartesian input coordinate type, or a
  cylindrical coordinate system

Further behaviour:

* Yee coordinates and inclusive index ranges are computed in CPU FP64; polygon
  interiors use the CSXCAD winding rule with exact ``orient2d`` sign
  (double-float coordinates), with a conservative fallback band near the
  decision boundary.
* Unsupported primitives make only the affected queries use the CSXCAD path;
  they are never silently dropped, and can remove most of the speedup for such
  geometries.
* The conducting-sheet extension keeps per-cell sheet state (``sigma``,
  thickness, tangent direction) only for resolved sheet cells instead of three
  full-grid lookup tables (~19 GB on a 696M-cell model).

UPML
----

UPML is currently available only through the explicit legacy diagnostic update
(``OPENEMS_METAL_FUSED_PIPELINE=0``). It is never selected automatically for a
normal Metal run.

* **Indexed layout (default in the legacy diagnostic).** UPML coefficients and fluxes are permuted once
  into increasing packed-field addresses with a per-component ``uint32`` index.
  Stepping then does one indexed field access per lane and contiguous auxiliary
  I/O, with no per-step coordinate math. Reordering the access order is what
  produces the speedup; specializing integer divisors alone did not.
* **Scalar layout.** The original no-copy scalar kernel remains as a low-memory
  fallback; the two layouts must be bit-identical.
* **In-place reuse.** The operator's coefficient arrays are permuted in place and
  restored on teardown, so a later CPU or scalar engine sees valid data.
* **Lossless coefficient dictionaries.** UPML coefficient triples are
  deduplicated by exact 32-bit pattern; the dense CPU arrays are rebuilt on
  teardown for the FP64 diagnostic and later CPU engines.
* The mapping excludes SIMD padding lanes and skips zero-extent regions left by
  opposing slabs. Hooks keep CPU order (pre reverse priority, post forward) and
  pending GPU work is completed before any CPU hook, source, probe or dump.

Coefficient dictionaries
------------------------

The engine deduplicates each packed position's 48 FP32 values (VV, VI, II, IV ×
three components × four lanes) by exact 32-bit pattern, with a shared ``uint16``
index buffer. This reduces the GPU coefficient working set, **not** total process
RAM; index and dictionary copies happen only at initialization. Construction
falls back to dense reads when unique records exceed ``min(65536, positions/4)``
(the packed ``uint16`` index limit). Very nonuniform meshes can hit the limit and
retain dense reads automatically; initialization reports which path was taken.
Coefficients must remain immutable during stepping.

Conducting-sheet ADE
--------------------

The conducting-sheet model advances two ADE poles per active edge every step.
The Metal engine runs that recurrence in two kernels: ``ade_advance`` before the
voltage update and ``ade_apply`` from ``Apply2Voltages`` after it. One thread
owns all poles of one packed field edge, so the apply is race-free and matches
the CPU subtraction order. Previously the recurrence ran on the CPU and the
engine drained the GPU before each hook, serializing CPU and GPU.

Only the explicit legacy diagnostic path currently runs the plain volt-ADE
offload. ADE has not yet migrated into the diamond wavefront; a default Metal
run requiring ADE aborts before stepping. Models needing Lorentz flux states or ADE
currents (Lorentz, Drude, Debye) likewise require explicit legacy diagnostics.

Diagnostic overrides
--------------------

These environment variables exist for A/B testing and debugging. They all default
to the feature enabled and are not required to use the engine.

.. list-table::
   :header-rows: 1
   :widths: 34 12 54

   * - Variable
     - Default
     - Effect
   * - ``OPENEMS_METAL_PEC``
     - GPU
     - ``0`` = CPU PEC mapping, ``verify`` = GPU + CPU compare
   * - ``OPENEMS_METAL_PML``
     - on
     - ``0`` = CPU UPML conditioning
   * - ``OPENEMS_METAL_PML_LAYOUT``
     - ``indexed``
     - ``scalar`` selects the no-copy scalar kernel
   * - ``OPENEMS_METAL_PML_REUSE``
     - on
     - ``0`` stores separate packed copies
   * - ``OPENEMS_METAL_PML_COMPRESS``
     - on
     - ``0`` keeps dense UPML coefficients
   * - ``OPENEMS_METAL_COMPRESS``
     - on
     - ``0`` keeps dense operator coefficients
   * - ``OPENEMS_METAL_COEFF_RECORDS``
     - full range
     - lowers the dictionary limit
   * - ``OPENEMS_METAL_FUSED_PIPELINE``
     - on
     - ``0`` explicitly selects the legacy two-dispatch diagnostic path
   * - ``OPENEMS_METAL_SERIAL_COEFFICIENTS``
     - off
     - ``1`` builds coefficients single-threaded
   * - ``OPENEMS_METAL_EARLY_EC_FREE``
     - on
     - ``0`` keeps EC arrays before extensions
   * - ``OPENEMS_METAL_FP64_REFERENCE``
     - off
     - ``1`` enables the FP64 diagnostic

Validation
----------

.. code-block:: sh

   python macos/tests/metal_fields.py --openems /absolute/path/to/openEMS --suite
   python macos/tests/metal_pec.py --openems /absolute/path/to/openEMS
   python macos/tests/metal_conductingsheet.py --openems /absolute/path/to/openEMS
   python macos/tests/metal_dispersive.py --openems /absolute/path/to/openEMS
   OPENEMS_METAL_FUSED_PIPELINE=0 python macos/tests/metal_ade.py --openems /absolute/path/to/openEMS
   OPENEMS_METAL_FUSED_PIPELINE=0 python macos/tests/metal_pml.py --openems /absolute/path/to/openEMS

``metal_fields.py`` compares SSE against Metal with relative-L2 limits and
requires dense/compressed and diamond/explicit-legacy variants to be
bit-identical. ``--stress-sources`` adds overlapping sources spanning tile
boundaries. The PEC, conducting-sheet and dispersive suites compare CPU vs GPU
winner resolution and require bit-identical dumps. ``metal_ade.py`` compares the
GPU conducting-sheet ADE against the SSE CPU recurrence.

A separate performance harness lives in ``macos/bench/`` and is documented in
``macos/doc/metal-benchmark.rst``.

Limitations and fallbacks
-------------------------

The CPU stays the geometry authority wherever a GPU predicate would be
approximate: those affected PEC queries run through CSXCAD FP64 and are never
silently dropped. The field-update engine has a stricter contract: a normal
Metal run either constructs the in-place diamond kernel or aborts before timestep 0.
It never automatically substitutes the legacy E/H update or a CPU extension
path. The legacy path exists only behind an explicit diagnostic override.

Platform and coordinate systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Metal is macOS-only and built only with ``-DWITH_METAL=ON`` (off by default).
* A Metal device is required. If missing, openEMS logs the reason and exits
  before operator setup (material/PEC/coefficient passes are skipped) instead
  of falling back to the CPU.
* Cartesian meshes only. For a cylindrical mesh ``SetupOperator()`` selects the
  cylindrical operator regardless of ``--engine``, so ``--engine=metal`` is
  ignored with a warning.
* MPI is not supported (``Operator_Metal`` is not an MPI operator).

The index format has a very high ceiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPU kernels use 32-bit indices. The diamond field update addresses ``float4``
words, and the legacy indexed UPML, excitation and ADE kernels address scalar
components. The scalar limit is around ``UINT32_MAX / 3`` -- roughly
1.4 billion cells, about 100 GB of field plus coefficient state -- and the
packed field update is good for roughly four times that. A model beyond the
scalar limit is rejected in ``SetupCSXGrid``, before any material or coefficient
work. This guard is not reachable by any measured workload and is documented
only so the abort is not a surprise.

Kernel compilation
~~~~~~~~~~~~~~~~~~

The kernels are compiled once at build time into ``openEMS.metallib``, embedded
in ``libopenEMS`` and loaded with ``newLibraryWithData:``; the runtime never
compiles. Building with ``-DWITH_METAL=ON`` therefore needs the optional Metal
toolchain component, installed once with:

.. code-block:: sh

   xcodebuild -downloadComponent MetalToolchain

A load or pipeline-creation failure aborts. The PEC pass alone catches its own
failure and runs on the CPU. Fast math is off at compile time.

Geometry fallbacks and explicit legacy diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometry fallbacks print what was skipped and why. Field-update incompatibility
aborts unless the user explicitly requested the legacy diagnostic.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Trigger
     - Behaviour
   * - no Metal device
     - logs the reason and exits before operator setup (no PEC/coefficient
       passes)
   * - PEC queue/pipeline unavailable but a device exists
     - whole PEC pass on CPU; field updates still run on the GPU
   * - ``OPENEMS_METAL_PEC=0``, or a non-Cartesian mesh
     - whole PEC pass on CPU
   * - grid lines non-finite, ``|coord| > 1e10``, not sorted, or polygon with
       too many vertices
     - whole PEC pass on CPU
   * - transformed, non-Cartesian or unsupported primitive (see the list above)
     - per-query CPU resolution
   * - degenerated primitive coordinates, or polygon interior ambiguous
       (``orient2d`` sign zero)
     - per-query CPU resolution
   * - more than 65536 unique UPML coefficient records, or UPML buffer
       allocation fails
     - dense UPML coefficients for that region
   * - more than ``min(65536, positions/4)`` unique operator records
     - dense operator coefficients
   * - UPML, ADE, or an arbitrary CPU extension hook in a normal Metal run
     - aborts before timestep 0; these operations have not migrated inside the
       diamond wavefront
   * - ``OPENEMS_METAL_FUSED_PIPELINE=0``
     - explicitly runs the legacy two-dispatch diagnostic path
   * - ``OPENEMS_METAL_PML=0`` or ``OPENEMS_METAL_PML_LAYOUT=scalar`` together
       with explicit legacy mode
     - CPU or scalar-UPML diagnostic respectively
   * - ``OPENEMS_METAL_FP64_REFERENCE=1``
     - explicitly selects the legacy path plus diagnostic CPU reference

Geometry fallbacks are per query: one unsupported primitive does not disable the
whole pass, but such geometry (and the near-boundary band) can remove most of
the speedup. When the geometry winners are unavailable (for example with
``OPENEMS_METAL_PEC=0``), the conducting-sheet and dispersive extensions rebuild
their full-grid state tables instead of the sparse per-cell form, which costs
far more memory (about 19 GB on a 696M-cell model).
``CanReleaseECBeforeExtensions`` likewise releases the EC arrays only for exactly
UPML, excitation and Mur ABC extensions; any other extension keeps them resident.

Accuracy
~~~~~~~~

* SSE and Metal long-run comparison: 3895 E values exceed the default pointwise
  tolerance on the 1000-step cavity case (max abs 7.06e-5). The dense Metal path
  shows the same, so it is not a compression or UPML artifact.
* The FP64 reference is a diagnostic; with UPML it checks local stencils, not
  flux evolution.

Performance
~~~~~~~~~~~

* Small or simple geometries are submission-bound and can be slower on the GPU
  than on SSE.
* Regular grids compress exceptionally well; no speedup is claimed for real PCB
  models that hit the dense-coefficient fallback.
* Performance figures are finite-run measurements, not convergence or SI
  validation.

Representative M4 Pro measurements (Release, fast math off):

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - Workload
     - Result
   * - In-place diamond E/H
     - PEC, 3.0M cells, 600 steps
     - 2.30x over legacy Metal stepping; ~3 MB RSS delta
   * - In-place diamond E/H
     - PEC, 17.0M cells, 1000 steps
     - 1.25x over legacy Metal stepping; ~8 MB RSS delta
   * - Indexed UPML (explicit legacy diagnostic)
     - 658×664×33 board, 788 steps
     - ~1.79x stepping, ~1.18x process
   * - GPU PEC mapping
     - 60-pair CoSwitch pilot
     - ~8.7 s → 0.09 s PEC pass, ~10.9 s → 2.3 s setup
   * - GPU ADE
     - 240×240×10, 100800 sheet edges
     - ~1.26x stepping
