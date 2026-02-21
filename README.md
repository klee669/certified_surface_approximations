# Certified Surface Paving

This repository contains the Julia implementation of the certified surface approximation algorithm presented in the paper ["Certified surface approximations using the interval Krawczyk test"](https://arxiv.org/abs/2602.07718). It uses interval arithmetic and the Krawczyk operator to compute certified coverings of implicit surfaces.

## Repository Structure

* `surface_core.jl`: Contains the core algorithms including the Universal Unitary Transformation, Universal Krawczyk Test, Newton Refinement, and the main paving loop.
* `surface_examples.jl`: Contains execution scripts and test cases for three different surfaces (Sphere, Saddle, Torus).

## Dependencies

To run the code, you need to install [Julia](https://julialang.org/downloads/) and the following required packages. You can install them via the Julia REPL:

```julia
using Pkg
Pkg.add(["LinearAlgebra", "ForwardDiff", "IntervalArithmetic", "StaticArrays", "IntervalBoxes"])

```

## Quick Start

You can run the examples directly from your terminal or the Julia REPL.

```julia
include("surface_examples.jl")

```

Running the script will automatically compute the certified tiles for the three test surfaces and generate the corresponding 3D model files and LaTeX visualization files.

## Test Cases

The `surface_examples.jl` script includes three implicit surfaces:

1. **Sphere**: 
2. **Saddle**: 
3. **Torus**: 

You can easily modify `max_tiles` (maximum number of boxes) or `init_r` (initial radius) inside the script to test different resolutions.

## Visualization Outputs

The algorithm exports the certified tiles in two formats for easy visualization:

* **`.obj` files** (e.g., `surface_boxes_3d.obj`): A standard 3D object format. You can open these files using standard 3D viewers [MeshLab](https://www.meshlab.net/) to inspect the full set of 3D boxes.
* **`.tex` files** (e.g., `surface_visualization.tex`): A standalone LaTeX file using the `tikz-3dplot` package.
  
## How to Pave a Custom Surface

To pave your own implicit surface, define your function to return a `SVector` and call `pave_surface_fixed`:

```julia
# 1. Define the equation (e.g., F(x, y, z) = 0)
function my_surface_eq(x)
    return SVector(x[1]^2 + x[2]^2 - x[3]) # Example: Paraboloid
end

# 2. Provide a valid initial point on the surface
p_init = [2.0, 2.0, 8.0]

# 3. Run the paving algorithm
tiles = pave_surface_fixed(my_surface_eq, p_init, max_tiles=1000, init_r=0.1)

# 4. Export the results
export_boxes_to_obj(tiles, "my_surface.obj")

```

