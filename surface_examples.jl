include("surface_core.jl")
# ==========================================
# 8. Execution / Test Cases
# ==========================================

# 1. Sphere Surface
function test_surface_eq(x)
    return SVector(x[1]^2 + x[2]^2 + x[3]^2 - 10.0)
end

p_init = [0.0, 0.0, 3.16]
tiles = pave_surface_fixed(test_surface_eq, p_init, max_tiles=2000, init_r=0.1; rho=1/8)
export_boxes_to_obj(tiles, "surface_boxes_3d.obj")
export_boxes_to_tikz(tiles, "surface_visualization.tex"; max_tiles_tikz=2000)

# 2. Saddle Surface
function saddle_eq(v)
    x, y, z = v[1], v[2], v[3]
    eq = -0.125 * x * y^2 + 0.25 * x^2 - z
    return SVector(eq)
end

p_init_saddle = [2, 2, 0]
tiles_saddle = pave_surface_fixed(saddle_eq, p_init_saddle, max_tiles=2000, init_r=0.1)
export_boxes_to_obj(tiles_saddle, "saddle_surface_boxes_3d.obj")
export_saddle_boxes_to_tikz(tiles_saddle, "saddle_surface_boxes.tex", max_tiles_tikz=200)

# 3. Torus Surface
function torus_eq(x)
    R = 2.0 
    r = 0.8 
    dist_xy = sqrt(x[1]^2 + x[2]^2)
    return SVector((dist_xy - R)^2 + x[3]^2 - r^2)
end

p_init_torus = [2.8, 0.0, 0.0] 
tiles_torus = pave_surface_fixed(torus_eq, p_init_torus, max_tiles=2500, init_r=0.01)
export_boxes_to_obj(tiles_torus, "torus_boxes_3d.obj")
export_boxes_to_tikz(tiles_torus, "torus_boxes_3d.tex", max_tiles_tikz=2500)
