using LinearAlgebra
using ForwardDiff
using IntervalArithmetic
using StaticArrays


using IntervalBoxes, IntervalArithmetic.Symbols


# ==========================================
# 1. Universal Unitary Transformation
# ==========================================
function get_local_coordinate_system(F, x)
    # 1. Compute Jacobian
    J = ForwardDiff.jacobian(F, x)
    
    # 2. Perform SVD (full=true is required)
    F_svd = svd(J, full=true)
    
    # 3. Dimension analysis (#variables N - #equations M)
    M, N = size(J)
    dim_manifold = N - M
    
    # 4. Extract basis vectors
    Normals = F_svd.Vt[1:M, :]'      # N x M
    Tangents = F_svd.Vt[M+1:end, :]' # N x (N-M)
    
    # 5. Build U matrix: [Tangent | Normal]
    U = hcat(Tangents, Normals)
    
    return U, dim_manifold
end

# ==========================================
# 2. Universal Krawczyk Test 
# ==========================================
function krawczyk_test(F, x_center, U, r_tangent, r_normal; tol=7/8)
    M = length(F(x_center))
    N = length(x_center)
    dim_manifold = N - M
    
    # 1. Build local interval box 
    intervals = Vector{Interval{Float64}}(undef, N)
    
    for i in 1:dim_manifold
        intervals[i] = interval(-r_tangent, r_tangent)
    end
    for i in 1:M
        intervals[dim_manifold + i] = interval(-r_normal, r_normal)
    end
    
    # 2. Transformation function (Local -> Global)
    local_to_global(p) = x_center + U * p
    
    # 3. Build Krawczyk operator
    # 4-1. Preconditioner Y (inverse of Jacobian restricted to normal space)
    J_center = ForwardDiff.jacobian(F, x_center)
    J_G_normal_center = J_center * U[:, dim_manifold+1:end]
    Y = inv(J_G_normal_center)
    
    # 4-2. Slope matrix (Mean Value Form)
    J_G_box = ForwardDiff.jacobian(p -> F(local_to_global(p)), intervals)
    
    J_G_normal_box = J_G_box[:, dim_manifold+1:end]
    Id = Matrix{Float64}(I, M, M)
    Slope = Id - Y * J_G_normal_box 
    
    # 4-3. Residual calculation (-Y * G(u, 0))
    intervals_u = zeros(Interval{Float64}, N)
#    for i in 1:M
#        intervals_u[dim_manifold + i] = 0
#    end
    
    residual = -Y * F(local_to_global(intervals_u))
    
    # 4-4. Krawczyk output check
    # W_box (normal-part intervals)
    W_box = [intervals[dim_manifold+i] for i in 1:M]
    
    # K(w) = residual + Slope * w
    K_output = residual + Slope * W_box
    
    # Check subset condition (issubset)
    is_subset = all(issubset_interval(K_output[i], tol*W_box[i]) for i in 1:M)
    
    return is_subset, K_output
end

# ==========================================
# 3. Newton Refinement (Normal Direction Correction)
# ==========================================
# Goal: move point x onto the surface F(x)=0 using only the normal direction
function refine_point(F, x_approx, U; tol=1e-12, max_iter=10)
    M = length(F(x_approx))
    N = length(x_approx)
    dim_manifold = N - M
    
    # Extract only the normal-direction vectors from U
    # (We only want to correct along these directions.)
    Normals = U[:, dim_manifold+1:end] # N x M matrix
    
    x_curr = copy(x_approx)
    
    for i in 1:max_iter
        # 1. Compute current residual
        val = F(x_curr)
        if norm(val) < tol
            return x_curr, true # Converged
        end
        
        # 2. Compute Jacobian
        J = ForwardDiff.jacobian(F, x_curr)
        
        # 3. Normal-direction Jacobian (M x M)
        # J_normal = J * Normals
        # This must be nonsingular (if local coordinates are well-defined).
        J_normal = J * Normals
        
        # 4. Newton update (normal direction only!)
        # Delta_w = - (J * N)^-1 * F(x)
        # Delta_x = N * Delta_w
        delta_w = -(J_normal \ val) # Linear solve
        
        x_curr += Normals * delta_w
    end
    
    return x_curr, false # Failed to converge (max_iter reached)
end


# ==========================================
# Combines Algorithm 2 with Algorithm 3
# ==========================================
function refine_and_realign(F, x_approx, init_r; r_min=1e-10, r_max=0.5, rho=7/8)
    curr_x = copy(x_approx)
    curr_r = init_r
#    println("Initial radius: $curr_r")
    
    # 1. First, project the point accurately onto the surface (Newton refinement)
    # If the point is far from the surface, Krawczyk will always fail, so pull it in first.
    U_temp, _ = get_local_coordinate_system(F, curr_x)
    curr_x, converged = refine_point(F, curr_x, U_temp)
    
    if !converged
        return curr_x, U_temp, r_min, false
    end

    # 2. Recompute the optimized local coordinate system (U) at this point (Re-alignment)
    # [Key in Algorithms 2 & 4] Recompute the normal at the moved point.
    U_curr, dim = get_local_coordinate_system(F, curr_x)

    # 3. Use Krawczyk test to validate a safe radius and shrink if needed (Shrink)
    # Check whether the box is safe at the current location using U_curr.
    is_safe, _ = krawczyk_test(F, curr_x, U_curr, curr_r, curr_r; tol=rho)
    if !is_safe
        while curr_r > r_min
            curr_r *= 0.5
            is_safe, _ = krawczyk_test(F, curr_x, U_curr, curr_r, curr_r; tol=rho)
            if is_safe; break; end
        end
    else
        # 4. [Algorithm 3 Line 14] If successful, grow as much as possible (Grow)
        # This is needed to prevent the box size from shrinking on surfaces like spheres.
        while curr_r < r_max
            test_r = curr_r * 5/4 
#            println("Current radius: $curr_r, trying to grow to $test_r")
            if test_r > r_max; break; end

            safe_big, _ = krawczyk_test(F, curr_x, U_curr, test_r, test_r; tol=rho)
            println("Trying to grow radius to $test_r : safe = $safe_big")
            if safe_big
                curr_r = test_r
            else
                break
            end
        end
    end

    return curr_x, U_curr, curr_r, is_safe
end


# 1. The struct stores data only.
struct CertifiedTile
    center::Vector{Float64}
    U::Matrix{Float64}      # Local frame [u, v, n]
    r::Float64              # Certified radius
    dim::Int
end

# 2. Define functions outside the struct. (Multiple dispatch)
function get_corners(tile::CertifiedTile)
    # 4 corners in local coordinates (assume 2D)
    # p1(-r, -r), p2(r, -r), p3(r, r), p4(-r, r)
    u_vec = tile.U[:, 1]
    v_vec = tile.U[:, 2]
    c = tile.center
    r = tile.r
    
    # Convert to global coordinates
    return [
        c - r*u_vec - r*v_vec,
        c + r*u_vec - r*v_vec,
        c + r*u_vec + r*v_vec,
        c - r*u_vec + r*v_vec
    ]
end
function is_overlapping(pt, tiles, threshold_ratio=0.8)

    for tile in tiles
        if norm(pt - tile.center) < tile.r * threshold_ratio
            return true
        end
    end
    return false
end

# ==========================================
# Surface Paving main loop
# ==========================================
function pave_surface_fixed(F, start_point; max_tiles=500, init_r=0.1, rho=7/8)
    # Refine the initial point and get a local frame
    p_cert, U_curr, r_cert, is_safe = refine_and_realign(F, start_point, init_r; rho=rho)
    
    if !is_safe; error("the initial point is not certified"); end
    
    queue = []
    # (current point, suggested radius, parent point, incoming velocity)
    push!(queue, (pt=p_cert, r=r_cert, parent=nothing, v_in=nothing))
    
    tiles = Vector{CertifiedTile}()
    
    while !isempty(queue) && length(tiles) < max_tiles
        item = popfirst!(queue)
        
        # Check if already covered (avoid duplicates)
        if is_overlapping(item.pt, tiles, 0.7); continue; end
        
        # [Key] Refine and "re-align the local frame" at the predicted point
        p_new, U_new, r_new, success = refine_and_realign(F, item.pt, item.r; rho=rho)
        
        if !success || r_new < 1e-4; continue; end
        if is_overlapping(p_new, tiles, 0.7); continue; end
        
        # Store tile
        push!(tiles, CertifiedTile(p_new, U_new, r_new, 2))
        
        # Expand (4 directions)
        Tangents = U_new[:, 1:2]
        # Move slightly less than the side length (2r) so tiles interlock
        step_dist = r_new * 1.6 
        
        for i in 1:2
            for sgn in [1.0, -1.0]
                v_out = Tangents[:, i] * sgn
                
                # Prediction (Linear or Hermite)
                next_pt = p_new + v_out * step_dist
                
                # Pass information to next step
                push!(queue, (pt=next_pt, r=r_new, parent=p_new, v_in=v_out))
            end
        end
    end
    return tiles
end



# ==========================================
# Predictors (Linear & Hermite) #### These are not used currently, but kept for future use.
# ==========================================

# 4-1. Linear predictor (for early steps)
# x: current point
# v: tangent direction to move (unit vector)
# h: step size
function predictor_linear(x, v, h)
    return x + h * v
end

# 4-2. 3rd order Hermite predictor (accelerated tracking)
# x_cur, v_cur: current point and velocity (tangent)
# x_prev, v_prev: previous point and velocity
# h_prev: previous step size (distance from x_prev to x_cur)
# h_new: next step size
function predictor_hermite(x_cur, v_cur, x_prev, v_prev, h_prev, h_new)
    # Use h_new directly instead of time variable t.
    # Hermite interpolation (cubic) matching:
    # at t=0: current, at t=-h_prev: previous.
    # P(t) = x_cur + v_cur*t + c2*t^2 + c3*t^3
    
    # Precompute powers
    h_sq = h_prev^2
    h_cu = h_prev^3
    
    diff_x = x_cur - x_prev
    diff_v = v_cur - v_prev
    
    inv_h = 1.0 / h_prev
    inv_h2 = inv_h * inv_h
    inv_h3 = inv_h2 * inv_h
    
    # t = h_new (future time we want to predict)
    t = h_new
    t2 = t^2
    t3 = t^3
    
    term_linear = v_cur * t
    
    c2 = (3 * v_cur * inv_h) - (diff_v * inv_h) - (3 * diff_x * inv_h2)
    c3 = (2 * v_cur * inv_h2) - (diff_v * inv_h2) - (2 * diff_x * inv_h3)
    
    x_pred = x_cur + term_linear + c2 * t2 + c3 * t3
    
    return x_pred
end

# ==========================================
# 3D box (Parallelepiped) visualization tools
# ==========================================

# Compute the 8 vertices of the 3D box
# Combine +/- r along u, v, n directions from the tile center
function get_corners_3d(tile::CertifiedTile)
    c = tile.center
    r = tile.r
    # Scale local basis vectors by radius
    vec_u = tile.U[:, 1] * r
    vec_v = tile.U[:, 2] * r
    vec_n = tile.U[:, 3] * r # In 3D, we also need the normal vector (n)!

    # 8 vertices (bottom 4, then top 4)
    # Bottom face (at -r along normal)
    p1 = c - vec_u - vec_v - vec_n
    p2 = c + vec_u - vec_v - vec_n
    p3 = c + vec_u + vec_v - vec_n
    p4 = c - vec_u + vec_v - vec_n
    
    # Top face (at +r along normal)
    p5 = c - vec_u - vec_v + vec_n
    p6 = c + vec_u - vec_v + vec_n
    p7 = c + vec_u + vec_v + vec_n
    p8 = c - vec_u + vec_v + vec_n

    return [p1, p2, p3, p4, p5, p6, p7, p8]
end

# [OBJ] Export 3D boxes
function export_boxes_to_obj(tiles, filename="surface_boxes_3d.obj")
    open(filename, "w") do io
        println(io, "# Certified Surface 3D Boxes Output")
        vc = 1 # vertex count start index

        for tile in tiles
            corners = get_corners_3d(tile)
            
            # 1. Write 8 vertices
            for p in corners
                println(io, "v $(p[1]) $(p[2]) $(p[3])")
            end
            
            # 2. Write 6 faces (quads)
            # OBJ uses 1-based indexing. Define each face CCW.
            println(io, "f $(vc+0) $(vc+1) $(vc+2) $(vc+3)") # Bottom
            println(io, "f $(vc+7) $(vc+6) $(vc+5) $(vc+4)") # Top
            println(io, "f $(vc+0) $(vc+4) $(vc+5) $(vc+1)") # Front
            println(io, "f $(vc+1) $(vc+5) $(vc+6) $(vc+2)") # Right
            println(io, "f $(vc+2) $(vc+6) $(vc+7) $(vc+3)") # Back
            println(io, "f $(vc+3) $(vc+7) $(vc+4) $(vc+0)") # Left
            
            vc += 8
        end
    end
    println("[OBJ] 3D obj file saved: $filename")
end
# ==========================================
# [Update] TikZ Export with Surface Wireframe
# ==========================================
function export_boxes_to_tikz(tiles, filename="surface_boxes.tex"; max_tiles_tikz=200)
    open(filename, "w") do io
        # Header
        println(io, raw"""
\documentclass[tikz,border=2pt,png]{standalone}
\usepackage{tikz-3dplot}
\begin{document}
\tdplotsetmaincoords{60}{130} 
\begin{tikzpicture}[tdplot_main_coords, scale=2]
    \coordinate (O) at (0,0,0);
    \draw[thick,->] (O) -- (1.5,0,0) node[anchor=north east]{$x$};
    \draw[thick,->] (O) -- (0,1.5,0) node[anchor=north west]{$y$};
    \draw[thick,->] (O) -- (0,0,1.5) node[anchor=south]{$z$};
    % --- True Surface (Red Wireframe) ---
    % Sphere Equation: x=sin(p)cos(t), y=sin(p)sin(t), z=cos(p)
    \foreach \phi in {0, 15, ..., 180} {
        \draw[red!80!black, very thin, opacity=0.4] 
            plot[domain=0:360, samples=60, variable=\theta] 
            ({sqrt(10)*sin(\phi)*cos(\theta)}, {sqrt(10)*sin(\phi)*sin(\theta)}, {sqrt(10)*cos(\phi)});
    }
    \foreach \theta in {0, 30, ..., 180} {
        \draw[red!80!black, very thin, opacity=0.4] 
            plot[domain=0:360, samples=60, variable=\phi] 
            ({sqrt(10)*sin(\phi)*cos(\theta)}, {sqrt(10)*sin(\phi)*sin(\theta)}, {sqrt(10)*cos(\phi)});
    }
""")
        
        # Boxes
        tiles_to_draw = length(tiles) > max_tiles_tikz ? tiles[1:max_tiles_tikz] : tiles
        println(io, raw"% --- Certified Boxes ---")
        
        for tile in tiles_to_draw
            cs = get_corners_3d(tile)
            toc(p) = "($(p[1]),$(p[2]),$(p[3]))"
            style = "fill=blue!30, fill opacity=0.4, draw=blue!80, very thin"
            
            # Draw 6 faces
            # Bottom
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[4])) -- cycle;") 
            # Top
            println(io, "\\draw[$style] $(toc(cs[5])) -- $(toc(cs[6])) -- $(toc(cs[7])) -- $(toc(cs[8])) -- cycle;") 
            # Sides
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[6])) -- $(toc(cs[5])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[7])) -- $(toc(cs[6])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[3])) -- $(toc(cs[4])) -- $(toc(cs[8])) -- $(toc(cs[7])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[4])) -- $(toc(cs[1])) -- $(toc(cs[5])) -- $(toc(cs[8])) -- cycle;")
        end
        
        # Footer
        println(io, raw"""
\end{tikzpicture}
\end{document}
""")
    end
    println("[TikZ] LaTeX file saved: $filename")
end


# Run paving 
# Test function (Surface)
function test_surface_eq(x)
    return SVector(x[1]^2 + x[2]^2 + x[3]^2 - 10.0)
end


p_init = [0.0, 0.0, 3.16]
tiles = pave_surface_fixed(test_surface_eq, p_init, max_tiles=2000, init_r=0.1;rho=1/8)

# 1. Export 3D OBJ (for checking full data)
export_boxes_to_obj(tiles, "surface_boxes_3d.obj")

# 2. Export TikZ tex (high-quality figure for paper/report)
# TikZ rendering is slow, so limit the number of boxes with an option.
export_boxes_to_tikz(tiles, "surface_visualization.tex"; max_tiles_tikz=2000)



# for saddle surface
function export_saddle_boxes_to_tikz(tiles, filename="surface_boxes.tex"; max_tiles_tikz=200)
    open(filename, "w") do io
        # Header
        println(io, raw"""
\documentclass[tikz,border=2pt,png]{standalone}
\usepackage{tikz-3dplot}
\begin{document}
\tdplotsetmaincoords{60}{130} 
\begin{tikzpicture}[tdplot_main_coords, scale=2]
    \coordinate (O) at (0,0,0);
    \draw[thick,->] (O) -- (1.5,0,0) node[anchor=north east]{$x$};
    \draw[thick,->] (O) -- (0,1.5,0) node[anchor=north west]{$y$};
    \draw[thick,->] (O) -- (0,0,1.5) node[anchor=south]{$z$};

    % --- True Surface (Red Wireframe) ---
    \foreach \xv in {-2, -1.5, ..., 2} {
        \draw[red!80!black, very thin, opacity=0.4] 
            plot[domain=-2:2, samples=30, variable=\yv] 
            ({\xv}, {\yv}, {0.25*\xv*\xv - 0.125*\xv*\yv*\yv});
    }
    
    % 2. Fix y and draw curves along the x-direction.
    \foreach \yv in {-2, -1.5, ..., 2} {
        \draw[red!80!black, very thin, opacity=0.4] 
            plot[domain=-2:2, samples=30, variable=\xv] 
            ({\xv}, {\yv}, {0.25*\xv*\xv - 0.125*\xv*\yv*\yv});
    }

""")
        
        # Boxes
        tiles_to_draw = length(tiles) > max_tiles_tikz ? tiles[1:max_tiles_tikz] : tiles
        println(io, raw"% --- Certified Boxes ---")
        
        for tile in tiles_to_draw
            cs = get_corners_3d(tile)
            toc(p) = "($(p[1]),$(p[2]),$(p[3]))"
            style = "fill=blue!30, fill opacity=0.4, draw=blue!80, very thin"
            
            # Draw 6 faces
            # Bottom
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[4])) -- cycle;") 
            # Top
            println(io, "\\draw[$style] $(toc(cs[5])) -- $(toc(cs[6])) -- $(toc(cs[7])) -- $(toc(cs[8])) -- cycle;") 
            # Sides
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[6])) -- $(toc(cs[5])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[7])) -- $(toc(cs[6])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[3])) -- $(toc(cs[4])) -- $(toc(cs[8])) -- $(toc(cs[7])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[4])) -- $(toc(cs[1])) -- $(toc(cs[5])) -- $(toc(cs[8])) -- cycle;")
        end
        
        # Footer
        println(io, raw"""
\end{tikzpicture}
\end{document}
""")
    end
    println("[TikZ] LaTeX file saved: $filename")
end
function saddle_eq(v)
    x, y, z = v[1], v[2], v[3]
    
    # Move everything to one side so the equation equals 0
    eq = -0.125 * x * y^2 + 0.25 * x^2 - z
    
    return SVector(eq)
end


# Find a starting point:
# Plug in x=2, y=2: z = -0.125*2*4 + 0.25*4 = -1 + 1 = 0
# So (2, 2, 0) lies on the surface.
p_init_saddle = [2,2,0]

# Run paving
# This surface is not a closed shape (torus/sphere); it extends infinitely,
# so without limiting max_tiles it will keep paving forever.
tiles_saddle = pave_surface_fixed(saddle_eq, p_init_saddle, max_tiles=2000, init_r=0.1)
export_boxes_to_obj(tiles_saddle, "saddle_surface_boxes_3d.obj")

export_saddle_boxes_to_tikz(tiles_saddle, "saddle_surface_boxes.tex", max_tiles_tikz=200)

# Torus surface
function torus_eq(x)
    R = 2.0 # Major radius (overall donut radius)
    r = 0.8 # Minor radius (tube thickness)
    
    # Distance to tube center in the xy-plane
    dist_xy = sqrt(x[1]^2 + x[2]^2)
    
    return SVector((dist_xy - R)^2 + x[3]^2 - r^2)
end

# Execution & visualization
# Starting point: (R+r, 0, 0) -> outermost point of the donut
p_init_torus = [2.8, 0.0, 0.0] 

tiles_torus = pave_surface_fixed(torus_eq, p_init_torus, max_tiles=2500, init_r=0.01)

# Export OBJ (for viewing in MeshLab)
export_boxes_to_obj(tiles_torus, "torus_boxes_3d.obj")
export_boxes_to_tikz(tiles_torus, "torus_boxes_3d.tex", max_tiles_tikz=2500)

