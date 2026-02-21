using LinearAlgebra
using ForwardDiff
using IntervalArithmetic
using StaticArrays
using IntervalBoxes, IntervalArithmetic.Symbols

# ==========================================
# 1. Universal Unitary Transformation
# ==========================================
function get_local_coordinate_system(F, x)
    J = ForwardDiff.jacobian(F, x)
    F_svd = svd(J, full=true)
    
    M, N = size(J)
    dim_manifold = N - M
    
    Normals = F_svd.Vt[1:M, :]'
    Tangents = F_svd.Vt[M+1:end, :]'
    
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
    
    intervals = Vector{Interval{Float64}}(undef, N)
    for i in 1:dim_manifold
        intervals[i] = interval(-r_tangent, r_tangent)
    end
    for i in 1:M
        intervals[dim_manifold + i] = interval(-r_normal, r_normal)
    end
    
    local_to_global(p) = x_center + U * p
    
    J_center = ForwardDiff.jacobian(F, x_center)
    J_G_normal_center = J_center * U[:, dim_manifold+1:end]
    Y = inv(J_G_normal_center)
    
    J_G_box = ForwardDiff.jacobian(p -> F(local_to_global(p)), intervals)
    J_G_normal_box = J_G_box[:, dim_manifold+1:end]
    
    Id = Matrix{Float64}(I, M, M)
    Slope = Id - Y * J_G_normal_box 
    
    intervals_u = zeros(Interval{Float64}, N)
    residual = -Y * F(local_to_global(intervals_u))
    
    W_box = [intervals[dim_manifold+i] for i in 1:M]
    K_output = residual + Slope * W_box
    
    is_subset = all(issubset_interval(K_output[i], tol*W_box[i]) for i in 1:M)
    
    return is_subset, K_output
end

# ==========================================
# 3. Newton Refinement (Normal Direction)
# ==========================================
function refine_point(F, x_approx, U; tol=1e-12, max_iter=10)
    M = length(F(x_approx))
    N = length(x_approx)
    dim_manifold = N - M
    
    Normals = U[:, dim_manifold+1:end]
    x_curr = copy(x_approx)
    
    for i in 1:max_iter
        val = F(x_curr)
        if norm(val) < tol
            return x_curr, true
        end
        
        J = ForwardDiff.jacobian(F, x_curr)
        J_normal = J * Normals
        delta_w = -(J_normal \ val)
        
        x_curr += Normals * delta_w
    end
    
    return x_curr, false
end

# ==========================================
# 4. Refine and Realign
# ==========================================
function refine_and_realign(F, x_approx, init_r; r_min=1e-10, r_max=0.5, rho=7/8)
    curr_x = copy(x_approx)
    curr_r = init_r
    
    U_temp, _ = get_local_coordinate_system(F, curr_x)
    curr_x, converged = refine_point(F, curr_x, U_temp)
    
    if !converged
        return curr_x, U_temp, r_min, false
    end

    U_curr, dim = get_local_coordinate_system(F, curr_x)
    is_safe, _ = krawczyk_test(F, curr_x, U_curr, curr_r, curr_r; tol=rho)
    
    if !is_safe
        while curr_r > r_min
            curr_r *= 0.5
            is_safe, _ = krawczyk_test(F, curr_x, U_curr, curr_r, curr_r; tol=rho)
            if is_safe; break; end
        end
    else
        while curr_r < r_max
            test_r = curr_r * 5/4 
            if test_r > r_max; break; end

            safe_big, _ = krawczyk_test(F, curr_x, U_curr, test_r, test_r; tol=rho)
            if safe_big
                curr_r = test_r
            else
                break
            end
        end
    end

    return curr_x, U_curr, curr_r, is_safe
end

# ==========================================
# 5. Core Data Structures and Paving Loop
# ==========================================
struct CertifiedTile
    center::Vector{Float64}
    U::Matrix{Float64}      
    r::Float64              
    dim::Int
end

function get_corners(tile::CertifiedTile)
    u_vec = tile.U[:, 1]
    v_vec = tile.U[:, 2]
    c = tile.center
    r = tile.r
    
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

function pave_surface_fixed(F, start_point; max_tiles=500, init_r=0.1, rho=7/8)
    p_cert, U_curr, r_cert, is_safe = refine_and_realign(F, start_point, init_r; rho=rho)
    if !is_safe; error("The initial point is not certified"); end
    
    queue = []
    push!(queue, (pt=p_cert, r=r_cert, parent=nothing, v_in=nothing))
    tiles = Vector{CertifiedTile}()
    
    while !isempty(queue) && length(tiles) < max_tiles
        item = popfirst!(queue)
        
        if is_overlapping(item.pt, tiles, 0.7); continue; end
        
        p_new, U_new, r_new, success = refine_and_realign(F, item.pt, item.r; rho=rho)
        
        if !success || r_new < 1e-4; continue; end
        if is_overlapping(p_new, tiles, 0.7); continue; end
        
        push!(tiles, CertifiedTile(p_new, U_new, r_new, 2))
        
        Tangents = U_new[:, 1:2]
        step_dist = r_new * 1.6 
        
        for i in 1:2
            for sgn in [1.0, -1.0]
                v_out = Tangents[:, i] * sgn
                next_pt = p_new + v_out * step_dist
                push!(queue, (pt=next_pt, r=r_new, parent=p_new, v_in=v_out))
            end
        end
    end
    return tiles
end

# ==========================================
# 6. Predictors
# ==========================================
function predictor_linear(x, v, h)
    return x + h * v
end

function predictor_hermite(x_cur, v_cur, x_prev, v_prev, h_prev, h_new)
    h_sq = h_prev^2
    h_cu = h_prev^3
    
    diff_x = x_cur - x_prev
    diff_v = v_cur - v_prev
    
    inv_h = 1.0 / h_prev
    inv_h2 = inv_h * inv_h
    inv_h3 = inv_h2 * inv_h
    
    t = h_new
    t2 = t^2
    t3 = t^3
    
    term_linear = v_cur * t
    
    c2 = (3 * v_cur * inv_h) - (diff_v * inv_h) - (3 * diff_x * inv_h2)
    c3 = (2 * v_cur * inv_h2) - (diff_v * inv_h2) - (2 * diff_x * inv_h3)
    
    return x_cur + term_linear + c2 * t2 + c3 * t3
end

# ==========================================
# 7. 3D Visualization Export Tools
# ==========================================
function get_corners_3d(tile::CertifiedTile)
    c = tile.center
    r = tile.r
    
    vec_u = tile.U[:, 1] * r
    vec_v = tile.U[:, 2] * r
    vec_n = tile.U[:, 3] * r 

    p1 = c - vec_u - vec_v - vec_n
    p2 = c + vec_u - vec_v - vec_n
    p3 = c + vec_u + vec_v - vec_n
    p4 = c - vec_u + vec_v - vec_n
    
    p5 = c - vec_u - vec_v + vec_n
    p6 = c + vec_u - vec_v + vec_n
    p7 = c + vec_u + vec_v + vec_n
    p8 = c - vec_u + vec_v + vec_n

    return [p1, p2, p3, p4, p5, p6, p7, p8]
end

function export_boxes_to_obj(tiles, filename="surface_boxes_3d.obj")
    open(filename, "w") do io
        println(io, "# Certified Surface 3D Boxes Output")
        vc = 1 

        for tile in tiles
            corners = get_corners_3d(tile)
            
            for p in corners
                println(io, "v $(p[1]) $(p[2]) $(p[3])")
            end
            
            println(io, "f $(vc+0) $(vc+1) $(vc+2) $(vc+3)") 
            println(io, "f $(vc+7) $(vc+6) $(vc+5) $(vc+4)") 
            println(io, "f $(vc+0) $(vc+4) $(vc+5) $(vc+1)") 
            println(io, "f $(vc+1) $(vc+5) $(vc+6) $(vc+2)") 
            println(io, "f $(vc+2) $(vc+6) $(vc+7) $(vc+3)") 
            println(io, "f $(vc+3) $(vc+7) $(vc+4) $(vc+0)") 
            
            vc += 8
        end
    end
end

function export_boxes_to_tikz(tiles, filename="surface_boxes.tex"; max_tiles_tikz=200)
    open(filename, "w") do io
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
        
        tiles_to_draw = length(tiles) > max_tiles_tikz ? tiles[1:max_tiles_tikz] : tiles
        
        for tile in tiles_to_draw
            cs = get_corners_3d(tile)
            toc(p) = "($(p[1]),$(p[2]),$(p[3]))"
            style = "fill=blue!30, fill opacity=0.4, draw=blue!80, very thin"
            
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[4])) -- cycle;") 
            println(io, "\\draw[$style] $(toc(cs[5])) -- $(toc(cs[6])) -- $(toc(cs[7])) -- $(toc(cs[8])) -- cycle;") 
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[6])) -- $(toc(cs[5])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[7])) -- $(toc(cs[6])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[3])) -- $(toc(cs[4])) -- $(toc(cs[8])) -- $(toc(cs[7])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[4])) -- $(toc(cs[1])) -- $(toc(cs[5])) -- $(toc(cs[8])) -- cycle;")
        end
        
        println(io, raw"""
\end{tikzpicture}
\end{document}
""")
    end
end

function export_saddle_boxes_to_tikz(tiles, filename="surface_boxes.tex"; max_tiles_tikz=200)
    open(filename, "w") do io
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

    \foreach \xv in {-2, -1.5, ..., 2} {
        \draw[red!80!black, very thin, opacity=0.4] 
            plot[domain=-2:2, samples=30, variable=\yv] 
            ({\xv}, {\yv}, {0.25*\xv*\xv - 0.125*\xv*\yv*\yv});
    }
    
    \foreach \yv in {-2, -1.5, ..., 2} {
        \draw[red!80!black, very thin, opacity=0.4] 
            plot[domain=-2:2, samples=30, variable=\xv] 
            ({\xv}, {\yv}, {0.25*\xv*\xv - 0.125*\xv*\yv*\yv});
    }
""")
        
        tiles_to_draw = length(tiles) > max_tiles_tikz ? tiles[1:max_tiles_tikz] : tiles
        
        for tile in tiles_to_draw
            cs = get_corners_3d(tile)
            toc(p) = "($(p[1]),$(p[2]),$(p[3]))"
            style = "fill=blue!30, fill opacity=0.4, draw=blue!80, very thin"
            
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[4])) -- cycle;") 
            println(io, "\\draw[$style] $(toc(cs[5])) -- $(toc(cs[6])) -- $(toc(cs[7])) -- $(toc(cs[8])) -- cycle;") 
            println(io, "\\draw[$style] $(toc(cs[1])) -- $(toc(cs[2])) -- $(toc(cs[6])) -- $(toc(cs[5])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[2])) -- $(toc(cs[3])) -- $(toc(cs[7])) -- $(toc(cs[6])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[3])) -- $(toc(cs[4])) -- $(toc(cs[8])) -- $(toc(cs[7])) -- cycle;")
            println(io, "\\draw[$style] $(toc(cs[4])) -- $(toc(cs[1])) -- $(toc(cs[5])) -- $(toc(cs[8])) -- cycle;")
        end
        
        println(io, raw"""
\end{tikzpicture}
\end{document}
""")
    end
end
