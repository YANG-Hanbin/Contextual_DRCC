# --------------------
# Helpers
# --------------------
function euclidean_costs(pos_f::AbstractMatrix, pos_d::AbstractMatrix)
    F = size(pos_f, 1); D = size(pos_d, 1)
    C = Matrix{Float64}(undef, F, D)
    @inbounds for f in 1:F
        fx, fy = pos_f[f,1], pos_f[f,2]
        for d in 1:D
            dx, dy = pos_d[d,1], pos_d[d,2]
            C[f,d] = hypot(fx - dx, fy - dy)
        end
    end
    return C
end

function default_thetas(; θ1::Float64=0.001, n::Int=10)
    return collect(θ1:θ1:θ1*n)
end
