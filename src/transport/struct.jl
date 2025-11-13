# --------------------
# Data container
# --------------------
struct TranspoInstance
    F::Int
    D::Int
    N::Int
    pos_f::Matrix{Float64}   # F×2
    pos_d::Matrix{Float64}   # D×2
    costs::Matrix{Float64}   # F×D
    μ::Vector{Float64}       # D
    Ξ::Matrix{Float64}       # N×D
    m::Vector{Float64}       # F
    # ϵ::Float64
    seed::Int
end