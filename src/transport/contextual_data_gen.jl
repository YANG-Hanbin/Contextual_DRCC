# ------------------------------------------------------------
# Probabilistic Transportation Problem — Data Generator
# ------------------------------------------------------------

# Contextual-demand instance generator
make_beta_master(Dmax::Int, K::Int; seed_beta::Int=42, β_scale::Float64=0.05) =
    β_scale .* randn(MersenneTwister(seed_beta), Dmax, K)


function generate_instance(
    F::Int, 
    D::Int, 
    N::Int; 
    seed::Int=1, 
    box::Float64=10.0,        
    K::Int=3, 
    ρ::Float64=0.2,    
    beta_master::Union{Nothing,AbstractMatrix}=nothing
)

    # random number generator
    rng = MersenneTwister(seed)

    # position and cost
    pos_f = rand(rng, F, 2) .* box
    pos_d = rand(rng, D, 2) .* box
    costs = euclidean_costs(pos_f, pos_d)

    # basic mean μ
    μ = 10.0 .* rand(rng, D)                     # mean demands Uniform(0, 10)

    # scenarios
    # X = randn(rng, N, K)                         # N×K, normal distribution
    X = 2 .* rand(rng, N, K) .- 1                # Uniform(-1, 1)
    # X = rand(rng, [-1.0, 0.0, 1.0], N, K)        # categorical features in {-1,0,1}
    ϵ_noise = (2ρ) .* rand(rng, N, D) .- ρ       # Uniform(-ρ, ρ)

    @assert beta_master !== nothing "beta_master must be provided and fixed globally."
    @assert size(beta_master, 1) ≥ D "beta_master must have at least D rows."
    @assert size(beta_master, 2) == K "beta_master has wrong number of columns K."
    β = @view beta_master[1:D, :]                # D×K

    # ξ_{n,d} = max{0, μ_d * (1 + β_d' X_n + ϵ)}
    Ξ = zeros(Float64, N, D)
    @inbounds for n in 1:N, d in 1:D
        adj = dot(@view(β[d, :]), @view(X[n, :]))
        Ξ[n, d] = max(0.0, μ[d] * (1.0 + adj + ϵ_noise[n, d]))
    end

    # 产能缩放到最大总需求的 150%
    raw = 0.5 .+ rand(rng, F)
    total_per_sample = vec(sum(Ξ, dims=2))
    max_total = maximum(total_per_sample)
    scale = 1.5 * max_total / sum(raw)
    m = raw .* scale

    return TranspoInstance(F, D, N, pos_f, pos_d, costs, μ, Ξ, m, seed), X, Array(β)
end

function generate_test_instance(
    D::Int, 
    x0::Matrix;
    test_data_size::Int = 1000,
    seed::Int=1,     
    K::Int=3, 
    ρ::Float64=0.2,   
    beta_master::Union{Nothing,AbstractMatrix}=nothing
)::Matrix

    # random number generator
    rng = MersenneTwister(seed)

    # basic mean μ
    μ = 10.0 .* rand(rng, D)                     # mean demands Uniform(0, 10)
    ϵ_noise = (2ρ) .* rand(rng, test_data_size, D) .- ρ       # Uniform(-ρ, ρ)

    @assert beta_master !== nothing "beta_master must be provided and fixed globally."
    @assert size(beta_master, 1) ≥ D "beta_master must have at least D rows."
    @assert size(beta_master, 2) == K "beta_master has wrong number of columns K."
    β = @view beta_master[1:D, :]                # D×K

    # ξ_{n,d} = max{0, μ_d * (1 + β_d' X_n + ϵ)}
    Ξ = zeros(Float64, test_data_size, D)
    @inbounds for d in 1:D
        adj = dot(@view(β[d, :]), x0)
        for n in 1:test_data_size
            Ξ[n, d] = max(0.0, μ[d] * (1.0 + adj + ϵ_noise[n, d]))
        end
    end

    return Ξ
end

# --------------------
# Save instance in JLD2 format
# --------------------
function save_instance_jld2(inst::TranspoInstance; root::AbstractString = pwd(),
                            X::AbstractMatrix=nothing, beta::AbstractMatrix=nothing)
    isdir(root) || mkpath(root)
    outpath = joinpath(root, "F$(inst.F)_D$(inst.D)_N$(inst.N).jld2")

    JLD2.jldopen(outpath, "w") do file
        file["inst"] = inst                 # keep full struct with type info
        if X !== nothing
            file["X"] = X                   # contextual features (N×K)
        end
        if beta !== nothing
            file["beta"] = beta             # sensitivities (D×K)
        end
        file["type"] = string(typeof(inst)) # explicit type name for validation
        file["timestamp"] = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    end

    return outpath
end

# --------------------
# Main batch generator
# --------------------
function generate_all(; 
    Ds=5:5:50, 
    Ns=(50,100,150), 
    F=5, 
    base_seed=20251113,
    K::Int=3,
    ρ::Float64=0.2,
    β_scale::Float64=0.05
)::Nothing
    base_path = pwd();
    data_dir = joinpath(base_path, "data")
    isdir(data_dir) || mkpath(data_dir)

    Dmax = maximum(Ds)
    seed_beta = 12345
    beta_master = make_beta_master(Dmax, K; seed_beta=seed_beta, β_scale=β_scale)

    idx = 0
    for D in Ds, N in Ns
        idx += 1
        seed = base_seed + idx
        inst, X, beta_used = generate_instance(
            F, D, N;
            seed=20251114, K=K, ρ=ρ, # 20251114
            beta_master=beta_master
        )
        _ = save_instance_jld2(inst; root=data_dir, X=X, beta=beta_used)
    end
    return nothing
end

# Example usage (uncomment to run):
# if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
# end
