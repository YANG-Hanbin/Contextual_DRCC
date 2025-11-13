## Load packages and modules
include(joinpath(@__DIR__, "loadMod.jl"))

# =============================
# Experimental settings
# =============================

ϵ_list = [0.05, 0.1, 0.15]
θ_list = [0.2, 0.15, 0.10, 0.05]
ε_list = [0.3, 0.5, 0.7, 0.9]

results = DataFrame(
    model_name      = String[],
    ϵ               = Float64[],
    θ               = Float64[],
    ε               = Union{Float64,Nothing}[],
    objective       = Union{Float64,Missing}[],
    satisfaction    = Union{Float64,Missing}[],
    status          = String[],
)

# =============================
# Experiment loops
# =============================
for ϵ in ϵ_list
    for θ in θ_list
        # nominal (no ε parameter)
        result = run_and_record!("nominal", test_data; inst=inst, ϵ=ϵ, θ=θ)
        push!(results, result)
        for ε in ε_list
            # contextual model
            result = run_and_record!("contextual", test_data; inst=inst, ϵ=ϵ, ε=ε, θ=θ)
            push!(results, result)
        end
    end
end

# =============================
# Experiment results
# =============================
println(results)