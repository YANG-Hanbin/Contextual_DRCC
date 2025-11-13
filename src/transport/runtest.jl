## Load packages and modules
include(joinpath(@__DIR__, "loadMod.jl"))

# =============================
# Experimental settings
# =============================

ϵ_list = [0.05, 0.1, 0.15]
θ_list = [0.1, 0.075, 0.05]
ε_list = [0.3, 0.5, 0.7]

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
ϵ_list = [0.1]
θ_list = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.3, 0.35, 0.4, 0.45, 0.5]
ε_list = [0.5]

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