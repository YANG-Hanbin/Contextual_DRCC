# =============================
#  Load Modules and Packages
# =============================
using Pkg
Pkg.activate(".")

using JuMP, Gurobi;
using LinearAlgebra;
using Statistics, StatsBase, Random, Dates, Distributions;
using Distributed;
using DataFrames, Printf, PrettyTables;
using JLD2, FilePathsBase;
const GRB_ENV = Gurobi.Env();

# cd("/Users/aaron/Contextual_DRCC/src/transport");
include(joinpath(@__DIR__, "struct.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "contextual_data_gen.jl"))

data = JLD2.load(joinpath(@__DIR__, "..", "..", "data", "F5_D10_N100.jld2"));
inst = data["inst"];
# --------------------
# Nominal DRCCP
# --------------------
x0 = 2 .* rand(MersenneTwister(1), 1, 3) .- 1;

# out-of-sample test data
historical_data = generate_test_instance(
    inst.D, 
    x0; 
    test_data_size = inst.N, 
    seed=inst.seed, 
    K=3, 
    ρ=0.2, 
    beta_master=data["beta"]
);

test_data = generate_test_instance(
    inst.D, 
    x0; 
    test_data_size = 1000, 
    seed=inst.seed, 
    K=3, 
    ρ=0.2, 
    beta_master=data["beta"]
);

function nominal_DR_CCP_model!(
    inst::TranspoInstance, 
    historical_data::Array{Float64,2},
    ϵ::Float64, 
    θ::Float64,
    M::Float64=1e3
)::Model
    model = Model(
        optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
    ); 
    MOI.set(model, MOI.Silent(), true);
    set_optimizer_attribute(model, "MIPGap", 1e-4);
    set_optimizer_attribute(model, "TimeLimit", 600);
    set_optimizer_attribute(model, "IntFeasTol", 1e-9)

    # variables
    @variable(model, z[f in 1:inst.F, d in 1:inst.D] ≥ 0)
    @variable(model, s[1:inst.N] ≥ 0)
    @variable(model, t ≥ 0)
    @variable(model, ζ[1:inst.N], Bin)

    # Objective function
    @objective(model, Min, dot(inst.costs, z))

    # constraints
    @constraint(model, capacity[f in 1:inst.F], sum(z[f,:]) ≤ inst.m[f])

    # DR-CC reformulation
    @constraint(model, ϵ * inst.N * t - sum(s) ≥ θ * inst.N )
    @constraint(model, [i in 1:inst.N, d in 1:inst.D], sum(z[:, d]) - historical_data[i, d] + M * ζ[i] ≥ t - s[i])
    @constraint(model, [i in 1:inst.N], M * (1 - ζ[i]) ≥ t - s[i])

    return model
end

# A function to run model, return result row
function run_and_record!(
    test_data,
    inst, 
    historical_data,
    ϵ,
    θ
)::NamedTuple
    # choose which model to build
    model = nominal_DR_CCP_model!(inst, historical_data, ϵ, θ)

    optimize!(model)
    st = termination_status(model)

    # record objective value if optimal
    obj = st == MOI.OPTIMAL ? objective_value(model) : missing

    # Out-of-Sample Satisfaction could be computed here if desired
    if st == MOI.OPTIMAL 
        ẑ = value.(model[:z])

        satisfaction = mean(
            [sum(vec(sum(ẑ, dims = 1)) .≥ test_data[i, :]) ≥ size(test_data)[2] for i in 1:size(test_data)[1]]
        )
    else
        satisfaction = missing
    end

    # append a row
    return (
        ϵ               = ϵ,
        θ               = θ,
        objective       = obj,
        satisfaction    = satisfaction,
        status          = string(st)
    )
end


results = DataFrame(
    ϵ               = Float64[],
    θ               = Float64[],
    objective       = Union{Float64,Missing}[],
    satisfaction    = Union{Float64,Missing}[],
    status          = String[],
)

θ = .1
ϵ = .1
M = 1e3

ϵ_list = [0.05, 0.1, 0.15]
θ_list = [0.10, 0.08, 0.06, 0.04, 0.02, 0.01]

for ϵ in ϵ_list
    for θ in θ_list
        row = run_and_record!(
            test_data,
            inst,
            historical_data,
            ϵ,
            θ
        )
        push!(results, row)
    end
end

println(results)
