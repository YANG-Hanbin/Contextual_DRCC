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
include(joinpath(@__DIR__, "modeling_DRCC.jl"))

# =============================
#  Load Data
# =============================
file_path = joinpath(@__DIR__, "..", "..", "data", "F5_D10_N50_seed20251114.jld2")
data = JLD2.load(file_path)
inst = data["inst"];
X = data["X"];

## a newly observed contextual info
# x0 = randn(MersenneTwister(1), 1, 3);
x0 = 2 .* rand(MersenneTwister(1), 1, 3) .- 1;
# x0 = rand(MersenneTwister(1), [-1.0, 0.0, 1.0], 1, 3);

# distance from x0 to each training context
dₓ = Dict()
for i in 1:inst.N 
    dₓ[i] = norm(vec(x0) - X[i, :]);
end

# out-of-sample test data
test_data = generate_test_instance(
    inst.D, 
    x0; 
    test_data_size = 1000, 
    seed=20251114, 
    K=3, 
    ρ=0.2, 
    beta_master=data["beta"]
);