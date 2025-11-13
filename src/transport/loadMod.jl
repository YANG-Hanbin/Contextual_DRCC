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

include(joinpath("struct.jl"))
include(joinpath("utils.jl"))
include(joinpath("contextual_data_gen.jl"))
include(joinpath("modeling_DRCC.jl"))


# =============================
#  Load Data
# =============================
file_path = joinpath("data", "F5_D10_N50.jld2");
data = JLD2.load(file_path);
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
    seed=rand(1:100), 
    K=3, 
    ρ=0.2, 
    beta_master=data["beta"]
);
# test_data = JLD2.load(joinpath("data", "F5_D10_N150.jld2"))["inst"].Ξ;
