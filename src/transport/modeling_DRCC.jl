# --------------------
# Contextual DRCCP
# --------------------
function contextual_DR_CCP_model!(
    inst::TranspoInstance, 
    ϵ::Float64, 
    ε::Float64,
    θ::Float64;
    dₓ::Dict = dₓ,
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
    @variable(model, v[1:inst.N] ≥ 0)
    @variable(model, s[1:inst.N] ≥ 0)
    @variable(model, t ≥ 0)
    @variable(model, λ)
    @variable(model, ζ[1:inst.N], Bin)

    # Objective function
    @objective(model, Min, dot(inst.costs, z))

    # constraints
    @constraint(model, capacity[f in 1:inst.F], sum(z[f,:]) ≤ inst.m[f])

    # DR-CC reformulation
    @constraint(model, ϵ * ε * inst.N * t - ε * inst.N * λ - sum(v) ≥ θ * inst.N )
    @constraint(model, [i in 1:inst.N], v[i] + λ ≥ s[i])
    @constraint(model, [i in 1:inst.N, d in 1:inst.D], sum(z[:, d]) - inst.Ξ[i, d] + M * ζ[i] ≥ t - s[i] - dₓ[i] )
    @constraint(model, [i in 1:inst.N], M * (1 - ζ[i]) ≥ t - s[i] - dₓ[i] )

    return model
end

# --------------------
# Nominal DRCCP
# --------------------
function DR_CCP_model!(
    inst::TranspoInstance, 
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
    @constraint(model, [i in 1:inst.N, d in 1:inst.D], sum(z[:, d]) - inst.Ξ[i, d] + M * ζ[i] ≥ t - s[i])
    @constraint(model, [i in 1:inst.N], M * (1 - ζ[i]) ≥ t - s[i])

    return model
end

# A function to run model, return result row
function run_and_record!(
    model_name::String, 
    test_data; 
    inst, 
    ϵ, 
    ε=nothing, θ
)::NamedTuple
    # choose which model to build
    model = model_name == "contextual" ?
        contextual_DR_CCP_model!(inst, ϵ, ε, θ; dₓ=dₓ) :
        DR_CCP_model!(inst, ϵ, θ)

    optimize!(model)
    st = termination_status(model)

    # record objective value if optimal
    obj = st == MOI.OPTIMAL ? objective_value(model) : missing

    # Out-of-Sample Satisfaction could be computed here if desired
    if st == MOI.OPTIMAL 
        ẑ = value.(model[:z])

        satisfaction = mean(
            [sum(vec(sum(ẑ, dims = 1)) .≥ test_data[i, :]) ≥ size(test_data)[2] for i in 1:size(test_data)[1]]
        )
    else
        satisfaction = missing
    end

    # append a row
    return (
        model_name      = model_name,
        ϵ               = ϵ,
        θ               = θ,
        ε               = ε,
        objective       = obj,
        satisfaction    = satisfaction,
        status          = string(st)
    )
end