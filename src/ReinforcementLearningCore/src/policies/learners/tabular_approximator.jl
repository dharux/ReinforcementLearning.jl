export TabularApproximator, TabularVApproximator, TabularQApproximator

using Flux

struct TabularApproximator{A,O} <: AbstractLearner where {A<:AbstractArray, O<:Union{Nothing,Flux.Optimise.AbstractOptimiser}}
    model::A
    optimiser::O
end

const TabularQApproximator = TabularApproximator{A,O} where {A<:AbstractMatrix, O<:Union{Nothing,Flux.Optimise.AbstractOptimiser}}
const TabularVApproximator = TabularApproximator{A,O} where {A<:AbstractVector, O<:Union{Nothing,Flux.Optimise.AbstractOptimiser}}

"""
    TabularApproximator(table<:AbstractArray [, opt<:AbstractOptimiser])

For `table` of 1-d, it will serve as a state value approximator.
For `table` of 2-d, it will serve as a state-action value approximator.
Also bundles a Flux optimiser to control how the updates are performed

!!! warning
    For `table` of 2-d, the first dimension is action and the second dimension is state.
"""
function TabularApproximator(table::A, opt::O=nothing) where {A<:AbstractArray, O<:Union{Nothing,Flux.Optimise.AbstractOptimiser}}
    n = ndims(table)
    n <= 2 || throw(ArgumentError("the dimension of table must be <= 2"))
    TabularApproximator{A,O}(table, opt)
end

TabularVApproximator(; n_state, opt = nothing, init = 0.0) =
    TabularApproximator(fill(init, n_state), opt)

"""
    TabularQApproximator(; n_state, n_action, opt = nothing, init = 0.0)

Create a `TabularQApproximator` with `n_state` states and `n_action` actions.
and Flux optimiser opt
"""
TabularQApproximator(; n_state, n_action, opt = nothing, init = 0.0) =
    TabularApproximator(fill(init, n_action, n_state), opt)

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
forward(L::TabularVApproximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(L, x))
forward(L::TabularQApproximator, env::E) where {E <: AbstractEnv} = env |> state |> (x -> forward(L, x))

RLCore.forward(
    app::TabularVApproximator{R},
    s::I,
) where {R<:AbstractVector,I} = @views app.model[s]

RLCore.forward(
    app::TabularQApproximator{R},
    s::I,
) where {R<:AbstractArray,I} = @views app.model[:, s]

RLCore.forward(
    app::TabularQApproximator{R},
    s::I1,
    a::I2,
) where {R<:AbstractArray,I1,I2} = @views app.model[a, s]

