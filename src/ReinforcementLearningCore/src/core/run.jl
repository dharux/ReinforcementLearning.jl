export Experiment

"""
    Experiment(policy::AbstractPolicy, env::AbstractEnv, stop_condition::AbstractStopCondition, hook::AbstractHook)

A struct to hold the information of an experiment. It is used to run an experiment with the given policy, environment, stop condition and hook.
"""
struct Experiment
    policy::AbstractPolicy
    env::AbstractEnv
    stop_condition::AbstractStopCondition
    hook::AbstractHook
end

Base.show(io::IO, m::MIME"text/plain", t::Experiment) = show(io, m, convert(AnnotatedStructTree, t))

function Base.run(ex::Experiment)
    run(ex.policy, ex.env, ex.stop_condition, ex.hook)
    return ex
end

function Base.run(
    policy::AbstractPolicy,
    env::AbstractEnv,
    stop_condition::AbstractStopCondition=StopAfterNEpisodes(1),
    hook::AbstractHook=EmptyHook(),
    reset_condition::AbstractResetCondition=ResetIfEnvTerminated()
)
    policy, env = check(policy, env)
    _run(policy, env, stop_condition, hook, reset_condition)
end

"Inject some customized checkings here by overwriting this function"
check(policy, env) = policy, env

function _run(policy::AbstractPolicy,
        env::AbstractEnv,
        stop_condition::AbstractStopCondition,
        hook::AbstractHook,
        reset_condition::AbstractResetCondition)
    push!(hook, PreExperimentStage(), policy, env)
    push!(policy, PreExperimentStage(), env)
    is_stop = false
    while !is_stop
        # NOTE: @timeit_debug statements are used for debug logging
        reset!(env)
        push!(policy, PreEpisodeStage(), env)
        optimise!(policy, PreEpisodeStage())
        push!(hook, PreEpisodeStage(), policy, env)


        while !check!(reset_condition, policy, env) # one episode
            push!(policy, PreActStage(), env)
            optimise!(policy, PreActStage())
            push!(hook, PreActStage(), policy, env)

            action = RLBase.plan!(policy, env)
            act!(env, action)

            push!(policy, PostActStage(), env, action)
            optimise!(policy, PostActStage())
            push!(hook, PostActStage(), policy, env)

            if check!(stop_condition, policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        push!(policy, PostEpisodeStage(), env)
        optimise!(policy, PostEpisodeStage())
        push!(hook, PostEpisodeStage(), policy, env)

    end
    push!(policy, PostExperimentStage(), env)
    push!(hook, PostExperimentStage(), policy, env)
    hook
end
