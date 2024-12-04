module NumericalExperiments

    using JLD2, Printf, GPCC, GPCCData, LinearAlgebra, Random, ELBOfy, ELBOfyUtilities, Distributions, ProgressMeter, ThreadTools, Optim

    # Following lines makes ProgressMeter work with tmap1

    ProgressMeter.ncalls(::typeof(tmap1), ::Function, args...) = ProgressMeter.ncalls_map(args...)


    ########
    # GPCC #
    ########
    
    include("PG213009/gpccloglikelihood.jl")
    include("PG213009/fitinversegamma.jl")
    include("PG213009/roundeduniform.jl")
    include("PG213009/run_PG2130099.jl")
    include("PG213009/setup_loglikel_PG2130099.jl")
    include("PG213009/warmup_PG.jl")

    export warmup_PG, setup_loglikel_PG2130099, run_PG2130099

end
