module NumericalExperiments

    using JLD2, Printf, GPCC, GPCCData, LinearAlgebra, Random, ELBOfy, ELBOfyUtilities, Distributions, ProgressMeter, ThreadTools, Optim

    # Following lines makes ProgressMeter work with tmap1

    ProgressMeter.ncalls(::typeof(tmap1), ::Function, args...) = ProgressMeter.ncalls_map(args...)


    ########
    # GPCC #
    ########
    
    include("GPCC/gpccloglikelihood.jl")
    include("GPCC/fitinversegamma.jl")
    include("GPCC/roundeduniform.jl")
    include("GPCC/run_GPCC.jl")
    include("GPCC/setup_loglikel_GPCC.jl")
    include("GPCC/warmup_PG.jl")

    export warmup_PG, setup_loglikel_GPCC, run_GPCC

end
