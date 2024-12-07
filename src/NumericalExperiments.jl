module NumericalExperiments

    using JLD2, Printf, GPCC, GPCCData, LinearAlgebra, Random, ELBOfy, ELBOfyUtilities, Distributions, ProgressMeter, ThreadTools, Optim

    using Distances, DelimitedFiles, StatsFuns
    # Following lines makes ProgressMeter work with tmap1

    ProgressMeter.ncalls(::typeof(tmap1), ::Function, args...) = ProgressMeter.ncalls_map(args...)


    include("GP/gpfit.jl")
    include("GP/read_concrete.jl")
    export gpfit, read_concrete

    ########
    # GPCC #
    ########
    
    include("GPCC/gpccloglikelihood.jl")
    include("GPCC/fitinversegamma.jl")
    include("GPCC/roundeduniform.jl")
    include("GPCC/run_GPCC.jl")
    include("GPCC/run_mixture_GPCC.jl")
    include("GPCC/setup_loglikel_GPCC.jl")
    include("GPCC/warmup_GPCC.jl")

    export warmup_GPCC, setup_loglikel_GPCC, run_GPCC, run_mixture_GPCC

end
