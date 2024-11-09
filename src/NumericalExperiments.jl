module NumericalExperiments

    using GPCC, GPCCData, LinearAlgebra, Random, ELBOfy, ELBOfyUtilities, Distributions, ProgressMeter, ThreadTools

    # Following lines makes ProgressMeter work with tmap1

    ProgressMeter.ncalls(::typeof(tmap1), ::Function, args...) = ProgressMeter.ncalls_map(args...)

    include("gpccloglikelihood.jl")
    include("fitinversegamma.jl")
    include("roundeduniform.jl")
    export gpccloglikelihood, roundeduniform

    # include("runapproximation.jl")

    include("3C120.jl")

end
