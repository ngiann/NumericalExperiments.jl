module NumericalExperiments

    using GPCC, GPCCData, LinearAlgebra, Random, ELBOfy, ELBOfyUtilities, Distributions

    include("gpccloglikelihood.jl")
    include("roundeduniform.jl")
    export gpccloglikelihood, roundeduniform

    include("runapproximation.jl")

    include("3C120.jl")

end
