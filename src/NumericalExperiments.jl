module NumericalExperiments

    using GPCC, GPCCData, LinearAlgebra, Random, ELBOfy, ELBOfyUtilities

    include("gpccloglikelihood.jl")

    export gpccloglikelihood

    include("runapproximation.jl")

    include("3C120.jl")

    export run_3C120

end
