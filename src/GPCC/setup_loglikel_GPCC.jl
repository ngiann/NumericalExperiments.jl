function setup_loglikel_GPCC(;source = "PG2130099")

    if ~in(source, GPCCData.listdatasets())

        @printf("source must be one of the following:\n")
        
        display(GPCCData.listdatasets())

        return nothing

    end

    rng = MersenneTwister(101)

    tobs, yobs, σobs = GPCCData.readdataset(source = source)

    logp, pred, unpack = NumericalExperiments.gpccloglikelihood(tobs, yobs, σobs, kernel=GPCC.matern32, maxdelay=30, rng = rng)

    return logp, pred, unpack
    
end
