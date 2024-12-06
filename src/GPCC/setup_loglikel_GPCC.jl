function setup_loglikel_GPCC(;source = source, maxdelay = maxdelay)

    if ~in(source, GPCCData.listdatasets())

        @printf("source must be one of the following:\n")
        
        display(GPCCData.listdatasets())

        return nothing

    end

    rng = MersenneTwister(101)

    tobs, yobs, σobs = GPCCData.readdataset(source = source)

    logp, pred, unpack = NumericalExperiments.gpccloglikelihood(tobs, yobs, σobs, kernel=GPCC.matern32, maxdelay=maxdelay, rng = rng)

    return logp, pred, unpack
    
end
