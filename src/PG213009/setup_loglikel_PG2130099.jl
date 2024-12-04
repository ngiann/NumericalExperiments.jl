function setup_loglikel_PG2130099()

    rng = MersenneTwister(101)

    tobs, yobs, σobs = GPCCData.readdataset(source="PG2130099")

    logp, pred, unpack = NumericalExperiments.gpccloglikelihood(tobs, yobs, σobs, kernel=GPCC.matern32, maxdelay=30, rng = rng)

    return logp, pred, unpack
    
end
