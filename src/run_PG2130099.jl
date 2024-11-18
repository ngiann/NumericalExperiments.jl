function setup_loglikel_PG2130099()

    rng = MersenneTwister(101)

    tobs, yobs, σobs = GPCCData.readdataset(source="PG2130099");

    logp, pred, unpack = NumericalExperiments.gpccloglikelihood(tobs, yobs, σobs, kernel=GPCC.matern32, maxdelay=30, rng = rng);


end



function run_PG2130099(; iterations = 1)

    rng = MersenneTwister(101)

    logp, = setup_loglikel_PG2130099()

    # DIAGONAL
    let

        elbodiag = elbofy_diag(logp, 6, 100)

        resdiag = maximise_elbo(elbodiag, randn(rng, numparam(elbodiag)), iterations = iterations)

        JLD2.save("PG2130099_diag.jld2", "out", resdiag)

    end

    # FULL
    let

        elbofull = elbofy_full(logp, 6, 100)

        resfull = maximise_elbo(elbofull, randn(rng, numparam(elbofull)), iterations = iterations)

        JLD2.save("PG2130099_full.jld2", "out", resfull)
        
    end

    # MVI - EIG
    let

        elbosphere = elbofy_sphere(logp, 6, 100)

        ressphere = maximise_elbo(elbosphere, randn(rng, elbosphere), iterations = iterations)

        elbomvi = elbofy_mvi(logp, geteigenvectors(logp, ressphere.minimizer[1:6]), 100)

        resmvi = maximise_elbo(elbomvi, [ressphere.minimizer[1:6]; ones(6)*0.1], iterations = iterations)

        JLD2.save("PG2130099_mvi.jld2", "out", resmvi)

    end


    # MVI EXT
    let

        elbomviext = elbofy_mvi_ext(logp, 1.0*Matrix(I,6,6), 100)

        resmviext = maximise_elbo(elbomviext, randn(rng, elbomviext), iterations = iterations)

        for _ in 1:10

            elbomviext, resmviext = updatecovariance(elbomviext, resmviext)

            resmviext = maximise_elbo(elbomviext, getsolution(resmviext), iterations = iterations)

        end

        JLD2.save("PG2130099_mviext.jld2", "out", resmviext)

    end


     # SKEW EXT
     let

        elboskewext = elbofy_skewdiag_ext(logp, 1.0*Matrix(I,6,6), 100)

        resskewext = maximise_elbo(elboskewext, randn(rng, elboskewext), iterations = iterations)

        for _ in 1:10

            elboskewext, resskewext = updatecovariance(elboskewext, resskewext)

            resskewext = maximise_elbo(elboskewext, getsolution(resskewext), iterations = iterations)

        end

        JLD2.save("PG2130099_skewext.jld2", "out", resskewext)

    end


end