function setup_3C120_joint_loglikel()

    tobs, yobs, σobs = GPCCData.readdataset(source="3C120")

    logp, pred, unpack = gpccloglikelihood(tobs, yobs, σobs, kernel = GPCC.matern32, maxdelay = 50)

end


function run_3C120()


    logp, = setup_3C120_joint_loglikel()

    nsamples_range = [50; 100; 200; 300; 400; 500]


    best_variational_params_diag = map(nsamples_range) do nsamples

        elbodiag = elbodiag_fy(logp, 6, nsamples)

        run_approximation(elbodiag)

    end

    best_variational_params_full = map(nsamples_range) do nsamples

        elbofull = elbofull_fy(logp, 6, nsamples)

        run_approximation(elbofull)

    end

    best_variational_params_mvi = map(nsamples_range) do nsamples

        elbomvi = elbomvi_fy(logp, 6, nsamples)

        run_approximation(elbomvi)

    end

    best_variational_params_mviext = map(nsamples_range) do nsamples

        elbomviext = elbomvi_ext_fy(logp, 6, nsamples)

        run_approximation(elbomviext)

    end


    # evaluate log-evidence and test log-evidence out-of-sample


    ########
    # DIAG #
    ########

    logevidence_diag = let

        elbodiag = elbodiag_fy(logp, 6, nsamples)

        elbodiag.(best_variational_params_diag)

    end

    test_logevidence_diag = let

        elbodiag = elbodiag_fy(logp, 6, nsamples)

        testelbo(elbodiag, best_variational_params_diag; Stest = 10_000, rng = MersenneTwister(13))

    end


    ########
    # FULL #
    ########

    logevidence_full = let

        elbofull = elbofull_fy(logp, 6, nsamples)

        elbofull.(best_variational_params_full)

    end

    test_logevidence_full = let

        elbofull = elbofull_fy(logp, 6, nsamples)

        testelbo(elbofull, best_variational_params_full; Stest = 10_000, rng = MersenneTwister(13))

    end

    #######
    # MVI #
    #######

    logevidence_mvi = let

        elbomvi = elbomvi_fy(logp, 6, nsamples)

        elbomvi.(best_variational_params_mvi)

    end

    test_logevidence_mvi = let

        elbomvi = elbomvi_fy(logp, 6, nsamples)

        testelbo(elbomvi, best_variational_params_mvi; Stest = 10_000, rng = MersenneTwister(13))

    end


    ###########
    # MVI EXT #
    ###########

    logevidence_mviext = let

        elbomviext = elbomvi_ext_fy(logp, 6, nsamples)

        elbomviext.(best_variational_params_mviext)

    end

    test_logevidence_mviext = let

        elbomviext = elbomvi_ext_fy(logp, 6, nsamples)

        testelbo(elbomviext, best_variational_params_mviext; Stest = 10_000, rng = MersenneTwister(13))

    end


    return logevidence_diag, test_logevidence_diag,
           logevidence_full, test_logevidence_full,
           logevidence_mvi, test_logevidence_mvi,
           logevidence_mviext, test_logevidence_mviext

end
