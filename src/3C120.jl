function setup_3C120_joint_loglikel()

    tobs, yobs, σobs = GPCCData.readdataset(source="3C120")

    logp, pred, unpack = gpccloglikelihood(tobs, yobs, σobs, kernel = GPCC.matern32, maxdelay = 50)

end


function run_3C120(; iterations = 1000, cmaesiterations = 500, repeats = 10)


    logp, = setup_3C120_joint_loglikel()

    nsamples_range = [50; 100]#; 200; 300; 400; 500]


    results_diag = map(nsamples_range) do nsamples

        elbodiag = elbofy_diag(logp, 6, nsamples)

        display(elbodiag)

        varparams = run_approximation(elbodiag; iterations = iterations, cmaesiterations = cmaesiterations, repeats = repeats)

        varparams, elbodiag(varparams), testelbo(elbodiag, varparams, Stest = 10_000)
        
    end

    results_full = map(nsamples_range) do nsamples

        elbofull = elbofy_full(logp, 6, nsamples)

        display(elbofull)

        varparams = run_approximation(elbofull; iterations = iterations, cmaesiterations = cmaesiterations, repeats = repeats)

        varparams, elbofull(varparams), testelbo(elbofull, varparams, Stest = 10_000)

    end

    results_mvi = map(nsamples_range) do nsamples

        elbomvi = elbofy_mvi(logp, 1.0*Matrix(I,6,6), nsamples)

        display(elbomvi)

        varparams = run_approximation(elbomvi; iterations = iterations, cmaesiterations = cmaesiterations, repeats = repeats)

        varparams, elbomvi(varparams), testelbo(elbomvi, varparams, Stest = 10_000)

    end

    results_mviext = map(nsamples_range) do nsamples

        elbomviext = elbofy_mvi_ext(logp, 1.0*Matrix(I,6,6), nsamples)

        display(elbomviext)

        varparams = run_approximation(elbomviext; iterations = iterations, cmaesiterations = cmaesiterations, repeats = repeats)

        varparams, elbomviext(varparams), testelbo(elbomviext, varparams, Stest = 10_000)

    end

    

    return results_diag, results_full, results_mvi, results_mviext

end
