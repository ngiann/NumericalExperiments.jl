function run_GPCC(; source = source, iterations = 1, S = 100, Stest = Stest, maxdelay = maxdelay)

    
    logp, = setup_loglikel_GPCC(source = source)


    ##########
    # Sphere #
    ##########

    p₀ = let

        rng = MersenneTwister(101)

        elbosphere = elbofy_sphere(logp, 6, S)

        ressphere = maximise_elbo(elbosphere, randn(rng, numparam(elbosphere)), iterations = iterations,  g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbosphere, getsolution(ressphere), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save(@sprintf("%s_sphere.jld2", source), "ressphere", ressphere, "elbosphere", elbosphere, "testevidence", testevidence)

        getsolution(ressphere)

    end


    ############
    # DIAGONAL #
    ############

    let
        
        elbodiag = elbofy_diag(logp, 6, S)

        p = [p₀[1:end-1]; p₀[end]*ones(6)]
        
        resdiag = maximise_elbo(elbodiag, p, iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbodiag, getsolution(resdiag), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save(@sprintf("%s_diag.jld2", source), "resdiag", resdiag, "elbodiag", elbodiag, "testevidence", testevidence)

    end


    ########
    # FULL #
    ########

    let
 
        elbofull = elbofy_full(logp, 6, S)

        resfull = maximise_elbo(elbofull, [p₀[1:6]; p₀[7]*vec(1.0*Matrix(I,6,6))], iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbofull, getsolution(resfull), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save(@sprintf("%s_full.jld2", source), "resfull", resfull, "elbofull", elbofull, "testevidence", testevidence)
        
    end
    

    ###########
    # MVI EXT #
    ###########

    let
        
        elbomviext = elbofy_mvi_ext(logp, 1.0*Matrix(I,6,6), S)

        p = [p₀[1:6]; p₀[7]*ones(6); 1]

        resmviext = maximise_elbo(elbomviext, p, iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbomviext, getsolution(resmviext), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save(@sprintf("%s_mviext_1.jld2", source), "resmviext", resmviext, "elbomviext", elbomviext, "testevidence", testevidence)


        for i in 2:10

            elbomviext, resmviext = updatecovariance(elbomviext, resmviext)

            resmviext = maximise_elbo(elbomviext, getsolution(resmviext), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

            testevidence = testelbo(elbomviext, getsolution(resmviext), rng = MersenneTwister(101), Stest = Stest)

            JLD2.save(@sprintf("%s_mviext_%d.jld2",source, i), "resmviext", resmviext, "elbomviext", elbomviext, "testevidence", testevidence)

        end

    end


end
