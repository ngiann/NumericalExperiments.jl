function run_mixture_GPCC(;source = source, iterations = 1, K = 2, S = 100, Stest = 100_000)

    
    logp, = setup_loglikel_GPCC(source = source)


    ##########
    # Sphere #
    ##########

    mixsphere, p₀ = let

        rng = MersenneTwister(101)

        elbomixsphere = elbofy_mixture(ELBOfy.ElboSphere, logp, 6, S, K = K)

        aux = () -> bbmaximise_elbo(elbomixsphere, randn(rng, numparam(elbomixsphere)), iterations = iterations, Method = :generating_set_search)

        A = [aux() for _ in 1:10] # run fitting multiple times

        fitness = [elbomixsphere(getsolution(a)) for a in A] # collect lower bounds

        bestindex = argmax(fitness) # get solution with highest evidence

        resmixsphere = getsolution(A[bestindex])

        JLD2.save(@sprintf("%s_K%d_mixture_sphere.jld2", source, K), "resmixsphere", resmixsphere, "elbomixsphere", elbomixsphere)

        elbomixsphere, getsolution(resmixsphere)

    end


    ############
    # DIAGONAL #
    ############

    let
        
        elbodiag = elbofy_mixture(ELBOfy.ElboDiag, logp, 6, S, K = K)

        resdiag = maximise_elbo(elbodiag, diagonal_parameters(mixsphere, p₀), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbodiag, getsolution(resdiag), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save(@sprintf("%s_K%d_mixture_diag.jld2", source, K), "resdiag", resdiag, "elbodiag", elbodiag, "testevidence", testevidence)

    end


    ########
    # FULL #
    ########

    let
 
        elbofull = elbofy_mixture(ELBOfy.ElboFull, logp, 6, S, K = K) 

        resfull = maximise_elbo(elbofull, full_parameters(mixsphere, p₀), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbofull, getsolution(resfull), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save(@sprintf("%s_K%d_mixture_full.jld2", source, K), "resfull", resfull, "elbofull", elbofull, "testevidence", testevidence)
        
    end

    ###########
    # MVI EXT #
    ###########

    let
        
        elbomviext = elbofy_mixture(ELBOfy.ElboMVIExt, logp, 6, S, K = K) 

        resmviext = maximise_elbo(elbomviext, mvi_parameters(mixsphere, p₀), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbomviext, getsolution(resmviext), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save(@sprintf("%s_K%d_mixture_mviext_1.jld2", source, K), "resmviext", resmviext, "elbomviext", elbomviext, "testevidence", testevidence)


        for i in 2:10

            elbomviext, resmviext = updatecovariance(elbomviext, resmviext)

            resmviext = maximise_elbo(elbomviext, getsolution(resmviext), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

            testevidence = testelbo(elbomviext, getsolution(resmviext), rng = MersenneTwister(101), Stest = Stest)

            JLD2.save(@sprintf("%s_K%d_mviext_%d.jld2", source, K, i), "resmviext", resmviext, "elbomviext", elbomviext, "testevidence", testevidence)

        end

    end


end
