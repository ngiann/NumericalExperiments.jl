function run_PG2130099(; iterations = 1, K = 2, S = 100, Stest = 100_000)

    
    logp, = setup_loglikel_PG2130099()


    ##########
    # Sphere #
    ##########

    p₀ = let

        rng = MersenneTwister(101)

        elbomixsphere = elbofy_mixture(ELBOfy.ElboSphere, logp, 6, S, K = K)

        aux = bbmaximise_elbo(elbomixsphere, randn(rng, numparam(elbomixsphere)), iterations = iterations, g_tol = 1e-6, Method = NelderMead(), Method = :generating_set_search)

        A = [aux() for _ in 1:10]

        fitness = [elbomixsphere(getsolution(a)) for a in A]

        bestindex = argmin(fitness)

        resmixsphere = getsolution(A[bestindex])

        JLD2.save("PG2130099_sphere.jld2", "resmixsphere", resmixsphere, "elbomixsphere", elbomixsphere)

        getsolution(resmixsphere)

    end


    ############
    # DIAGONAL #
    ############

    let
        
        elbodiag = elbofy_diag(logp, 6, S)

        p = [p₀[1:end-1]; p₀[end]*ones(6)]
        
        resdiag = maximise_elbo(elbodiag, p, iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbodiag, getsolution(resdiag), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save("PG2130099_diag.jld2", "resdiag", resdiag, "elbodiag", elbodiag, "testevidence", testevidence)

    end


    ########
    # FULL #
    ########

    let
 
        elbofull = elbofy_full(logp, 6, S)

        resfull = maximise_elbo(elbofull, [p₀[1:6]; p₀[7]*vec(1.0*Matrix(I,6,6))], iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbofull, getsolution(resfull), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save("PG2130099_full.jld2", "resfull", resfull, "elbofull", elbofull, "testevidence", testevidence)
        
    end

    ###########
    # MVI EXT #
    ###########

    let
        
        elbomviext = elbofy_mvi_ext(logp, 1.0*Matrix(I,6,6), S)

        p = [p₀[1:6]; p₀[7]*ones(6); 1]

        resmviext = maximise_elbo(elbomviext, p, iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbomviext, getsolution(resmviext), rng = MersenneTwister(101), Stest = Stest)

        JLD2.save("PG2130099_mviext_1.jld2", "resmviext", resmviext, "elbomviext", elbomviext, "testevidence", testevidence)


        for i in 2:10

            elbomviext, resmviext = updatecovariance(elbomviext, resmviext)

            resmviext = maximise_elbo(elbomviext, getsolution(resmviext), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

            testevidence = testelbo(elbomviext, getsolution(resmviext), rng = MersenneTwister(101), Stest = Stest)

            JLD2.save(@sprintf("PG2130099_mviext_%d.jld2",i), "resmviext", resmviext, "elbomviext", elbomviext, "testevidence", testevidence)

        end

    end


end
