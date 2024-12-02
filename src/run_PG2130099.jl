function setup_loglikel_PG2130099()

    rng = MersenneTwister(101)

    tobs, yobs, σobs = GPCCData.readdataset(source="PG2130099");

    logp, pred, unpack = NumericalExperiments.gpccloglikelihood(tobs, yobs, σobs, kernel=GPCC.matern32, maxdelay=30, rng = rng);


end



function run_PG2130099(; iterations = 1)

    
    logp, = setup_loglikel_PG2130099()

     # Sphere
     p₀ = let

        rng = MersenneTwister(101)

        elbosphere = elbofy_sphere(logp, 6, 150)

        ressphere = maximise_elbo(elbosphere, randn(rng, numparam(elbosphere)), iterations = iterations,  g_tol = 1e-6, Method = NelderMead())

        JLD2.save("PG2130099_sphere.jld2", "ressphere", ressphere, "elbosphere", elbosphere)

        getsolution(ressphere)
    end

    
    # DIAGONAL
    let
        
        elbodiag = elbofy_diag(logp, 6, 150)

        p = [p₀[1:end-1]; p₀[end]*ones(6)]
        
        resdiag = maximise_elbo(elbodiag, p, iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbodiag, getsolution(resdiag), rng = MersenneTwister(101), Stest = 100_000)

        JLD2.save("PG2130099_diag.jld2", "resdiag", resdiag, "elbodiag", elbodiag, "testevidence", testevidence)

    end

    # FULL
    let
 
        elbofull = elbofy_full(logp, 6, 150)

        resfull = maximise_elbo(elbofull, [p₀[1:6]; p₀[7]*vec(1.0*Matrix(I,6,6))], iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        testevidence = testelbo(elbofull, getsolution(resfull), rng = MersenneTwister(101), Stest = 100_000)

        JLD2.save("PG2130099_full.jld2", "resfull", resfull, "elbofull", elbofull, "testevidence", testevidence)
        
    end

    # # MVI - EIG
    # let
        
    #     elbomvi = elbofy_mvi(logp, geteigenvectors(logp, ressphere.minimizer[1:6]), 100)

    #     resmvi = maximise_elbo(elbomvi, [ressphere.minimizer[1:6]; ones(6)*0.1], iterations = iterations)

    #     JLD2.save("PG2130099_mvi.jld2", "resmvi", resmvi, "elbomvi", elbomvi)

    # end


    # MVI EXT
    elbomviext, pmviext = let
        
        elbomviext = elbofy_mvi_ext(logp, 1.0*Matrix(I,6,6), 150)

        p = [p₀[1:6]; p₀[7]*ones(6); 1]

        resmviext = maximise_elbo(elbomviext, p, iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        for _ in 1:10

            elbomviext, resmviext = updatecovariance(elbomviext, resmviext)

            resmviext = maximise_elbo(elbomviext, getsolution(resmviext), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        end

        testevidence = testelbo(elbomviext, getsolution(resmviext), rng = MersenneTwister(101), Stest = 100_000)

        JLD2.save("PG2130099_mviext.jld2", "resmviext", resmviext, "elbomviext", elbomviext, "testevidence", testevidence)

        elbomviext, getsolution(resmviext)

    end


     # SKEW EXT
     let
        
        μ₀, C₀ = ELBOfy.getμC(elbomviext, pmviext)

        elboskewext = elbofy_skewdiag_ext(logp, C₀, 150)


        p = [μ₀; zeros(6); zeros(6); 1]


        resskewext = maximise_elbo(elboskewext, p, iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        for _ in 1:10

            elboskewext, resskewext = updatecovariance(elboskewext, resskewext)

            resskewext = maximise_elbo(elboskewext, getsolution(resskewext), iterations = iterations, g_tol = 1e-6, Method = NelderMead())

        end

        testevidence = testelbo(elboskewext, getsolution(resskewext), rng = MersenneTwister(101), Stest = 100_000)

        JLD2.save("PG2130099_skewext.jld2", "resskewext", resskewext, "elboskewext", elboskewext, "testevidence", testevidence)

    end


end


function warmup_PG()


    logp, = setup_loglikel_PG2130099()

    let

        elbosphere = elbofy_sphere(logp, 6, 30)

        maximise_elbo(elbosphere, iterations = 3, Method = NelderMead())

    end

    let

        elbodiag = elbofy_diag(logp, 6, 30)

        maximise_elbo(elbodiag, iterations = 3, Method = NelderMead())

    end

    let

        elbofull = elbofy_full(logp, 6, 30)

        maximise_elbo(elbofull, iterations = 3, Method = NelderMead())

    end

   
    let
        
        elbomviext = elbofy_mvi_ext(logp, 1.0*Matrix(I,6,6), 30)

        resmviext = maximise_elbo(elbomviext, iterations = 3, Method = NelderMead())

        elbomviext, resmviext = updatecovariance(elbomviext, resmviext)

        maximise_elbo(elbomviext, getsolution(resmviext), iterations = 3, Method = NelderMead())

    end

    let
        
        elboskewext = elbofy_skewdiag_ext(logp, 1.0*Matrix(I,6,6), 30)

        resskewext = maximise_elbo(elboskewext, iterations = 3, Method = NelderMead())

        elboskewext, resskewext = updatecovariance(elboskewext, resskewext)

        maximise_elbo(elboskewext, getsolution(resskewext), iterations = 3, Method = NelderMead())

    end

end