function warmup_PG()


    logp, = setup_loglikel_PG2130099()

    let

        elbosphere = elbofy_sphere(logp, 6, 30)

        maximise_elbo(elbosphere, iterations = 3, Method = NelderMead())

        elbomix = elbofy_mixture(ELBOfy.ElboSphere, logp, 6, 30, K = 3)

        maximise_elbo(elbomix, iterations = 3, Method = NelderMead())


    end

    let

        elbodiag = elbofy_diag(logp, 6, 30)

        maximise_elbo(elbodiag, iterations = 3, Method = NelderMead())

        elbomix = elbofy_mixture(ELBOfy.ElboDiag, logp, 6, 30, K = 3)

        maximise_elbo(elbomix, iterations = 3, Method = NelderMead())

    end

    let

        elbofull = elbofy_full(logp, 6, 30)

        maximise_elbo(elbofull, iterations = 3, Method = NelderMead())

        elbomix = elbofy_mixture(ELBOfy.ElboFull, logp, 6, 30, K = 3)

        maximise_elbo(elbomix, iterations = 3, Method = NelderMead())

    end

    let
        
        elbomviext = elbofy_mvi_ext(logp, 1.0*Matrix(I,6,6), 30)

        resmviext = maximise_elbo(elbomviext, iterations = 3, Method = NelderMead())

        elbomviext, resmviext = updatecovariance(elbomviext, resmviext)

        maximise_elbo(elbomviext, getsolution(resmviext), iterations = 3, Method = NelderMead())

        elbomix = elbofy_mixture(ELBOfy.ElboMviEXT, logp, 6, 30, K = 3)

        maximise_elbo(elbomix, iterations = 3, Method = NelderMead())
        
    end


end