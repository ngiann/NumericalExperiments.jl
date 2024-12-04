function fitinversegamma(;μ=μ, σ=σ)

    @assert(σ > 0)

    α = (μ*μ + 2*σ*σ) / (σ*σ)

    β = (α - 1) * μ

    InverseGamma(α, β)


end