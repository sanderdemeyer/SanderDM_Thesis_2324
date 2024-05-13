import MKL
using DrWatson
@quickactivate :FermionicTN

## SETUP
# -------
import ThreadPinning
import LinearAlgebra
using DelimitedFiles
ThreadPinning.mkl_set_dynamic(0)
LinearAlgebra.BLAS.set_num_threads(1)
try
    ThreadPinning.pinthreads(:affinitymask)
catch
    @warn "Pinning failed"
end
ThreadPinning.threadinfo(; blas=true, hints=true)

## PARAMETERS
# -----------

t = 1.0
Δ = 1.0
μs = [-1.0 1.0; -2.0 2.0; -3.0 3.0]
L = 100

num = 3
svalue = 8.0

tol = 1e-12
krylovdim = 30
maxiter = 1_000

verbose = true
force = false

delim = ','

# derived parameters
lattice = FiniteChain(L)
result_dir = datadir("exp_pro", "kitaev")
~isdir(result_dir) && mkpath(result_dir)

## FUNCTIONS
# -----------

number_operator = c_number();
creation_operator = c_plus();
annihilation_operator = c_min();

function LinearAlgebra.dot(ψ::FiniteMPS, (i, O)::Pair{Int,<:AbstractTensorMap{S,1,2}}, ϕ::MPSKit.QP) where {S}
    @assert ϕ.left_gs == ψ == ϕ.right_gs "not implemented"
    @assert length(ψ) == (L = length(ϕ))
    @assert 0 < i <= L

    # on-site contribution
    E = @planar -ϕ[i][7 3; 1 2] * τ[1 2; 5 4] * O[6; 3 4] * conj(ψ.AC[i][7 6; 5])
    i == L && return E

    # rhs contribution
    @planar init[-1; -2 -3] := ϕ[L][-1 1; -2 2] * conj(ψ.AC[L][-3 1; 2])
    GBR = foldr(i+1:L-1; init) do j, gbr
        @planar tmp[-1; -2 -3] := ϕ[j][-1 1; -2 2] * conj(ψ.AC[j][-3 1; 2])
        return MPSKit.transfer_right(gbr, ϕ.left_gs.AL[j], ψ.AL[j]) + tmp
    end
    E += @planar ϕ.left_gs.AL[i][6 5; 3] * τ[3 4; 2 1] * GBR[1; 2 8] * O[7; 5 4] * conj(ψ.AL[i][6 7; 8])
    return E
end

## GENERATE DATA
# --------------

for μ in μs
    @info "μ = $μ"
    
    model = KitaevModel(t, μ, Δ, lattice)
    ψ, envs, delta = groundstate(model; tol, svalue, maxiter, verbose, force);
    
    χ_max = maximum(map(i -> dim(left_virtualspace(ψ, i-1)), 1:(L+1)))
    @info "χ_max = $χ_max"
    
    E₀ = sum(expectation_value(ψ, hamiltonian(model), envs))
    @assert abs(imag(E₀)) < 1e-12 "E₀ is not real $E₀"
    @info "E₀ = $(real(E₀))"
    
    Es, ϕs = FermionicTN.excitations(model; krylovdim, num, sector=fℤ₂(1), svalue, tol, maxiter, force);
    resize!(Es, num);
    resize!(ϕs, num);
    
    @assert all(abs.(imag.(Es)) .< 1e-12) "Es are not real $Es"
    @info "Es = $(real.(Es))"
    
    # energy densities
    e₀ = real(expectation_value(ψ, hamiltonian(model)));
    e = map(ϕ -> real(expectation_value(ϕ, hamiltonian(model))), ϕs);
    fn_energy = savename("energy", (; μ, t, Δ, L, svalue), "csv")
    open(joinpath(result_dir, fn_energy), "w") do io
        writedlm(io, zip(e₀, e...), delim)
    end
    fn_energy_diff = savename("energy_diff", (; μ, t, Δ, L, svalue), "csv")
    open(joinpath(result_dir, fn_energy_diff), "w") do io
        writedlm(io, zip(Ref(e₀) .- e...), delim)
    end
    
    # particle densities
    n₀ = real(expectation_value(ψ, number_operator))
    n = map(ϕ -> real(expectation_value(ϕ, number_operator)), ϕs);
    
    fn_particle = savename("particle", (; μ, t, Δ, L, svalue), "csv")
    open(joinpath(result_dir, fn_particle), "w") do io
        writedlm(io, zip(n₀, n...), delim)
    end
    
    fn_particle_diff = savename("particle_diff", (; μ, t, Δ, L, svalue), "csv")
    open(joinpath(result_dir, fn_particle_diff), "w") do io
        writedlm(io, zip(Ref(n₀) .- n...), delim)
    end
    
    # expansion coefficients
    c_i = map(ϕs) do ϕ
        map(1:L) do i
            dot(ϕ.left_gs, i => creation_operator, ϕ)
        end
    end
    d_i = map(ϕs) do ϕ
        map(1:L) do i
            dot(ϕ.left_gs, i => annihilation_operator, ϕ)
        end
    end
    expansion_coeffs = vcat.(c_i, d_i)
    
    fn_expansion = savename("expansion_coefficients", (; μ, t, Δ, L, svalue), "csv")
    open(joinpath(result_dir, fn_expansion), "w") do io
        writedlm(io, zip(expansion_coeffs...), delim)
    end
end
