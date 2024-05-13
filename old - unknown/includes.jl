println("test:Running the module")
using TensorKit, KrylovKit, OptimKit, FastClosures
using Base.Threads, FLoops, Transducers, FoldsThreads
using Base.Iterators
using RecipesBase
using VectorInterface
using Accessors

using LinearAlgebra: diag, Diagonal
using LinearAlgebra: LinearAlgebra
using Base: @kwdef

# bells and whistles for mpses
export InfiniteMPS, FiniteMPS, WindowMPS, MPSMultiline
export PeriodicArray, Window
export MPSTensor
export QP, LeftGaugedQP, RightGaugedQP
export leftorth,
       rightorth, leftorth!, rightorth!, poison!, uniform_leftorth, uniform_rightorth
export r_LL, l_LL, r_RR, l_RR, r_RL, r_LR, l_RL, l_LR # should be properties

# useful utility functions?
export add_util_leg, max_Ds, recalculate!
export left_virtualspace, right_virtualspace, physicalspace
export entanglementplot, transferplot

# hamiltonian things
export Cache
export SparseMPO, MPOHamiltonian, DenseMPO, MPOMultiline, LazySum
export ∂C, ∂AC, ∂AC2, environments, expectation_value, effective_excitation_hamiltonian
export leftenv, rightenv

# algos
export find_groundstate!, find_groundstate, leading_boundary
export VUMPS, DMRG, DMRG2, IDMRG1, IDMRG2, GradientGrassmann
export excitations, FiniteExcited, QuasiparticleAnsatz
export marek_gap, correlation_length, correlator
export timestep!, timestep, TDVP, TDVP2, make_time_mpo, WI, WII, TaylorCluster
export splitham, infinite_temperature, entanglement_spectrum, transfer_spectrum, variance
export changebonds!, changebonds, VUMPSSvdCut, OptimalExpand, SvdCut, UnionTrunc, RandExpand
export entropy
export propagator, NaiveInvert, Jeckelmann, DynamicalDMRG
export fidelity_susceptibility
export approximate!, approximate
export periodic_boundary_conditions
export exact_diagonalization

# transfer matrix
export TransferMatrix
export transfer_left, transfer_right

@deprecate virtualspace left_virtualspace # there is a possible ambiguity when C isn't square, necessitating specifying left or right virtualspace
@deprecate params(args...) environments(args...)
@deprecate InfiniteMPO(args...) DenseMPO(args...)

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\utility\\defaults.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\utility/periodicarray.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\utility/multiline.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\utility/utility.jl") # random utility functions
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\utility/plotting.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\utility/linearcombination.jl")

# maybe we should introduce an abstract state type
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/window.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/abstractmps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/infinitemps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/mpsmultiline.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/finitemps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/windowmps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/orthoview.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/quasiparticle_state.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states/ortho.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/densempo.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/sparsempo/sparseslice.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/sparsempo/sparsempo.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/mpohamiltonian.jl") # the mpohamiltonian objects
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/mpomultiline.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/projection.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/lazysum.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/multipliedoperator.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators/sumofoperators.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\transfermatrix/transfermatrix.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\transfermatrix/transfer.jl")

abstract type Cache end # cache "manages" environments

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/FinEnv.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/abstractinfenv.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/permpoinfenv.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/mpohaminfenv.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/qpenv.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/multipleenv.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/idmrgenv.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments/lazylincocache.jl")

abstract type Algorithm end

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/derivatives.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/expval.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/toolbox.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/grassmann.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/correlators.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/changebonds/changebonds.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/changebonds/optimalexpand.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/changebonds/vumpssvd.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/changebonds/svdcut.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/changebonds/randexpand.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/timestep/tdvp.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/timestep/timeevmpo.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/groundstate/vumps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/groundstate/idmrg.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/groundstate/dmrg.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/groundstate/gradient_grassmann.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/groundstate/find_groundstate.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/propagator/corvector.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/excitation/excitations.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/excitation/quasiparticleexcitation.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/excitation/dmrgexcitation.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/excitation/exci_transfer_system.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/statmech/vumps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/statmech/gradient_grassmann.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/fidelity_susceptibility.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/approximate/approximate.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/approximate/vomps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/approximate/fvomps.jl")
include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/approximate/idmrg.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/ED.jl")

include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms/unionalg.jl")
