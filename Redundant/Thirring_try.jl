using MPSKitModels, TensorKit, MPSKit

g = 1.0
J1 = 1.0
J2 = 1.0

gs = get_groundstate_energy(0.1, -0.8, 20)
spectrum = transfer_spectrum(gs)

println(spectrum)

break

H_J1J2 = @mpoham sum(J1 * S_exchange(){i, j} for (i, j) in nearest_neighbours(InfiniteCylinder(4))) + sum(J2 * S_exchange(){i,j} for (i, j) in next_nearest_neighbours(InfiniteCylinder(4)))

pspace = [ℂ^2, ℂ^2]

state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^10, ℂ^10]);
state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^50, ℂ^50]);


am_tilde_0 = 0.1
Delta_g = -0.8

Hopping_term = (-1/4) * @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
Mass_term = am_tilde_0 * @mpoham sum(S_z(){i} + id(domain(S_z())){i}/2 for i in vertices(InfiniteChain(2)))
Interaction_term = Delta_g * @mpoham sum(S_zz(){i, i + 1} + S_z(){i}/2 + id(domain(S_z())){i}/4 for i in vertices(InfiniteChain(2)))


#=
Mass_term = m_0 * @mpoham sum(S_z(){i} + id(ℂ^2){i} for i in vertices(InfiniteChain(2)))
#Interaction_term = (g_tilde/(2*a)) * @mpoham sum((S_z(){i} + id(pspace){i}/2)*(S_z(){i+1} + id(pspace){i}/2) for i in vertices(InfiniteChain(1)))
#Interaction_term = (g_tilde/(2*a)) * @mpoham sum(S_zz(){i, i + 1} + S_z(){i}/2 + S_z(){i+1}/2 + id(pspace){i}/4 for i in vertices(InfiniteChain(1)))

=#



try_h = id(ℂ^2)
try_h = id(ℂ^2 ⊗ ℂ^2)
elt = Float64
pspace = Z2Space(0 => 1, 1 => 1)
X = TensorMap(zeros, elt, pspace, pspace)
blocks(X)[Z2Irrep(0)] .= one(elt) / 2
blocks(X)[Z2Irrep(1)] .= -one(elt) / 2

# Hopping_term = @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} + Delta * S_zz(){i, i + 1} for i in vertices(InfiniteChain(1)))

println("here")
hamiltonian = Hopping_term# + Mass_term + Interaction_term

(groundstate,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 5));

println(correlation_length(groundstate))
println(energy(groundstate))

#println(expectation_value(groundstate, hamiltonian))

#=

J = [1.0 -1.0]  # staggered couplings over unit cell of length 2
H_heisenberg_ = @mpoham sum(J[i] * S_exchange(SU2Irrep; spin=1){i, i + 1} for i in vertices(InfiniteChain(2)))

H_heisenberg_cylinder =
    @mpoham sum(J1 * S_exchange(; spin=1){i, j} for (i, j) in nearest_neighbours(InfiniteCylinder(3)))

J1 = 0.8
J2 = 0.2

H_J1J2 = @mpoham sum(J1 * S_exchange(){i, j} for (i, j) in nearest_neighbours(InfiniteCylinder(4))) +
    sum(J2 * S_exchange(){i,j} for (i, j) in next_nearest_neighbours(InfiniteCylinder(4)))

=#

#=
Hopping_term = (-Z_g/(4*a)) * @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in -Inf:Inf)
Mass_term = m_0 * @mpoham sum((-1)^i*(S_z(){i} + 1/2) for i in -Inf:Inf)
Interaction_term = (g_tilde/(2*a)) * @mpoham sum((S_z(){i} + 1/2)*(S_z(){i+1} + 1/2) for i in -Inf:Inf)

=#
println("done")
