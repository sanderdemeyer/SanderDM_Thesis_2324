# change bond dimension every time step, max bond dimension is 120
if dim(left_virtualspace(Ψ,1)) < 120
    (Ψ, envs) = changebonds(Ψ, H, OptimalExpand(trscheme=truncbelow(10e-6)), envs)
end