function get_initial_mps(D_start = 3)
    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)
    vspace_L = U1Space(1//2 => D_start, -1//2 => D_start, 3//2 => D_start, -3//2 => D_start)
    vspace_R = U1Space(2 => D_start, 1 => D_start, 0 => D_start, -1 => D_start, -2 => D_start)
    mps = InfiniteMPS([pspace, pspace], [vspace_L, vspace_R])
    return mps
end