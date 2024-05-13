using Pkg
using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

# List the names of the packages that are imported in your script
function get_imported_packages(script_path)
    packages = Set{String}()
    open(script_path, "r") do file
        for line in eachline(file)
            match_result = match(r"^\s*using\s+([^\s#]+)", line)
            if match_result !== nothing
                push!(packages, match_result.captures[1])
            end
        end
    end
    return packages
end

# Replace "your_script.jl" with the path to your Julia script
script_path = "your_script.jl"
imported_packages = get_imported_packages(script_path)

println("Imported packages in $script_path:")
println(imported_packages)
