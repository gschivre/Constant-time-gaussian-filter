# benchmark
# note that SpecialFunctions is needed only for the computation of erfc in CBLF
using BenchmarkTools, ImageFiltering, TestImages, Images, SpecialFunctions, Statistics, Plots
include("CstTimeGauss.jl")
img = testimage("mandrill");
# check the result of O1gaussnd
O1gaussnd(float64.(img), 3)
# vs the one from imfilter
imfilter(img, Kernel.gaussian(3), "reflect")
# time it for different σ
σ = [1; 3; 5; 10; 15; 20; 30; 50]
tcst = []
timf = []
tcon = []
for s in σ # this for loop actualy take time...
    b = @benchmark O1gaussnd(float64.($img), $s);
    push!(tcst, b.times)
    b = @benchmark imfilter($img, Kernel.gaussian($s), "reflect");
    push!(timf, b.times)
    b = @benchmark convnd(float64.($img), $s, 6 * $s);
    push!(tcon, b.times)
end
# display mean time and standard deviation for each σ
m = hcat([mean(tcst[i]) for i in 1:length(σ)], [mean(timf[i]) for i in 1:length(σ)], [mean(tcon[i]) for i in 1:length(σ)]) ./ 1e6
s = hcat([std(tcst[i]) for i in 1:length(σ)], [std(timf[i]) for i in 1:length(σ)], [std(tcon[i]) for i in 1:length(σ)]) ./ 1e6
scatter(σ, m[:, 1]; yerrors = s[:, 1], xlabel = "σ", ylabel = "time (ms)", label = "O1gaussnd", legend = :topleft)
scatter!(σ, m[:, 2]; yerrors = s[:, 2], label = "imfilter")
scatter!(σ, m[:, 3]; yerrors = s[:, 3], label = "convnd")
# the error seem hight at first glance but is much reduce when we force imfilter to use a kernel radius of at least 6*σ
maximum(abs.(imfilter(img, Kernel.gaussian(3), "reflect") .- O1gaussnd(float64.(img), 3)))
maximum(abs.(imfilter(img, KernelFactors.gaussian((3, 3), (19, 19)), "reflect") .- O1gaussnd(float64.(img), 3)))