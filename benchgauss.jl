# benchmark
# note that SpecialFunctions is needed only for the computation of erfc in CBLF
using BenchmarkTools, ImageFiltering, TestImages, Images, SpecialFunctions, Statistics, Plots
include("CstTimeGauss.jl");
img = testimage("mandrill");
# check the result of O1gaussnd
O1gaussnd(float64.(img), 3)
# vs the one from imfilter
imfilter(img, Kernel.gaussian(3), "reflect")
# compare the computation time for this quiet small σ
@btime O1gaussnd(float64.($img), 3);
@btime imfilter($img, Kernel.gaussian(3), "reflect");
# the computation time are realy close increase σ to see the benefit from O1gaussnd
@btime O1gaussnd(float64.($img), 10);
@btime imfilter($img, Kernel.gaussian(10), "reflect");
# time it for different σ
σ = [1; 3; 5; 10; 15; 20; 30; 50]
tcst = []
timf = []
tcon = []
for s in σ # this for loop actualy take time...
    # it could be a nice idea to limit the number of sample and seconds allocated for each benchmark
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
# this constant time gaussian filter also allow for the implementation of a constant time bilateral
@btime CBLF(float64.($img), 50, 20 / 255);
#  finaly note also that the filter also works on gray Images
gimg = testimage("cameraman");
@btime O1gaussnd(float64.($gimg), 3);
@btime CBLF(float64.($gimg), 50, 20 / 255);
# finaly constant time derivative kernel can also be used although they are not optimized yet
include("Fastdgaussian.jl");
include("Fastd2gaussian.jl");
# for color images
colorview(RGB, permutedims(reduce((a, b) -> cat(a, b; dims = 3), [Fastdgauss(Float64.(channelview(img)[i, :, :]), 1.0, "x") for i in 1:3]), (3, 1, 2)))
colorview(RGB, permutedims(reduce((a, b) -> cat(a, b; dims = 3), [Fastdgauss(Float64.(channelview(img)[i, :, :]), 1.0, "y") for i in 1:3]), (3, 1, 2)))
colorview(RGB, permutedims(reduce((a, b) -> cat(a, b; dims = 3), [Fastd2gauss(Float64.(channelview(img)[i, :, :]), 1.0, "x") for i in 1:3]), (3, 1, 2)))
colorview(RGB, permutedims(reduce((a, b) -> cat(a, b; dims = 3), [Fastd2gauss(Float64.(channelview(img)[i, :, :]), 1.0, "y") for i in 1:3]), (3, 1, 2)))
# for gray images
colorview(Gray, Fastdgauss(Float64.(channelview(gimg)), 1.0, "x"))
colorview(Gray, Fastdgauss(Float64.(channelview(gimg)), 1.0, "y"))
colorview(Gray, Fastd2gauss(Float64.(channelview(gimg)), 1.0, "x"))
colorview(Gray, Fastd2gauss(Float64.(channelview(gimg)), 1.0, "y"))