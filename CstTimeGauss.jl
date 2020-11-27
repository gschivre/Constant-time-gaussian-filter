# function to perform compressive bilateral filter
function CBLF(img,σs,σr,guide=nothing,τ=0.005)
    # normalize tone of img and guide to [0,1]
    img = (img.-minimum(img[:]))./(maximum(img[:])-minimum(img[:]))
    if guide === nothing
        guide = deepcopy(img)
    else
        guide = (guide.-minimum(guide[:]))./(maximum(guide[:])-minimum(guide[:]))
    end
    # optimize for K
    ξ = erfcinv(τ*τ)
    K = Int64(ceil(ξ*ξ/(2*π)+ξ/(2*π*σr)-0.5))
    function Dkernelerror(T::Float64)
        ϕ = (π*σr*(2*K+1))/T
        ψ = (T-1.0)/σr
        # derivative of E(K,T) = erfc(ϕ)+erfc(ψ)
        return ((2*sqrt(π)*σr*(2*K+1))/(T*T))*exp(-ϕ*ϕ)-(2/(sqrt(π)*σr))*exp(-ψ*ψ)
    end
    # Binary search
    # extend search domain by 0.03
    low = σr*ξ+1.0-0.03
    up = (π*σr*(2*K+1))/ξ+0.03
    # loop 10times to find T
    for i in 1:10
        mid = (low+up)/2
        if Dkernelerror(mid) < 0
            low = mid
        else
            up = mid
        end
    end
    T = (low+up)/2
    b0 = ones(size(img))
    b = gaussiannd(img,σs)
    @inbounds for k in 1:K
        ω = 2*π*k/T
        ak = 2*exp(-0.5*ω*ω*σr*σr)
        c = cos.(ω.*guide)
        s = sin.(ω.*guide)
        Ψc = gaussiannd(c,σs)
        Ψs = gaussiannd(s,σs)
        b0 += ak.*(c.*Ψc.+s.*Ψs)
        Ψc = gaussiannd(c.*img,σs)
        Ψs = gaussiannd(s.*img,σs)
        b += ak.*(c.*Ψc.+s.*Ψs)
    end
    return b./b0
end

# function to perform inplace constant time gaussian convolution on ND image
function O1gaussnd!(img,σ,K=2)
    N = ndims(img)
    (N < 1 || N > 3 ? throw(ArgumentError("image should be of dimension 1,2 or 3")) : nothing)
    (N != length(σ) && length(σ) != 1 ? throw(ArgumentError("length of σ not equal to ndims of img and length of σ != 1")) : nothing)
    if N == 1
        R = findR(σ,K)
        (length(img) < R+2 ? throw(ArgumentError("Kernel radius is too large")) : nothing)
        O1gauss1d!(img,σ,K,R,precomputecoeff(σ,R,K))
    elseif N == 2
        sz = size(img)
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = [findR(σ[i],K) for i = 1:N]
        table = [precomputecoeff(σ[i],R[i],K) for i = 1:N]
        # filter x
        (sz[2] < R[1]+2 ? throw(ArgumentError("Kernel radius is too large for x direction")) : nothing)
        @inbounds Threads.@threads for y in 1:sz[1]
            O1gauss1d!(view(img,y,:),σ[1],K,R[1],table[1])
        end
        (sz[1] < R[2]+2 ? throw(ArgumentError("Kernel radius is too large for y direction")) : nothing)
        # filter y
        @inbounds Threads.@threads for x in 1:sz[2]
            O1gauss1d!(view(img,:,x),σ[2],K,R[2],table[2])
        end
    elseif N == 3
        sz = size(img)
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = [findR(σ[i],K) for i = 1:N]
        table = [precomputecoeff(σ[i],R[i],K) for i = 1:N]
        # filter x
        (sz[2] < R[1]+2 ? throw(ArgumentError("Kernel radius is too large for x direction")) : nothing)
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[3]))
            (y,z) = Tuple(idx)
            O1gauss1d!(view(img,y,:,z),σ[1],K,R[1],table[1])
        end
        # filter y
        (sz[1] < R[2]+2 ? throw(ArgumentError("Kernel radius is too large for y direction")) : nothing)
        @inbounds Threads.@threads for idx in CartesianIndices((sz[2],sz[3]))
            (x,z) = Tuple(idx)
            O1gauss1d!(view(img,:,x,z),σ[2],K,R[2],table[2])
        end
        # filter z
        (sz[3] < R[3]+2 ? throw(ArgumentError("Kernel radius is too large for z direction")) : nothing)
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[2]))
            (y,x) = Tuple(idx)
            O1gauss1d!(view(img,y,x,:),σ[3],K,R[3],table[3])
        end
    end
end

# function to perform constant time gaussian convolution on ND image
function O1gaussnd(img,σ,K=2)
    N = ndims(img)
    out = deepcopy(img)
    (N < 1 || N > 3 ? throw(ArgumentError("image should be of dimension 1,2 or 3")) : nothing)
    (N != length(σ) && length(σ) != 1 ? throw(ArgumentError("length of σ not equal to ndims of img and length of σ != 1")) : nothing)
    if N == 1
        R = findR(σ,K)
        (length(out) < R+2 ? throw(ArgumentError("Kernel radius is too large")) : nothing)
        O1gauss1d!(out,σ,K,R,precomputecoeff(σ,R,K))
    elseif N == 2
        sz = size(img)
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = [findR(σ[i],K) for i = 1:N]
        table = [precomputecoeff(σ[i],R[i],K) for i = 1:N]
        # filter x
        (sz[2] < R[1]+2 ? throw(ArgumentError("Kernel radius is too large for x direction")) : nothing)
        @inbounds Threads.@threads for y in 1:sz[1]
            O1gauss1d!(view(out,y,:),σ[1],K,R[1],table[1])
        end
        (sz[1] < R[2]+2 ? throw(ArgumentError("Kernel radius is too large for y direction")) : nothing)
        # filter y
        @inbounds Threads.@threads for x in 1:sz[2]
            O1gauss1d!(view(out,:,x),σ[2],K,R[2],table[2])
        end
    elseif N == 3
        sz = size(out)
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = [findR(σ[i],K) for i = 1:N]
        table = [precomputecoeff(σ[i],R[i],K) for i = 1:N]
        # filter x
        (sz[2] < R[1]+2 ? throw(ArgumentError("Kernel radius is too large for x direction")) : nothing)
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[3]))
            (y,z) = Tuple(idx)
            O1gauss1d!(view(out,y,:,z),σ[1],K,R[1],table[1])
        end
        # filter y
        (sz[1] < R[2]+2 ? throw(ArgumentError("Kernel radius is too large for y direction")) : nothing)
        @inbounds Threads.@threads for idx in CartesianIndices((sz[2],sz[3]))
            (x,z) = Tuple(idx)
            O1gauss1d!(view(out,:,x,z),σ[2],K,R[2],table[2])
        end
        # filter z
        (sz[3] < R[3]+2 ? throw(ArgumentError("Kernel radius is too large for z direction")) : nothing)
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[2]))
            (y,x) = Tuple(idx)
            O1gauss1d!(view(out,y,x,:),σ[3],K,R[3],table[3])
        end
    end
    return out
end

# function to performc onstant time gaussian convolution on 1D signal
function O1gauss1d!(f::AbstractArray,σ::Number,K::Integer,R::Integer,table::Tuple)
    # length of the array
    n = length(f)
    # temporary container of length R+2
    container = similar(f,eltype(f),R+2)
    # Zprev[k] = Σγcos(k,r)*f(r) for r in [-R,R]
    Zprev = @views table[2]*f[1:(R+1)].*2 .- table[2][:,1].*f[1]
    # Z[k] = Σγcos(k,r)*f(r+1) for r in [-R,R]
    Z = @views table[2]*f[2:(R+2)].+table[2][:,2:end]*f[1:R]
    # F0 = Σf(r) for r in [-R,R]
    F0 = @views 2*sum(f[2:(R+1)])+f[1]
    # fill the container
    container[1] = table[1]*(F0+sum(Zprev))
    # F0 = F0 - f(x-R) + f(x+R+1)
    F0 += f[R+2]-f[R+1]
    @inbounds for x in 2:(R+2)
        container[x] = table[1]*(F0+sum(Z))
        # F0 = F0 - f(x-R) + f(x+R+1)
        F0 += f[(x+R+1 > n ? 2*n-x-R-1 : x+R+1)]-f[(x-1-R < 1 ? R+2-x : x-R)]
        # Δ = f(x+R+1) - f(x-R) - f(x+R) + f(x-R-1)
        Δ = f[(x+R+1 > n ? 2*n-x-R-1 : x+R+1)]-f[(x-1-R < 1 ? R+2-x : x-R)]-f[(x+R > n ? 2*n-x-R : x+R)]+f[(x-R-2 < 1 ? R+3-x : x-1-R)]
        # update Z and Zprev
        @inbounds @simd for k in 1:K
            ξ = table[3][k]*Z[k]-Zprev[k]+table[2][k,end]*Δ
            Zprev[k] = Z[k]
            Z[k] = ξ
        end
    end
    # empty the container in f on the way
    @inbounds for x in (R+3):n
        f[x-R-2] = container[(x % (R+2) == 0 ? R+2 : x % (R+2))]
        container[(x % (R+2) == 0 ? R+2 : x % (R+2))] = table[1]*(F0+sum(Z))
        # F0 = F0 - f(x-R) + f(x+R+1)
        F0 += f[(x+R+1 > n ? 2*n-x-R-1 : x+R+1)]-f[(x-1-R < 1 ? R+2-x : x-R)]
        # Δ = f(x+R+1) - f(x-R) - f(x+R) + f(x-R-1)
        Δ = f[(x+R+1 > n ? 2*n-x-R-1 : x+R+1)]-f[(x-1-R < 1 ? R+2-x : x-R)]-f[(x+R > n ? 2*n-x-R : x+R)]+f[(x-R-2 < 1 ? R+3-x : x-1-R)]
        # update Z and Zprev
        @inbounds @simd for k in 1:K
            ξ = table[3][k]*Z[k]-Zprev[k]+table[2][k,end]*Δ
            Zprev[k] = Z[k]
            Z[k] = ξ
        end
    end
    # empty the last remaining values
    @inbounds for x in (n-R-1):n
        f[x] = container[(x % (R+2) == 0 ? R+2 : x % (R+2))]
    end
    return f
end

# function to find the kernel radius that minimize the spacial error
function findR(σ::Number,K::Integer)
    # Derivation of the Gaussian spatial kernel error
    function Dkernelerror(R,σ)
        ϕ = (2*R+1)/(2*σ)
        ψ = π*σ*(2*K+1)/(2*R+1)
        # derivative of Es(R)+Ef(K,R) = erfc(ϕ)+(erfc(ψ)-erfc(πσ)) wrt R
        # constant scaling factor 4/sqrt(π) has been omited
        return (ψ*exp(-ψ*ψ)-ϕ*exp(-ϕ*ϕ))/(2*R+1)
    end
    # find R minimizing kernel error via binary search
    R = 0
    # lower and upper bound for R is [2*σ,4*σ]
    low = eltype(R)(floor(2*σ))
    up = eltype(R)(ceil(4*σ))
    # binary search
    while up > low+1
        mid = eltype(R)(round((low+up)/2))
        if Dkernelerror(mid,σ) < 0
            low = mid
        else
            up = mid
        end
    end
    if abs(Dkernelerror(low,σ)) <= abs(Dkernelerror(up,σ))
        R = low
    else
        R = up
    end
    # kernel min size = 3
    if R < 3
        R = 3
    end
    return R
end

# function to precompute coefficient of DCT
function precomputecoeff(σ::Number,R::Integer,K::Integer)
    # precompute γcos and other constant
    ϕ = 2*π/(2*R+1)
    G0 = 1/(2*R+1)
    # cos(-x) = cos(x) and reflecting boundary condition on img
    γcos = [2*exp(-0.5*σ*σ*ϕ*ϕ*k*k)*cos(ϕ*k*r) for k = 1:K, r = 0:R]
    coeff = [2*cos(ϕ*k) for k = 1:K]
    return (G0,γcos,coeff)
end

# function to perform inplace gaussian convolution on ND image
function convnd!(img,σ,R)
    N = ndims(img)
    (N < 1 || N > 3 ? throw(ArgumentError("image should be of dimension 1,2 or 3")) : nothing)
    (N != length(σ) && length(σ) != 1 ? throw(ArgumentError("length of σ not equal to ndims of img and length of σ != 1")) : nothing)
    if N == 1
        k = gausskernel(σ,R)
        conv!(img,k)
    elseif N == 2
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = (length(R) == N ? R : fill(R,N))
        k = ntuple(i -> gausskernel(σ[i],R[i]), N)
        sz = size(img)
        # filter x
        @inbounds Threads.@threads for y in 1:sz[1]
            conv!(view(img,y,:),k[1])
        end
        # filter y
        @inbounds Threads.@threads for x in 1:sz[2]
            conv!(view(img,:,x),k[2])
        end
    elseif N == 3
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = (length(R) == N ? R : fill(R,N))
        k = ntuple(i -> gausskernel(σ[i],R[i]), N)
        sz = size(img)
        # filter x
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[3]))
            (y,z) = Tuple(idx)
            conv!(view(img,y,:,z),k[1])
        end
        # filter y
        @inbounds Threads.@threads for idx in CartesianIndices((sz[2],sz[3]))
            (x,z) = Tuple(idx)
            conv!(view(img,:,x,z),k[2])
        end
        # filter z
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[2]))
            (y,x) = Tuple(idx)
            conv!(view(img,y,x,:),k[3])
        end
    end
end

# function to perform gaussian convolution on ND image
function convnd(img,σ,R)
    N = ndims(img)
    out = deepcopy(img)
    (N < 1 || N > 3 ? throw(ArgumentError("image should be of dimension 1,2 or 3")) : nothing)
    (N != length(σ) && length(σ) != 1 ? throw(ArgumentError("length of σ not equal to ndims of img and length of σ != 1")) : nothing)
    if N == 1
        k = gausskernel(σ,R)
        conv!(out,k)
    elseif N == 2
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = (length(R) == N ? R : fill(R,N))
        k = ntuple(i -> gausskernel(σ[i],R[i]), N)
        sz = size(img)
        # filter x
        @inbounds Threads.@threads for y in 1:sz[1]
            conv!(view(out,y,:),k[1])
        end
        # filter y
        @inbounds Threads.@threads for x in 1:sz[2]
            conv!(view(out,:,x),k[2])
        end
    elseif N == 3
        σ = (length(σ) == N ? σ : fill(σ,N))
        R = (length(R) == N ? R : fill(R,N))
        k = ntuple(i -> gausskernel(σ[i],R[i]), N)
        sz = size(img)
        # filter x
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[3]))
            (y,z) = Tuple(idx)
            conv!(view(out,y,:,z),k[1])
        end
        # filter y
        @inbounds Threads.@threads for idx in CartesianIndices((sz[2],sz[3]))
            (x,z) = Tuple(idx)
            conv!(view(out,:,x,z),k[2])
        end
        # filter z
        @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[2]))
            (y,x) = Tuple(idx)
            conv!(view(out,y,x,:),k[3])
        end
    end
    return out
end

# function to perform gaussian convolution on 1D signal
function conv!(f::AbstractArray,kernel::AbstractArray)
    # length of img
    n = length(f)
    # kernel radius
    R::Integer = (length(kernel)-1)/2
    # check if the kernel radius isn't too large
    (n < R+1 ? throw(ArgumentError("Kernel radius is too large")) : nothing)
    # temporary container of length R+1
    container = zeros(eltype(f),R+1)
    # fill the container
    @inbounds for x in 1:(R+1)
        @inbounds for k in 1:(2*R+1)
            # use mirror boundary f(-x) = f(x)
            container[x] += kernel[k]*f[(x+(k-1-R) < 1 ? 2-(x+k-1-R) : (x+(k-1-R) > n ? 2*n-x-(k-1-R) : x+(k-1-R)))]
        end
    end
    # empty the container in f on the way
    @inbounds for x in (R+2):n
        f[x-R-1] = container[(x % (R+1) == 0 ? R+1 : x % (R+1))]
        container[(x % (R+1) == 0 ? R+1 : x % (R+1))]::eltype(f) = 0
        @inbounds for k in 1:(2*R+1)
            container[(x % (R+1) == 0 ? R+1 : x % (R+1))] += kernel[k]*f[(x+(k-1-R) > n ? 2*n-x-(k-1-R) : x+(k-1-R))]
        end
    end
    # empty the last remaining values
    @inbounds for x in (n-R):n
        f[x] = container[(x % (R+1) == 0 ? R+1 : x % (R+1))]
    end
    return f
end

# function to create a gaussian kernel of radius R with variance σ
function gausskernel(σ::Number,R::Integer)
    return [exp(-(x^2)/(2*σ^2))/(σ*sqrt(2*π)) for x = -R:R]
end