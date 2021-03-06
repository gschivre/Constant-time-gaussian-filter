function Fastd2gauss(img, σ::Number=1.0, dir::String="x", K::Int64=2)
    if length(size(img)) > 3 || length(size(img)) < 1
        println("Choose a 1D signal or 2D/3D image")
    elseif (length(size(img)) == 1) && (dir != "x")
        println("Only \"x\" direction for 1D signal")
    elseif (length(size(img)) == 2) && (dir != "x") && (dir != "y")
        println("Only \"x\" or \"y\" direction for 2D image")
    elseif (length(size(img)) == 3) && (dir != "x") && (dir != "y") && (dir != "z")
        println("Only \"x\", \"y\" or \"z\" direction for 3D image")
    elseif minimum(σ) < 0
        println("σ out of range, must be >= 0")
    else
        # Derivation of the Gaussian spatial kernel error
        function Dkernelerror(R::Int64, σ::Float64)
            ϕ = (2*R+1)/(2*σ)
            ψ = π*σ*(2*K+1)/(2*R+1)
            # derivative of Es(R)+Ef(K,R) = erfc(ϕ)+(2ϕ^2-1)2ϕexp(-ϕ^2)/3sqrt(π)+(erfc(ψ)+(2/3ψ^2-1)2ψexp(-ψ^2)/sqrt(π)-erfc(πσ)) wrt R
            return (((ψ*ψ-0.5)*(ψ*ψ+0.75)*sqrt(π)*σ*(2*K+1))/(R*R+R+0.25))*exp(-ψ*ψ)-(((4*ϕ*ϕ)*(2*ϕ*ϕ+1))/(3*sqrt(π)*σ))*exp(-ϕ*ϕ)
        end
        if length(σ) > 1
            R = zeros(Int64,length(σ))
            for s in 1:length(σ)
                # Binary search
                low = Int64(floor(2*σ[s]))
                up = Int64(ceil(4*σ[s]))
                while up > low+1
                    mid = Int64(round((low+up)/2))
                    if Dkernelerror(mid, σ[s]) < 0
                        low = mid
                    else
                        up = mid
                    end
                end
                if abs(Dkernelerror(low, σ[s])) <= abs(Dkernelerror(up, σ[s]))
                    R[s] = low
                else
                    R[s] = up
                end
                # kernel min size = 3
                if R[s] < 3
                    R[s] = 3
                end
            end
        elseif length(σ) == 1
            # Binary search
            low = Int64(floor(2*σ))
            up = Int64(ceil(4*σ))
            while up > low+1
                mid = Int64(round((low+up)/2))
                if Dkernelerror(mid, σ) < 0
                    low = mid
                else
                    up = mid
                end
            end
            if abs(Dkernelerror(low, σ)) <= abs(Dkernelerror(up, σ))
                R = low
            else
                R = up
            end
            # kernel min size = 3
            if R < 3
                R = 3
            end
        end
        # Filter for one dimension
        function filter1d(f::Array{Float64,1}, σ::Float64, R::Int64, K::Int64)
            # output
            fo = zeros(length(f))
            # reflecting padding of f
            f = [f[(R+1):-1:2];f;f[(end-1):-1:(end-R-1)]]
            # precompute γkcos(ϕku) and G0
            ϕ = 2*π/(2*R+1)
            G0 = 1/(2*R+1)
            coeff = fill(Float64[],2)
            coeff[1] = 2 .*cos.(ϕ.*(1:K))
            coeff[2] = 2 .*exp.((-0.5*σ*σ*ϕ*ϕ).*(1:K).*(1:K)).*cos.((ϕ*R).*(1:K))
            Zprev = zeros(K)
            Z = zeros(K)
            for k in 1:K
                γcos = (2*exp(-0.5*σ*σ*ϕ*ϕ*k*k)).*cos.((ϕ*k).*(-R:R))
                Zprev[k] = sum(γcos.*f[1:(2*R+1)])
                Z[k] = sum(γcos.*f[2:(2*R+2)])
            end
            fo[1] = -G0*sum(Zprev)
            for x in 1:(length(fo)-1)
                fo[x+1] = -G0*sum(Z)
                # Δ = f(x+R+1) - f(x+R) - f(x-R) + f(x-R-1)
                Δ = f[2*R+2+x]-f[2*R+1+x]-f[x+1]+f[x]
                for k in 1:K
                    ξ = coeff[1][k]*Z[k]-Zprev[k]+coeff[2][k]*Δ
                    Zprev[k] = Z[k]
                    Z[k] = ξ
                end
            end
            return fo
        end
        # apply filter1d in each dimension
        sz = size(img)
        if length(sz) == 1
            # only one dimension
            imgo = filter1d(img,σ,R,K)
        elseif length(sz) == 2
            if dir == "x"
                if R >= sz[2]-1
                    println("Kernel size is too large")
                else
                    # return derivative in x direction
                    # output image
                    imgo = zeros(sz)
                    # filter x
                    @inbounds Threads.@threads for y in 1:sz[1]
                        imgo[y,:] = filter1d(img[y,:],σ,R,K)
                    end
                end
            elseif dir == "y"
                if R >= sz[1]-1
                    println("Kernel size is too large")
                else
                    # return derivative in y direction
                    # output image
                    imgo = zeros(sz)
                    # filter y
                    @inbounds Threads.@threads for x in 1:sz[2]
                        imgo[:,x] = filter1d(img[:,x],σ,R,K)
                    end
                end
            end
        elseif length(sz) == 3
            if dir == "x"
                if R >= sz[2]-1
                    println("Kernel size is too large")
                else
                    # return derivative in x direction
                    # output image
                    imgo = zeros(sz)
                    # filter x
                    @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[3]))
                        (y,z) = Tuple(idx)
                        imgo[y,:,z] = filter1d(img[y,:,z],σ,R,K)
                    end
                end
            elseif dir == "y"
                if R >= sz[1]-1
                    println("Kernel size is too large")
                else
                    # return derivative in y direction
                    # output image
                    imgo = zeros(sz)
                    # filter y
                    @inbounds Threads.@threads for idx in CartesianIndices((sz[2],sz[3]))
                        (x,z) = Tuple(idx)
                        imgo[:,x,z] = filter1d(img[:,x,z],σ,R,K)
                    end
                end
            elseif dir == "z"
                if R >= sz[3]-1
                    println("Kernel size is too large")
                else
                    # return derivative in z direction
                    # output image
                    imgo = zeros(sz)
                    # filter z
                    @inbounds Threads.@threads for idx in CartesianIndices((sz[1],sz[2]))
                        (y,x) = Tuple(idx)
                        imgo[y,x,:] = filter1d(img[y,x,:],σ,R,K)
                    end
                end
            end
        end
        return imgo
    end
end
