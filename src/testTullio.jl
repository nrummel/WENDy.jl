using Tullio, Test
using Test: @test
using BenchmarkTools: @btime
##
M = rand(1:20, 3, 7)
##
@tullio S[1,c] := M[r,c]  # sum over r ∈ 1:3, for each c ∈ 1:7
S_true = sum(M, dims=1) 
@test S == S_true
##
@tullio Q[ρ,c] := M[ρ,c] + sqrt(S[1,c])  # loop over ρ & c, no sum -- broadcasting
@test Q ≈ M .+ sqrt.(S)
##
mult(M,Q) = @tullio P[x,y] := M[x,c] * Q[y,c]  # sum over c ∈ 1:7 -- matrix multiplication
@test mult(M,Q) ≈ M * transpose(Q)
##
R = [rand(Int8, 3, 4) for δ in 1:5]

@tullio T[j,i,δ] := R[δ][i,j] + 10im  # three nested loops -- concatenation
@test T == permutedims(cat(R...; dims=3), (2,1,3)) .+ 10im

@tullio (max) X[i] := abs2(T[j,i,δ])  # reduce using max, over j and δ
@test X == dropdims(maximum(abs2, T, dims=(1,3)), dims=(1,3))

dbl!(M, S) = @tullio M[r,c] = 2 * S[1,c]  # write into existing matrix, M .= 2 .* S
dbl!(M, S)
@test all(M[r,c] == 2*S[1,c] for r ∈ 1:3, c ∈ 1:7)

## 
M = 1028 
K = 250 
D = 1
J = 2
gradF = rand(1:20, M,D,J)
V = rand(1:20, K, M);
##
mymult(V,gradF) = @tullio gradG[k,d,j] := V[k,m] * gradF[m,d,j] 
@time gradG = mymult(V, gradF);
@time begin
gradG_true = zeros(K,D,J)
for j = 1:J
    gradG_true[:,:,j] = V*gradF[:,:,j]
end
end;
@test gradG == gradG_true