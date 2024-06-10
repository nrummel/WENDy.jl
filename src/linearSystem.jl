function B!(B, Vp, uobs)
    @tullio B[k,d] = - Vp[k,m] * uobs[d,m]
    nothing
end

function G!(G, FF, w, V, _F!, uobs) 
    F!(FF, w,_F!,uobs)
    @tullio G[k,d] = V[k,m] * FF[m,d]
    nothing
end

function residual!(r, G,FF,B,w,V,Vp,_F!,uobs)
    G!(G, FF, w,V,_F!,uobs) 
    B!(B, Vp, uobs)
    r[:] = reshape(G, prod(size(G))) - reshape(B, prod(size(B)))
    nothing
end

function jacG!(jacG,JJ,V,w) 
    JJ = jacF(w)
    @tullio jacG[k,d1,d2] = V[k,m] * JJ[m,d1,d2]
    return jacG 
end 

function F!(F, w, _F!, uobs)
    for m in 1:Mp1
        _F!(view(F,m,:),w,uobs[:,m])
    end 
    nothing
end

function jacuF!(jacF, w, _jacuF!,uobs)
    for m in 1:Mp1
        @time _jacuF!(view(jacF,m,:,:), w, uobs[:,m])
    end 
    nothing
end

function jacwF!(jacF, w, _jacwF!, uobs)
    for m in 1:Mp1
        _jacwF!(view(jacF,m,:,:), w, uobs[:,m])
    end 
    nothing
end
function L0!(L0,Vp,sig)
    @tullio L0[k,m,d,d] = Vp[k,m]*sig[d]
    nothing
end 

function LL!(LL,JJ, w, sig, L0, _jacuF!, uobs) 
    jacuF!(JJ,w,_jacuF!, uobs)
    @tullio LL[k,m,d2,d1] = V[k,m] * JJ[m,d2,d1] * sig[d1]
    LL[:] += L0[:]
    nothing
end

function L!(L,LL,JJ, w, sig, L0, _jacuF!, uobs) 
    LL!(LL,JJ, w, sig, L0, _jacuF!, uobs) 
    permutedims!(L,LL,(1,3,2,4))
    nothing
end

function S!(S, L, LL,JJ, w, sig, L0, _jacuF!, uobs)

end



# @btime L_naive!($L1, $buf, $uobs,$w_rand)#;L0=ZZ)
# @btime L_naive!($L11, $FF, $w_rand)#;L0=ZZ)
# @time L_naive!(L1, buf, uobs,w_rand)#;L0=ZZ)
# @time L_naive!(L11, FF, w_rand)#;L0=ZZ)
# @btime L_naive2!($L2, $FF, $w_rand)#;L0=ZZ)
# relErr0 = norm(L11-L1)/norm(L1)
# relErr1 = norm(L2-L1)/norm(L1)
# relErr2 = norm(L_tullio-L1)/norm(L1)
# @info "relErr naive1 = $relErr0"
# @info "relErr naive2 = $relErr1"
# @info "relErr tullio = $relErr2"

# function L_naive!(L, buf, uobs, w; L0=L0)
#     for m in 1:Mp1
#         _jacuF!(buf, w, uobs[:,m])
#         for k in 1:K
#             for d1 in 1:D
#                 for d2 in 1:D
#                     L[k,m,d2,d1] = buf[d2,d1] * V[k,m]
#                 end
#             end
#         end
#     end
  
#     L += L0
#     nothing
# end
# function L_naive!(L, FF, w; L0=L0)
#     jacuF!(FF, w)
#     for k in 1:K
#         for m in 1:Mp1
#             for d1 in 1:D
#                 for d2 in 1:D
#                     L[k,m,d2,d1] = FF[m,d2,d1] * V[k,m]
#                 end
#             end
#         end
#     end
  
#     L += L0
#     nothing
# end

# function L_naive2!(L,FF, w; L0=L0)
#     jacuF!(FF, w)
#     for d1 in 1:D
#         for d2 in 1:D
#             for k in 1:K
#                 L[k,:,d2,d1] = V[k,:] .* FF[:,d2,d1] 
#             end      
#         end
#     end
#     L += L0
#     nothing
# end