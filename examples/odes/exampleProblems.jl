EXAMPLES = []
for path_to_file in readdir(@__DIR__, join=true)
    (path_to_file == @__FILE__) && continue
    @info "Loading example from $path_to_file"
    push!(
        EXAMPLES, 
        include(path_to_file)
    ) 
end



