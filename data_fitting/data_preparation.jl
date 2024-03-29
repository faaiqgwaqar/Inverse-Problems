### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 38925e1a-817b-4575-9768-02fb68bec2d6
begin
	import Pkg; Pkg.activate()
	using CairoMakie, ColorSchemes, CSV, DataFrames, StatsBase, JLD2, PlutoUI, Colors, Random
end

# ╔═╡ bc2cd3cb-c982-4f4e-a74c-5119a6851b04
TableOfContents()

# ╔═╡ 637f2cf6-0cb0-4ea2-b1c5-6caccf46c06d
begin
	import AlgebraOfGraphics as AoG
	AoG.set_aog_theme!(fonts=[AoG.firasans("Light"), AoG.firasans("Light")])
	update_theme!(fontsize=20, linewidth=3, resolution=(500, 400))
end

# ╔═╡ 6a246c23-9d0b-4ed9-85ab-18ac16e6de88
md"## read raw data
the good runs, where $t=0$ corresponds to taking lime out of fridge, are runs 11 and 12.
"

# ╔═╡ cc51a112-7d01-46d0-86b0-85e142a6b7ca
function read_data(filename::String)
	data = CSV.read(joinpath("..", "Arduino Results", filename), DataFrame)

	# unit conversion
	data[:, "Time [s]"] = data[:, "Time [ms]"] / 1000.0
	data[:, "t [min]"]  = data[:, "Time [s]"] / 60.0
	rename!(data, "Temp [C]" => "T [°C]")
	
	return data[:, ["t [min]", "T [°C]"]]
end

# ╔═╡ 3c2e28af-1506-4ca6-be48-4e38bd67097d
run = 11 # run 11 and 12

# ╔═╡ 60a97118-1fc1-403b-97fa-fce162e8d6fd
filename = "limev$run.csv"

# ╔═╡ dba3e7ce-cf99-4ad7-b1fe-560aea998273
data = read_data(filename)

# ╔═╡ d0920710-db3f-4b23-8879-3aac8a45a5dd
md"## air temperature
take as after 4 hr.
"

# ╔═╡ 5fddc914-4df3-476f-9dfe-53e7bff1e38c
T̄ₐ = mean(filter(row -> row["t [min]"] > 400, data)[:, "T [°C]"])

# ╔═╡ 2c062784-d93d-4077-aa33-ffb62ba947f3
md"use only first 10 hrs"

# ╔═╡ da39d074-f729-4442-8072-34423f743d63
filter!(row -> row["t [min]"] < 600.0, data)

# ╔═╡ c2e1bf81-e416-4b93-9d83-84129f9c6001
md"## initial temperature"

# ╔═╡ dd37ff0f-fc4f-4076-9da5-f7735ac3d829
T₀ = data[1, "T [°C]"]

# ╔═╡ 33bdddea-6f17-4ea9-b074-d7d9bebc4321
md"## viz the data"

# ╔═╡ f68fcc2b-7f79-4896-83a7-f0141e8bf02d
function viz_data(data::DataFrame, Tₐ=nothing; shld_i_save::Bool=false)
	max_t = maximum(data[:, "t [min]"])
	
	fig = Figure()
	ax  = Axis(fig[1, 1], 
		       xlabel="time, t [min]",
		       ylabel="temperature [°C]",
               xtickalign=1, ytickalign=1
	)
	vlines!(ax, [0.0], color="gray", linewidth=1)
	if ! isnothing(Tₐ)
		hlines!(ax, Tₐ, style=:dash, linestyle=:dot, 
			label=rich("θ", subscript("∞")), color=Cycled(3))
	end
	scatter!(data[:, "t [min]"], data[:, "T [°C]"], 
		label=rich("{(t", subscript("i"), ", θ", subscript("i,obs"), ")}"), strokewidth=1)
	axislegend(position=:rb)
	xlims!(-0.03*max_t, 1.03*max_t)
	if shld_i_save
		save("raw_data_run_$run.pdf")
	end
	fig
end

# ╔═╡ 5d3acc22-3a32-4ff6-9a25-cd9a7594a34a
viz_data(data, T̄ₐ)

# ╔═╡ f07ab15c-fe20-4138-a28e-6eca0f774ef5
md"## downsample
choose $n$ before and after equilibrium points at random. (but include the first).
equilibrium = 95% of way to air temp.
"

# ╔═╡ a84116af-b2e4-4853-aff9-f19738ae5733
vcat(data[1:2, :], data[2:3, :])

# ╔═╡ f950ade9-1a1a-4398-a954-635d03599afa
function downsample(data::DataFrame, n_per_block::Int, n_blocks::Int=5)
	T_cuts = range(minimum(data[:, "T [°C]"]), 
		maximum(data[:, "T [°C]"]), length=n_blocks+1)

	new_data = data[[1, end], :] # keep initial condition

	Random.seed!(run == 12 ? 97330 : 97329)

	for b = 1:n_blocks
		# get data from this block
		data_this_block = filter(
			row -> (row["T [°C]"] >= T_cuts[b]) & (row["T [°C]"] <= T_cuts[b+1]),
			data
		)
		nb_sample = n_per_block
		if b == 1 #|| b == n_blocks
			nb_sample -= 1
		end
		ids_sample = sample(1:nrow(data_this_block), nb_sample, replace=false)
		new_data = vcat(new_data, data_this_block[ids_sample, :])
	end

	new_data = sort(new_data, "t [min]")
	
	# # remove if within 2 min of each other.
	id_keep = trues(nrow(new_data))
	for i = 2:nrow(new_data)
		if new_data[i, "t [min]"] - new_data[i-1, "t [min]"] < 5
			id_keep[i] = false
		end
	end

	return new_data[id_keep, :]
end

# ╔═╡ e35322f8-5065-43ef-a846-77de00f4d065
downsampled_data = downsample(data, 5, 4)

# ╔═╡ e51f13f6-a48f-4ced-a028-62e8f745b7c3
T̄ₐ

# ╔═╡ 31c49bdf-6e6e-495f-b20a-1805a88fc76f
Tₐ = data[end, "T [°C]"]

# ╔═╡ 9c6d23d9-fcc2-4704-86c5-50a13d6af2e0
viz_data(downsampled_data, Tₐ, shld_i_save=true)

# ╔═╡ a5535284-1d62-4748-b9c8-c28fa2dc3eb3
md"## export data for other tasks"

# ╔═╡ ac7665d4-d2ab-42f1-b9d5-f0ab7884e331
downsampled_data[:, "t [hr]"] = downsampled_data[:, "t [min]"] / 60.0

# ╔═╡ ed732170-4f38-4dec-ae73-d45822c7e490
select!(downsampled_data, ["t [hr]", "T [°C]"])

# ╔═╡ 221e0b99-e2ac-47f9-90ff-ab7263d2e178
rename!(downsampled_data, "T [°C]" => "θ [°C]")

# ╔═╡ 322c60c0-1e04-4bd8-a3f9-2802e8deb605
jldsave("data_run_$run.jld2"; data=downsampled_data[1:end-1, :], θᵃⁱʳ=data[end, "T [°C]"])

# ╔═╡ 15905572-a13c-48d3-9e2f-f2bde6783dcc
md"for the tutorial..."

# ╔═╡ f0a3c3fa-01a7-4a59-9c97-25ebb1dcc44c
CSV.write("../tutorial/lime_temp_param_id.csv", load("data_run_12.jld2")["data"])

# ╔═╡ 9b325cec-cb8c-4274-ab73-802d87d2c372
load("data_run_12.jld2")

# ╔═╡ Cell order:
# ╠═38925e1a-817b-4575-9768-02fb68bec2d6
# ╠═bc2cd3cb-c982-4f4e-a74c-5119a6851b04
# ╠═637f2cf6-0cb0-4ea2-b1c5-6caccf46c06d
# ╟─6a246c23-9d0b-4ed9-85ab-18ac16e6de88
# ╠═cc51a112-7d01-46d0-86b0-85e142a6b7ca
# ╠═3c2e28af-1506-4ca6-be48-4e38bd67097d
# ╠═60a97118-1fc1-403b-97fa-fce162e8d6fd
# ╠═dba3e7ce-cf99-4ad7-b1fe-560aea998273
# ╟─d0920710-db3f-4b23-8879-3aac8a45a5dd
# ╠═5fddc914-4df3-476f-9dfe-53e7bff1e38c
# ╟─2c062784-d93d-4077-aa33-ffb62ba947f3
# ╠═da39d074-f729-4442-8072-34423f743d63
# ╟─c2e1bf81-e416-4b93-9d83-84129f9c6001
# ╠═dd37ff0f-fc4f-4076-9da5-f7735ac3d829
# ╟─33bdddea-6f17-4ea9-b074-d7d9bebc4321
# ╠═f68fcc2b-7f79-4896-83a7-f0141e8bf02d
# ╠═5d3acc22-3a32-4ff6-9a25-cd9a7594a34a
# ╟─f07ab15c-fe20-4138-a28e-6eca0f774ef5
# ╠═a84116af-b2e4-4853-aff9-f19738ae5733
# ╠═f950ade9-1a1a-4398-a954-635d03599afa
# ╠═e35322f8-5065-43ef-a846-77de00f4d065
# ╠═e51f13f6-a48f-4ced-a028-62e8f745b7c3
# ╠═31c49bdf-6e6e-495f-b20a-1805a88fc76f
# ╠═9c6d23d9-fcc2-4704-86c5-50a13d6af2e0
# ╟─a5535284-1d62-4748-b9c8-c28fa2dc3eb3
# ╠═ac7665d4-d2ab-42f1-b9d5-f0ab7884e331
# ╠═ed732170-4f38-4dec-ae73-d45822c7e490
# ╠═221e0b99-e2ac-47f9-90ff-ab7263d2e178
# ╠═322c60c0-1e04-4bd8-a3f9-2802e8deb605
# ╟─15905572-a13c-48d3-9e2f-f2bde6783dcc
# ╠═f0a3c3fa-01a7-4a59-9c97-25ebb1dcc44c
# ╠═9b325cec-cb8c-4274-ab73-802d87d2c372
