### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ e59c82de-4b75-11ed-07bf-633bef93f934
using Unitful

# ╔═╡ 25d373f0-fca2-43e6-97ff-20a6947cc0af
m = 100u"g"

# ╔═╡ da1decc8-02fd-4fe2-b597-3ade45a1fc44
cₚ = 4u"kJ/(kg*K)"

# ╔═╡ c3a99ed5-e801-4313-876f-b4be0736859a
U = ((2.5+25)/2)u"J/(m^2*s*K)"

# ╔═╡ 42576346-8ae0-40e2-b9d4-bc226f3a9350
r = 2.5u"cm"

# ╔═╡ 9a7587bc-9217-49ba-86dd-831b4e2f92b4
A = 4 * π * r ^ 2

# ╔═╡ 456897ef-ecdc-46a3-8daa-408200278e5d
τ = uconvert(u"hr", (m * cₚ) / (U * A))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
Unitful = "~1.12.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "787e7aee8f35448ebe36fe42bb004a1688cafb2d"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d57a4ed70b6f9ff1da6719f5f2713706d57e0d66"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╠═e59c82de-4b75-11ed-07bf-633bef93f934
# ╠═25d373f0-fca2-43e6-97ff-20a6947cc0af
# ╠═da1decc8-02fd-4fe2-b597-3ade45a1fc44
# ╠═c3a99ed5-e801-4313-876f-b4be0736859a
# ╠═42576346-8ae0-40e2-b9d4-bc226f3a9350
# ╠═9a7587bc-9217-49ba-86dd-831b4e2f92b4
# ╠═456897ef-ecdc-46a3-8daa-408200278e5d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
