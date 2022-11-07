using SUNSpinWaveTheory
using Documenter

DocMeta.setdocmeta!(SUNSpinWaveTheory, :DocTestSetup, :(using SUNSpinWaveTheory); recursive=true)

makedocs(;
    modules=[SUNSpinWaveTheory],
    authors="wwangnju <wwangnju@163.com> and contributors",
    repo="https://github.com/Quantum-Many-Body/SUNSpinWaveTheory.jl/blob/{commit}{path}#{line}",
    sitename="SUNSpinWaveTheory.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/SUNSpinWaveTheory.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/SUNSpinWaveTheory.jl",
    devbranch="master",
)
