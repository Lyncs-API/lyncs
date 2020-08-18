from lyncs_setuptools import setup

install_requires = [
    "simpy",
]

extras_require = {
}

setup(
    "lyncs_hmc",
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=["Python", "HMC", "Hybrid", "Hamiltonian", "MonteCarlo",],
)
