from .._types import DefaultConfig  # noqa: D100


def get_ddm_normalt_config() -> DefaultConfig:
    """
    Get the default configuration for the DDM with Normal ndt variability.

    Trial non-decision time is ``Normal(t, st)``, so ``st`` is the kernel SD
    and the kernel support is unbounded.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "st"],
        "choices": [-1, 1],
        "description": "The DDM with Normal variability in non-decision time",
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "ddm_normalt.onnx",
                "backend": "jax",
                # The Normal ndt kernel is unbounded (real density extends below
                # t - st), so the admissibility floor sits at the kernel's
                # practical 3-sigma edge, t - 3 * st.
                "ndt_edge_width": 3.0,
                "default_priors": {},
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 2.5),
                    "z": (0.3, 0.7),
                    "t": (0.25, 2.25),
                    "st": (1e-3, 0.25),
                },
                "extra_fields": None,
            },
        },
    }
