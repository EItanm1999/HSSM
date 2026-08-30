from .._types import DefaultConfig  # noqa: D100


def get_ddm_st_config() -> DefaultConfig:
    """
    Get the default configuration for the DDM with uniform ndt variability.

    Trial non-decision time is ``Uniform(t - st, t + st)``, so ``st`` is the
    kernel half-width and the kernel SD is ``st / sqrt(3)``.

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
        "description": "The DDM with uniform variability in non-decision time",
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "ddm_st.onnx",
                "backend": "jax",
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
