from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from decoder_confidence.decoding._decoder_adapter import _build_relay_decoder


def _install_fake_relay_bp(monkeypatch: pytest.MonkeyPatch) -> type:
    class FakeRelayDecoderF64:
        last_kwargs: dict | None = None

        def __init__(
            self,
            check_matrix,
            error_priors,
            *,
            pre_iter=80,
            set_max_iter=60,
            gamma_dist_interval=...,
        ) -> None:
            self.check_matrix = check_matrix
            self.error_priors = error_priors
            self.pre_iter = pre_iter
            self.set_max_iter = set_max_iter
            self.gamma_dist_interval = gamma_dist_interval
            FakeRelayDecoderF64.last_kwargs = {
                "pre_iter": pre_iter,
                "set_max_iter": set_max_iter,
                "gamma_dist_interval": gamma_dist_interval,
            }

    relay_pkg = types.ModuleType("relay_bp")
    bp_mod = types.ModuleType("relay_bp.bp")
    bp_mod.RelayDecoderF64 = FakeRelayDecoderF64
    monkeypatch.setitem(sys.modules, "relay_bp", relay_pkg)
    monkeypatch.setitem(sys.modules, "relay_bp.bp", bp_mod)
    return FakeRelayDecoderF64


def test_relay_bp_pre_iter_option_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls = _install_fake_relay_bp(monkeypatch)

    decoder = _build_relay_decoder(
        np.array([[1, 0]], dtype=np.uint8),
        np.array([0.1, 0.2], dtype=np.float64),
        {"pre_iter": 7, "set_max_iter": 11, "gamma_dist_interval": [0.1, 0.2]},
    )

    assert isinstance(decoder, fake_cls)
    assert fake_cls.last_kwargs == {
        "pre_iter": 7,
        "set_max_iter": 11,
        "gamma_dist_interval": (0.1, 0.2),
    }


def test_relay_bp_unknown_option_fails_before_pybind_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_relay_bp(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported relay-bp decoder option"):
        _build_relay_decoder(
            np.array([[1, 0]], dtype=np.uint8),
            np.array([0.1, 0.2], dtype=np.float64),
            {"does_not_exist": 1},
        )


def test_relay_bp_rejects_per_iter_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_relay_bp(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported relay-bp decoder option.*per_iter"):
        _build_relay_decoder(
            np.array([[1, 0]], dtype=np.uint8),
            np.array([0.1, 0.2], dtype=np.float64),
            {"per_iter": 7},
        )
