"""Shared utils for third-party library examples."""

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import dm_env
import numpy as np
import tree
from gymnasium import spaces
from meltingpot.utils.substrates import substrate
from meltingpot.utils.substrates.wrappers import observables

PLAYER_STR_FORMAT = 'player_{index}'
# _IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']
_IGNORE_KEYS = []


def downsample_observation(array: np.ndarray, scaled) -> np.ndarray:
    """Downsample image component of the observation.
    Args:
      array: RGB array of the observation provided by substrate
      scaled: Scale factor by which to downsaple the observation

    Returns:
      ndarray: downsampled observation
    """

    frame = cv2.resize(
        array, (array.shape[1] // scaled, array.shape[0] // scaled), interpolation=cv2.INTER_AREA)
    return frame


def timestep_to_observations(timestep: dm_env.TimeStep) -> Mapping[str, Any]:
    """Extract observation from timestep structure returned from substrate."""

    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value
            for key, value in observation.items()
            if key not in _IGNORE_KEYS
        }
    return gym_observations


def _downsample_multi_timestep(timestep: dm_env.TimeStep, scaled) -> dm_env.TimeStep:
    return timestep._replace(
        observation=[{k: downsample_observation(v, scaled) if k == 'RGB' or k == "WORLD.RGB" else v for k, v in
                      observation.items()
                      } for observation in timestep.observation])


def _downsample_multi_spec(spec, scaled):
    return dm_env.specs.Array(shape=(spec.shape[1] // scaled, spec.shape[0] // scaled, spec.shape[2]), dtype=spec.dtype)


def remove_unrequired_observations_from_space(
        observation: spaces.Dict) -> spaces.Dict:
    """Remove observations that are not supposed to be used by policies."""

    return spaces.Dict({
        key: observation[key] for key in observation if key not in _IGNORE_KEYS
    })


def spec_to_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
    """Converts a dm_env nested structure of specs to a Gym Space.

    BoundedArray is converted to Box Gym spaces. DiscreteArray is converted to
    Discrete Gym spaces. Using Tuple and Dict spaces recursively as needed.

    Args:
      spec: The nested structure of specs

    Returns:
      The Gym space corresponding to the given spec.
    """
    if isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f'Unsupported dtype {spec.dtype}')
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([spec_to_space(s) for s in spec])
    elif isinstance(spec, dict):
        return spaces.Dict({key: spec_to_space(s) for key, s in spec.items()})
    else:
        raise ValueError('Unexpected spec of type {}: {}'.format(type(spec), spec))


class DownSamplingSubstrateWrapper(observables.ObservableLab2dWrapper):
    """Downsamples 8x8 sprites returned by substrate to 1x1.

    This related to the observation window of each agent and will lead to observation RGB shape to reduce
    from [88, 88, 3] to [11, 11, 3]. Other downsampling scales are allowed but not tested. This will lead
    to significant speedups in training.
    """

    def __init__(self, substrate: substrate.Substrate, scaled):
        super().__init__(substrate)
        self._scaled = scaled

    def reset(self) -> dm_env.TimeStep:
        timestep = super().reset()
        return _downsample_multi_timestep(timestep, self._scaled)

    def step(self, actions) -> dm_env.TimeStep:
        timestep = super().step(actions)
        return _downsample_multi_timestep(timestep, self._scaled)

    def observation_spec(self) -> Sequence[Mapping[str, Any]]:
        spec = super().observation_spec()
        return [
            {k: _downsample_multi_spec(v, self._scaled) if k == 'RGB' or k == "WORLD.RGB" else v for k, v in s.items()}
            for s in spec]
