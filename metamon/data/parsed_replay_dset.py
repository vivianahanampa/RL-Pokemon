import os
import json
import random
import csv
import copy
from typing import Optional, Dict, Tuple, List, Any, Set
from datetime import datetime
from collections import defaultdict

from torch.utils.data import Dataset
import lz4.frame
import numpy as np
import tqdm

import metamon
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    UniversalState,
    ActionSpace,
    UniversalAction,
)
from metamon.data.download import download_parsed_replays


class ParsedReplayDataset(Dataset):
    """An iterable dataset of "parsed replays"

    Parsed replays are records of Pokémon Showdown battles that have been converted to the partially observed
    point-of-view of a single player, matching the problem our agents face in the RL environment. They are created by the
    `metamon.backend.replay_parser` module from "raw" Showdown replay logs
    downloaded from publicly available battles.

    This is a pytorch `Dataset` that returns (nested_obs, actions, rewards, dones) trajectory tuples,
    where:
    - nested_obs: List of numpy arrays of length seq_len (arrays may have different shapes).
      If the observation space is a dict, this becomes a dict of lists of arrays for each key.
    - actions: Dict with keys:
        - "chosen": list (length seq_len) of actions taken by the agent in the chosen action space
        - "legal": list (length seq_len) of sets of legal actions available at each timestep in the chosen action space
        - "missing": list (length seq_len) of bools indicating the action is missing (should probably be masked)
    - rewards: Numpy array of shape (seq_len,)
    - dones: Numpy array of shape (seq_len,)

    Note that depending on the observation space, you may need a custom pad_collate_fn in the pytorch dataloader
    to handle the variable-shaped arrays in nested_obs.

    Missing actions are a bool mask where idx i = True if action i is missing (actions[i] == -1, or was originally
    missing but has since been filled by some prediction scheme). Missing actions are caused by player choices that
    are not revealed to spectators and do not show up in the replay logs (e.g., paralysis, sleep, flinch).

    Data is stored as interface.UniversalStates and observations and rewards are created on the fly. This
    means we no longer have to create new versions of the parsed replay dataset to experiment with different
    observation spaces or reward functions.

    Example:
        ```python
        dset = ParsedReplayDataset(
            observation_space=TokenizedObservationSpace(
                DefaultObservationSpace(),
                tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            ),
            reward_function=DefaultShapedReward(),
            formats=["gen1nu"],
            verbose=True,
        )

        obs, action_infos, rewards, dones = dset[0]
        ```

    Args:
        observation_space: The observation space to use. Must be an instance of `interface.ObservationSpace`.
        reward_function: The reward function to use. Must be an instance of `interface.RewardFunction`.
        dset_root: The root directory of the parsed replays. If not specified, the parsed replays will be
            downloaded and extracted from the latest version of the huggingface dataset, but this may take minutes.
        formats: A list of formats to load (e.g. ["gen1ou", "gen2ubers"]). Defaults to all supported formats
            (Gen 1-4 ou, uu, nu, and ubers), but this will take a long time to download and extract the first time.
        wins_losses_both: Whether to only load the perspective of players who won their battle, lost their
            battle, or both. {"wins", "losses", "both"}
        min_rating: The minimum rating of battles to load (in ELO). Note that most replays are Unrated, which
            is mapped to 1000 ELO (the minimum rating on Showdown). In reality many of these battles were played
            as part of tournaments and should probably not be ignored.
        max_rating: The maximum rating of battles to load (in ELO). In Generations 1-4, ELO ratings above 1500
            are very good.
        min_date: The minimum date of battles to load (as a datetime). Our dataset begins in 2014. Many replays
            from 2021-2024 are missing due to a Showdown database issue. See the raw-replay dataset README on
            HF for a visual timeline of the dataset.
        max_date: The maximum date of battles to load (as a datetime). The latest date available will depend on
            the current version of the parsed replays dataset.
        max_seq_len: The maximum sequence length to load. Trajectories are randomly sliced to this length.
        verbose: Whether to print progress bars while loading large datasets.
        shuffle: Whether to shuffle the filenames. Defaults to False.
        use_cached_filenames: Whether to use the cached filenames from a manifest.csv file saved during a previous experiment with this replay directory.
            Saves time on startup of large training runs. Defaults to False.
    """

    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        reward_function: RewardFunction,
        dset_root: Optional[str] = None,
        formats: Optional[List[str]] = None,
        wins_losses_both: str = "both",
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        max_seq_len: Optional[int] = None,
        verbose: bool = False,
        shuffle: bool = False,
        use_cached_filenames: bool = False,
    ):
        formats = formats or metamon.SUPPORTED_BATTLE_FORMATS

        if dset_root is None:
            for format in formats:
                path_to_format_data = download_parsed_replays(format)
            dset_root = os.path.dirname(path_to_format_data)

        assert dset_root is not None and os.path.exists(dset_root)
        self.observation_space = copy.deepcopy(observation_space)
        self.action_space = copy.deepcopy(action_space)
        self.reward_function = copy.deepcopy(reward_function)
        self.dset_root = dset_root
        self.formats = formats
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.min_date = min_date
        self.max_date = max_date
        self.wins_losses_both = wins_losses_both
        self.verbose = verbose
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.manifest_path = os.path.join(self.dset_root, "manifest.csv")
        if os.path.exists(self.manifest_path) and use_cached_filenames:
            with open(self.manifest_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                self.filenames = [row[0] for row in reader]
            if verbose:
                print(f"Loaded {len(self.filenames)} battles from {self.manifest_path}")
            if shuffle:
                random.shuffle(self.filenames)
        else:
            self.refresh_files()

    def parse_battle_date(self, filename: str) -> datetime:
        # parsed replays saved by our own gym env will have hour/minute/sec
        # while Showdown replays will not.
        date_str = filename.split("_")[-2]

        # Try the more common format first (without time) for faster parsing
        try:
            return datetime.strptime(date_str, "%m-%d-%Y")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%m-%d-%Y-%H:%M:%S")
            except ValueError:
                raise ValueError(f"Could not parse date string: {date_str}")

    def refresh_files(self):
        self.filenames = []

        def _rating_to_int(rating: str) -> int:
            try:
                return int(rating)
            except ValueError:
                return 1000

        bar = lambda it, desc: (
            it if not self.verbose else tqdm.tqdm(it, desc=desc, colour="red")
        )

        has_rating_filter = self.min_rating is not None or self.max_rating is not None
        has_date_filter = self.min_date is not None or self.max_date is not None
        has_result_filter = self.wins_losses_both in ("wins", "losses")

        for format in self.formats:
            path = os.path.join(self.dset_root, format)
            if not os.path.exists(path):
                if self.verbose:
                    print(
                        f"Requested data for format `{format}`, but did not find {path}"
                    )
                continue

            # Get all files at once and filter by extension first
            all_files = os.listdir(path)
            json_files = [f for f in all_files if f.endswith((".json", ".json.lz4"))]

            for filename in bar(json_files, desc=f"Finding {format} battles"):
                name_without_ext = (
                    filename[:-9] if filename.endswith(".json.lz4") else filename[:-5]
                )

                parts = name_without_ext.split("_")
                if len(parts) != 7:
                    continue

                battle_id, rating_str, p1_name, _, p2_name, mm_dd_yyyy, result = parts

                if has_result_filter:
                    if self.wins_losses_both == "wins" and result != "WIN":
                        continue
                    if self.wins_losses_both == "losses" and result != "LOSS":
                        continue

                battle_id_clean = (
                    battle_id.replace("[", "").replace("]", "").replace(" ", "").lower()
                )
                if format not in battle_id_clean:
                    continue

                if has_rating_filter:
                    rating = _rating_to_int(rating_str)
                    if (self.min_rating is not None and rating < self.min_rating) or (
                        self.max_rating is not None and rating > self.max_rating
                    ):
                        continue

                if has_date_filter:
                    try:
                        date = self.parse_battle_date(filename)
                        if (self.min_date is not None and date < self.min_date) or (
                            self.max_date is not None and date > self.max_date
                        ):
                            continue
                    except ValueError:
                        continue
                self.filenames.append(os.path.join(path, filename))

        if self.shuffle:
            random.shuffle(self.filenames)

        with open(self.manifest_path, "w") as f:
            if self.verbose:
                print(
                    f"Writing {self.manifest_path} with {len(self.filenames)} battles"
                )
            writer = csv.writer(f)
            writer.writerow(["filename"])  # Write header row
            for filename in self.filenames:
                writer.writerow([filename])

        if self.verbose:
            print(f"Dataset contains {len(self.filenames)} battles")

    def __len__(self):
        return len(self.filenames)

    def _load_json(self, filename: str) -> dict:
        if filename.endswith(".json.lz4"):
            with lz4.frame.open(filename, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
        elif filename.endswith(".json"):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unknown file extension: {filename}")
        return data

    def load_filename(self, filename: str):
        data = self._load_json(filename)
        states = [UniversalState.from_dict(s) for s in data["states"]]
        # reset the observation space, then call once on each state, which lets
        # any history-dependent features behave as they would in an online battle
        self.observation_space.reset()
        obs = [self.observation_space.state_to_obs(s) for s in states]
        # TODO: handle case where observation space is not a dict. don't have one to test yet.
        nested_obs = defaultdict(list)
        for o in obs:
            for k, v in o.items():
                nested_obs[k].append(v)
        action_infos = {
            "chosen": [],
            "legal": [],
            "missing": [],
        }
        # NOTE: the replay parser leaves a blank final action
        for s, a_idx in zip(states, data["actions"][:-1]):
            universal_action = UniversalAction(action_idx=a_idx)
            missing = universal_action.missing
            chosen_agent_action = self.action_space.action_to_agent_output(
                s, universal_action
            )
            legal_universal_actions = UniversalAction.maybe_valid_actions(s)
            legal_agent_actions = set(
                self.action_space.action_to_agent_output(s, l)
                for l in legal_universal_actions
            )
            action_infos["chosen"].append(chosen_agent_action)
            action_infos["legal"].append(legal_agent_actions)
            action_infos["missing"].append(missing)
        rewards = np.array(
            [
                self.reward_function(s_t, s_t1)
                for s_t, s_t1 in zip(states[:-1], states[1:])
            ],
            dtype=np.float32,
        )
        dones = np.zeros_like(rewards, dtype=bool)
        dones[-1] = True

        if self.max_seq_len is not None:
            # s s s s s s s s
            # a a a a a a a
            # r r r r r r r
            # d d d d d d d
            safe_start = random.randint(
                0, max(len(action_infos["chosen"]) - self.max_seq_len, 0)
            )
            nested_obs = {
                k: v[safe_start : safe_start + 1 + self.max_seq_len]
                for k, v in nested_obs.items()
            }
            action_infos = {
                k: v[safe_start : safe_start + self.max_seq_len]
                for k, v in action_infos.items()
            }
            rewards = rewards[safe_start : safe_start + self.max_seq_len]
            dones = dones[safe_start : safe_start + self.max_seq_len]

        return dict(nested_obs), action_infos, rewards, dones

    def random_sample(self):
        filename = random.choice(self.filenames)
        return self.load_filename(filename)

    def __getitem__(self, i) -> Tuple[
        Dict[str, list[np.ndarray]],
        Dict[str, list[Any]],
        np.ndarray,
        np.ndarray,
    ]:
        return self.load_filename(self.filenames[i])


if __name__ == "__main__":
    from argparse import ArgumentParser
    from metamon.interface import (
        DefaultShapedReward,
        get_observation_space,
        TokenizedObservationSpace,
        DefaultActionSpace,
    )
    from metamon.tokenizer import get_tokenizer

    parser = ArgumentParser()
    parser.add_argument("--dset_root", type=str, default=None)
    parser.add_argument("--formats", type=str, default=None, nargs="+")
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    args = parser.parse_args()

    dset = ParsedReplayDataset(
        dset_root=args.dset_root,
        observation_space=TokenizedObservationSpace(
            get_observation_space(args.obs_space),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
        ),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        formats=args.formats,
        verbose=True,
        shuffle=True,
    )
    for i in tqdm.tqdm(range(len(dset))):
        obs, actions, rewards, dones = dset[i]
