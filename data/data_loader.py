
import math
import os
import sys

import json
import torch
from torch.utils.data import Dataset, DataLoader

from data.label_mapping import map_fulltime_label, map_htft_label, map_handicap_label
from nn_modules.embedding.token_schema import TOKEN_COUNT

top_config = {}
cwd = os.getcwd()  # current working directory
cfp = os.path.dirname(os.path.abspath(__file__))  # current file path
os.chdir(cfp)
root_full_path = os.path.abspath("..")
sys.path.append(root_full_path)
config_full_path = os.path.join(root_full_path, 'config/config.json')
if not os.path.exists(config_full_path):
    print(f"{config_full_path} doesn't exist.")
    exit(0)
with open(config_full_path, 'r') as file:
    top_config = json.load(file)
os.chdir(cwd)

# TODO(diwei): Modify the class' detail including preproess, dataset or remove it.
class TopDataset(Dataset):
    def __init__(self):
        self.num = 10

        self.xs = torch.rand(top_config["model_params"]["source_length"],
                             self.num, top_config["model_params"]["d_model"])
        self.us = torch.rand(top_config["model_params"]["target_length"],
                             self.num, top_config["model_params"]["d_model"])
        self.gts = torch.rand(
            self.num, top_config["model_params"]["num_classes"])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return {"x": self.xs[:, idx, :], "u": self.us[:, idx, :], "gt": self.gts[idx, :]}


# TODO(diwei): Encapsulate the dataset and the dataloader into a class
train_dataset = TopDataset()
validation_dataset = TopDataset()
batch_size = top_config["train_params"]["batch_size"]
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)


def preprocess(batch):
    """The function to do preprocess the dataset.

    Parameters
    ----------
    batch : dict
        The data batch including source, target and ground truth from 
        data loader.

    Returns
    -------
    tuple
        Token values and ground truth labels.
    """
    return batch["token_values"], batch["gt"]


def load_json_or_jsonl(file_path: str):
    """Load JSON array/object or JSONL records.

    Parameters
    ----------
    file_path : str
        Path to a .json or .jsonl file.

    Returns
    -------
    list
        List of dict records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    if file_path.endswith(".jsonl"):
        records = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with open(file_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {file_path}: {type(payload)}")


def flatten_prematch_features(record: dict):
    """Convert nested pre-match feature dict into a numeric feature vector.

    This keeps the initial Phase 1 implementation simple and explicit. Features can
    be replaced by a dedicated feature catalog pipeline later.
    """
    home = record.get("home", {})
    away = record.get("away", {})

    def safe_float(value, default=0.0):
        if value is None:
            return float(default)
        return float(value)

    return [
        safe_float(home.get("league_position")),
        safe_float(away.get("league_position")),
        safe_float(home.get("points_last_5")),
        safe_float(away.get("points_last_5")),
        safe_float(home.get("goals_scored_last_5")),
        safe_float(away.get("goals_scored_last_5")),
        safe_float(home.get("goals_conceded_last_5")),
        safe_float(away.get("goals_conceded_last_5")),
        safe_float(home.get("elo_rating")),
        safe_float(away.get("elo_rating")),
        safe_float(home.get("promoted_this_season")),
        safe_float(away.get("promoted_this_season")),
        safe_float(home.get("team_strength_prior")),
        safe_float(away.get("team_strength_prior")),
        safe_float(home.get("strength_gap_vs_division_avg")),
        safe_float(away.get("strength_gap_vs_division_avg")),
    ]


def extract_market_tokens(markets):
    """Extract fixed market-related token values from market records."""

    def safe_float(value, default=0.0):
        if value is None:
            return float(default)
        return float(value)

    fulltime = None
    handicap = None
    for market in markets:
        if market.get("play_type") == "fulltime_1x2":
            fulltime = market
        elif market.get("play_type") == "handicap_1x2":
            handicap = market

    fulltime_home = safe_float((fulltime or {}).get("home_odds"))
    fulltime_draw = safe_float((fulltime or {}).get("draw_odds"))
    fulltime_away = safe_float((fulltime or {}).get("away_odds"))

    handicap_line = safe_float((handicap or {}).get("handicap_line"))
    handicap_home = safe_float((handicap or {}).get("home_odds"))
    handicap_away = safe_float((handicap or {}).get("away_odds"))

    return [
        fulltime_home,
        fulltime_draw,
        fulltime_away,
        handicap_line,
        handicap_home,
        handicap_away,
    ]


class PreMatchLotteryDataset(Dataset):
    """Dataset for pre-match lottery tasks.

    Each item returns a common feature tensor and labels for available tasks.
    Missing task labels are set to -1 and can be ignored in training.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "match_id": sample["match_id"],
            "features": torch.tensor(sample["features"], dtype=torch.float32),
            "fulltime_label": torch.tensor(sample.get("fulltime_label", -1), dtype=torch.long),
            "htft_label": torch.tensor(sample.get("htft_label", -1), dtype=torch.long),
            "handicap_label": torch.tensor(sample.get("handicap_label", -1), dtype=torch.long),
        }


def build_samples(prematch_records, match_meta_records, lottery_market_records=None):
    """Build model samples from processed records.

    Parameters
    ----------
    prematch_records : list[dict]
        Records from prematch feature table.
    match_meta_records : list[dict]
        Records from match metadata table.
    lottery_market_records : list[dict] or None
        Optional market records. Required for handicap labels.
    """
    meta_by_match = {item["match_id"]: item for item in match_meta_records}
    market_by_match = {}

    if lottery_market_records is not None:
        for item in lottery_market_records:
            market_by_match.setdefault(item["match_id"], []).append(item)

    samples = []
    for record in prematch_records:
        match_id = record["match_id"]
        if match_id not in meta_by_match:
            continue

        meta = meta_by_match[match_id]
        fulltime_label = map_fulltime_label(meta["final_result"])
        htft_label = map_htft_label(
            meta["halftime_home_goals"],
            meta["halftime_away_goals"],
            meta["home_goals"],
            meta["away_goals"],
        )

        handicap_label = -1
        markets = market_by_match.get(match_id, [])
        for market in markets:
            if market.get("play_type") == "handicap_1x2":
                handicap_label = map_handicap_label(
                    meta["home_goals"],
                    meta["away_goals"],
                    market.get("handicap_line"),
                )
                break

        token_values = flatten_prematch_features(record) + extract_market_tokens(markets)

        samples.append(
            {
                "match_id": match_id,
                "features": flatten_prematch_features(record),
                "token_values": token_values,
                "fulltime_label": fulltime_label,
                "htft_label": htft_label,
                "handicap_label": handicap_label,
                "promoted_match": int(
                    bool(record.get("home", {}).get("promoted_this_season", 0))
                    or bool(record.get("away", {}).get("promoted_this_season", 0))
                ),
            }
        )

    return samples


def create_prematch_data_loader(samples, batch_size=32, shuffle=True):
    dataset = PreMatchLotteryDataset(samples)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


class TransformerPreMatchDataset(Dataset):
    """Adapter dataset that maps tabular pre-match features into Transformer inputs."""

    def __init__(self, samples, source_length, target_length, d_model, label_key="fulltime_label"):
        self.samples = samples
        self.source_length = source_length
        self.target_length = target_length
        self.d_model = d_model
        self.label_key = label_key

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        token_values = torch.tensor(sample["token_values"], dtype=torch.float32)
        if token_values.numel() != TOKEN_COUNT:
            raise ValueError(f"token_values length mismatch: expected {TOKEN_COUNT}, got {token_values.numel()}")
        gt = torch.tensor(sample.get(self.label_key, -1), dtype=torch.long)
        return {
            "match_id": sample["match_id"],
            "token_values": token_values,
            "gt": gt,
            "promoted_match": torch.tensor(sample.get("promoted_match", 0), dtype=torch.long),
        }


def split_samples_by_season(samples, match_meta_records, season_split):
    """Split samples by season definitions from config/data_config.json."""
    season_by_match = {row["match_id"]: row["season"] for row in match_meta_records}
    split_to_seasons = {name: set(seasons) for name, seasons in season_split.items()}

    split_samples = {name: [] for name in split_to_seasons}
    for sample in samples:
        season = season_by_match.get(sample["match_id"])
        if season is None:
            continue
        for split_name, seasons in split_to_seasons.items():
            if season in seasons:
                split_samples[split_name].append(sample)
                break
    return split_samples


def create_transformer_prematch_data_loaders(
    processed_dir,
    season_split,
    source_length,
    target_length,
    d_model,
    batch_size=32,
    label_key="fulltime_label",
):
    """Create train/validation loaders from processed JSONL data for Transformer training."""
    prematch_records = load_json_or_jsonl(os.path.join(processed_dir, "prematch_features.jsonl"))
    match_meta_records = load_json_or_jsonl(os.path.join(processed_dir, "match_meta.jsonl"))
    lottery_market_records = load_json_or_jsonl(os.path.join(processed_dir, "lottery_market.jsonl"))

    samples = build_samples(prematch_records, match_meta_records, lottery_market_records)
    split_samples = split_samples_by_season(samples, match_meta_records, season_split)

    train_dataset = TransformerPreMatchDataset(
        split_samples.get("train", []),
        source_length=source_length,
        target_length=target_length,
        d_model=d_model,
        label_key=label_key,
    )
    validation_dataset = TransformerPreMatchDataset(
        split_samples.get("validation", []),
        source_length=source_length,
        target_length=target_length,
        d_model=d_model,
        label_key=label_key,
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, split_samples


def create_transformer_prematch_loader_for_split(
    processed_dir,
    season_split,
    split_name,
    source_length,
    target_length,
    d_model,
    batch_size=32,
    label_key="fulltime_label",
    shuffle=False,
):
    """Create a single split DataLoader (train/validation/test) for Transformer tasks."""
    prematch_records = load_json_or_jsonl(os.path.join(processed_dir, "prematch_features.jsonl"))
    match_meta_records = load_json_or_jsonl(os.path.join(processed_dir, "match_meta.jsonl"))
    lottery_market_records = load_json_or_jsonl(os.path.join(processed_dir, "lottery_market.jsonl"))

    samples = build_samples(prematch_records, match_meta_records, lottery_market_records)
    split_samples = split_samples_by_season(samples, match_meta_records, season_split)

    if split_name not in split_samples:
        raise ValueError(f"Unknown split '{split_name}'. Available: {list(split_samples.keys())}")

    dataset = TransformerPreMatchDataset(
        split_samples.get(split_name, []),
        source_length=source_length,
        target_length=target_length,
        d_model=d_model,
        label_key=label_key,
    )
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, split_samples
