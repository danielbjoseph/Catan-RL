"""
Rules profile system.

A RulesProfile toggles optional game subsystems so the game can be trained
on a simplified curriculum first (spec Phase 3) and expanded later.

Built-in profiles:
  standard              full rules, 10 VP to win
  simplified_v1         no dev cards (hence no largest army), 10 VP to win
  standard_trading      full rules plus player trades, 10 VP to win
  simplified_trading_v1 no dev cards, player trades enabled, 10 VP to win

Profiles can also be loaded from YAML files in configs/rules_<name>.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class RulesProfile:
    name: str = "standard"
    dev_cards_enabled: bool = True
    win_vp: int = 10
    trades_enabled: bool = False
    max_trades_per_turn: int = 3

    @classmethod
    def get(cls, profile: Union[None, str, "RulesProfile"]) -> "RulesProfile":
        """Resolve None / builtin name / RulesProfile instance to a RulesProfile."""
        if profile is None:
            return STANDARD
        if isinstance(profile, RulesProfile):
            return profile
        if profile in _BUILTIN:
            return _BUILTIN[profile]
        # Fall back to a YAML config if one exists for this name
        yaml_path = _CONFIG_DIR / f"rules_{profile}.yaml"
        if yaml_path.exists():
            return cls.load(yaml_path)
        raise ValueError(
            f"Unknown rules profile {profile!r}; "
            f"expected one of {sorted(_BUILTIN)} or a configs/rules_<name>.yaml file"
        )

    @classmethod
    def load(cls, name_or_path: Union[str, Path]) -> "RulesProfile":
        """Load a profile from configs/rules_<name>.yaml or an explicit path."""
        import yaml

        path = Path(name_or_path)
        if not path.suffix:  # bare name like "simplified_v1"
            path = _CONFIG_DIR / f"rules_{name_or_path}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            dev_cards_enabled=bool(data.get("dev_cards_enabled", True)),
            win_vp=int(data.get("win_vp", 10)),
            trades_enabled=bool(data.get("trades_enabled", False)),
            max_trades_per_turn=int(data.get("max_trades_per_turn", 3)),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dev_cards_enabled": self.dev_cards_enabled,
            "win_vp": self.win_vp,
            "trades_enabled": self.trades_enabled,
            "max_trades_per_turn": self.max_trades_per_turn,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RulesProfile":
        if d is None:
            return STANDARD
        return cls(
            name=d["name"],
            dev_cards_enabled=d["dev_cards_enabled"],
            win_vp=d["win_vp"],
            trades_enabled=d.get("trades_enabled", False),
            max_trades_per_turn=d.get("max_trades_per_turn", 3),
        )


STANDARD = RulesProfile(name="standard", dev_cards_enabled=True, win_vp=10)
SIMPLIFIED_V1 = RulesProfile(name="simplified_v1", dev_cards_enabled=False, win_vp=10)
STANDARD_TRADING = RulesProfile(
    name="standard_trading", dev_cards_enabled=True, win_vp=10, trades_enabled=True
)
SIMPLIFIED_TRADING_V1 = RulesProfile(
    name="simplified_trading_v1", dev_cards_enabled=False, win_vp=10, trades_enabled=True
)

_BUILTIN = {
    p.name: p
    for p in (STANDARD, SIMPLIFIED_V1, STANDARD_TRADING, SIMPLIFIED_TRADING_V1)
}
