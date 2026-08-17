# Rules Audit — Catan Base Game (3–4 players)

Systematic pass over the official Catan base-game rules (5th/6th edition, 3–4
players) against the engine in `catan_rl/env/`. Every row is grounded in
either a test in `tests/`, a commit that fixed the rule, or an explicit
rationale for an intentional simplification.

**Status legend**

| Status | Meaning |
|---|---|
| `correct` | Implemented per the official rule; evidence is a test in the suite |
| `fixed` | Was wrong, fixed during the rules-audit task series; evidence is the commit subject |
| `simplified` | Intentional deviation; evidence is the rationale |

Engine files audited: `rules.py`, `validators.py`, `scoring.py`,
`game_state.py`, `player_state.py`, `board.py`.

---

## 1. Setup

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| Board is 19 hexes: 4 wood, 3 brick, 4 sheep, 4 wheat, 3 ore, 1 desert | `board.py` `_STANDARD_RESOURCES` | correct | `test_mechanics.py::TestBoardGeometry::test_resource_counts` |
| 18 number tokens: one 2, one 12, two each of 3–6 and 8–11; desert gets none | `board.py` `_STANDARD_TOKENS`, `BoardConfig.standard` | correct | `test_mechanics.py::TestBoardGeometry::test_token_counts` |
| 9 harbors: 4 generic 3:1, 5 resource-specific 2:1 | `board.py` `_PORT_DEFS`, `_build_ports` | correct | `test_mechanics.py::TestBoardGeometry::test_9_ports` |
| Robber starts on the desert | `game_state.py` `GameState.new_game` (`robber_hex=config.desert_hex`) | correct | Set directly at construction; exercised by every game in `test_env_smoke.py::TestAECSmoke::test_single_game_completes` |
| Setup is snake order: each player places settlement+road forward P0→Pn, then a second pair in reverse Pn→P0 | `rules.py` `_advance_setup` | correct | `tests/test_rules.py::TestDistanceRule::test_all_setup_settlements_respect_distance` (drives full setup); `test_mechanics.py::TestSetupInitialResources::test_second_settlement_gives_adjacent_resources` (asserts reverse pass reached) |
| Distance rule applies during setup (no settlement adjacent to another) | `validators.py` `_setup_settlement_actions` | correct | `tests/test_rules.py::TestDistanceRule::test_no_adjacent_settlement_placement` |
| Setup road must attach to the settlement just placed | `validators.py` `_setup_road_actions` | correct | Enforced by restricting to edges of the player's road-less settlement; exercised legally end-to-end by `tests/test_rules.py::TestDistanceRule::test_all_setup_settlements_respect_distance` |
| Setup placements are free | `rules.py` `_build_settlement` / `_build_road` setup branches (no `spend`) | correct | `test_mechanics.py::TestSetupInitialResources::test_setup_phase_no_resource_cost` |
| Second settlement yields one resource per adjacent producing hex | `rules.py` `_build_settlement` `SETUP_SETTLEMENT_2` branch | correct | `test_mechanics.py::TestSetupInitialResources::test_second_settlement_gives_adjacent_resources` |

## 2. Turn structure

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| A turn is: roll dice → resolve production/robber → build/trade/dev freely → pass | `game_state.py` `Phase`, `rules.py` `_roll_dice` / `_end_turn` | correct | `test_mechanics.py::TestTurnProgression::test_full_round_robin` |
| Turns pass clockwise (fixed player order, wraps around) | `rules.py` `_end_turn` | correct | `test_mechanics.py::TestTurnProgression::test_end_turn_advances_player`, `::test_player_wraps_around` |
| Dice are 2d6 | `rules.py` `_roll_dice` (`randint(1,6)` twice) | correct | `test_mechanics.py::TestTurnProgression::test_full_round_robin` (dice consumed each turn); dice cleared per `::test_dice_cleared_on_end_turn` |
| Rolling is mandatory; no building/trading before the roll | `validators.py` `legal_actions` ROLL branch | correct | `test_action_mask.py::TestMaskCorrectness::test_roll_dice_legal_at_roll_phase`; `test_rules_audit.py::TestDevCardBeforeRoll::test_simplified_v1_roll_phase_only_roll_dice` |
| One dev card may be played at any time during your turn, **including before the roll** | `validators.py` ROLL branch + `_dev_card_actions`; `rules.py` `_main_return_phase`, `rolled_this_turn` | fixed | Commit `fix(rules): allow one dev card before the roll`; `test_rules_audit.py::TestDevCardBeforeRoll` (7 tests) |
| Domestic (player-to-player) trading on your turn | — | simplified | Not implemented; planned for Package B (trading + personalities). Only maritime trade exists. |

## 3. Resource production

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| Every hex with the rolled number produces for all adjacent buildings; settlement = 1, city = 2 | `rules.py` `_produce_resources` | correct | `test_mechanics.py::TestResourceProduction::test_settlement_yields_one_resource`, `::test_city_yields_two_resources`, `::test_only_correct_token_produces`, `::test_multiple_settlements_same_roll` |
| The robber's hex does not produce | `rules.py` `_produce_resources` (`hex_id == state.robber_hex` skip) | correct | `test_mechanics.py::TestResourceProduction::test_robber_blocks_production` |
| The desert never produces | `board.py` (desert token 0), `_produce_resources` desert skip | correct | `test_mechanics.py::TestResourceProduction::test_desert_never_produces` |
| Bank shortage: if the bank cannot fully supply **all** players owed a resource type, no one receives that type — unless exactly one player is owed it, who takes what's left | `rules.py` `_produce_resources` (owed-matrix payout) | fixed | Commit `fix(rules): official bank-shortage rule for production`; `test_rules_audit.py::TestBankShortageOnProduction` (3 tests) |
| Bank holds 19 of each resource | `game_state.py` `_BANK_START` | correct | Constant; all builds/trades/discards return cards to the bank — `test_mechanics.py::TestBuildingCosts::test_road_costs_wood_brick`, `TestDiscardOnSeven::test_discard_removes_resource_returns_to_bank` |

## 4. Robber and rolling a 7

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| On a 7: no production; every player with more than 7 cards discards half, rounded down | `rules.py` `_handle_seven` | correct | `test_mechanics.py::TestDiscardOnSeven::test_player_with_8_must_discard_4`, `::test_player_with_7_no_discard`, `::test_player_with_14_discards_7`, `::test_multiple_players_discard` |
| Discarded cards return to the bank | `rules.py` `_discard` | correct | `test_mechanics.py::TestDiscardOnSeven::test_discard_removes_resource_returns_to_bank` |
| All affected players discard (simultaneously in the official game) | `rules.py` `_discard` (players resolve in player-id order, one card per action) | simplified | Discards resolve one card per action per player, sequentially. Net effect is identical to simultaneous discarding — production and stealing are frozen until every obligation clears. `test_mechanics.py::TestDiscardOnSeven::test_discard_obligation_consumed_one_at_a_time` |
| The roller must then move the robber to a **different** hex | `validators.py` `_robber_actions` | correct | `test_mechanics.py::TestRobber::test_cannot_move_robber_to_current_hex`, `::test_robber_covers_all_other_hexes` |
| The roller steals 1 random card from an adjacent opponent **who has cards**; if no adjacent opponent has cards, no steal happens | `validators.py` `_steal_actions`; `rules.py` `_move_robber` (both filter `total_resources > 0`) | fixed | Commit `fix(rules): steal targets must hold cards; longest-road revocation`; `test_rules_audit.py::TestStealTargetMustHoldCards` (3 tests) |
| Stolen card is random from the victim's hand | `rules.py` `_steal` (uniform over victim's cards) | correct | `test_mechanics.py::TestRobber::test_robber_steal_transfers_resource` |
| No opponent adjacent → no steal at all | `rules.py` `_move_robber` no-target branch | correct | `test_mechanics.py::TestRobber::test_steal_no_action_when_no_opponents_adjacent` |

## 5. Building — costs and placement

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| Road costs 1 wood + 1 brick | `player_state.py` `BUILD_COSTS["road"]` | correct | `test_mechanics.py::TestBuildingCosts::test_road_costs_wood_brick` |
| Settlement costs 1 wood + 1 brick + 1 sheep + 1 wheat | `BUILD_COSTS["settlement"]` | correct | `test_mechanics.py::TestBuildingCosts::test_settlement_costs_wood_brick_sheep_wheat` |
| City costs 2 wheat + 3 ore | `BUILD_COSTS["city"]` | correct | `test_mechanics.py::TestBuildingCosts::test_city_costs_wheat_ore` |
| Dev card costs 1 sheep + 1 wheat + 1 ore | `BUILD_COSTS["dev_card"]`, `rules.py` `_buy_dev_card` | correct | `test_rules_audit.py::TestDevCardDeckAndPurchase::test_buy_dev_card_costs_sheep_wheat_ore` |
| Piece limits: 15 roads, 5 settlements, 4 cities per player | `player_state.py` `MAX_ROADS`/`MAX_SETTLEMENTS`/`MAX_CITIES`, `can_afford_*` | correct | `test_rules_audit.py::TestPieceLimits::test_road_limit_15`, `::test_settlement_limit_5`, `::test_city_limit_4` |
| Roads must connect to your own road network or buildings; occupied edges are unavailable | `validators.py` `_connected_road_actions` | correct | `tests/test_rules.py::TestRoadConnectivity::test_roads_must_connect_in_main_phase` |
| A road may not continue past an opponent's settlement/city | `_connected_road_actions` (enemy-vertex endpoint skip) | correct | `test_rules_audit.py::TestRoadBlockedByEnemyBuilding::test_cannot_extend_road_through_enemy_settlement` |
| Settlements must sit on your own road and respect the distance rule | `validators.py` `_settlement_actions` (reachable-from-roads set + neighbor scan, same check as setup) | correct | `tests/test_rules.py::TestDistanceRule` (identical neighbor check); connectivity via the reachable set built from `player.road_vertices` |
| A city replaces your **own** settlement; the settlement piece returns to supply | `validators.py` `_main_actions` (city targets = own settlements), `rules.py` `_build_city` | correct | `tests/test_rules.py::TestCityUpgrade::test_city_only_on_own_settlement`, `::test_city_not_on_opponent_settlement`; piece return asserted in `test_mechanics.py::TestBuildingCosts::test_city_costs_wheat_ore` |
| Build costs are paid to the bank | `rules.py` `_build_*` (bank credit per cost) | correct | `test_mechanics.py::TestBuildingCosts::test_road_costs_wood_brick`, `::test_city_costs_wheat_ore` |

## 6. Maritime trade

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| 4:1 with the bank from anywhere | `board.py` `best_trade_rate`, `validators.py` `_maritime_trade_actions` | correct | `test_mechanics.py::TestMaritimeTrade::test_default_rate_is_4`, `::test_trade_4to1_no_port` |
| 3:1 with a generic harbor (building on a harbor vertex) | same | correct | `test_mechanics.py::TestMaritimeTrade::test_generic_port_rate_3` |
| 2:1 for the matching resource at a specific harbor | same | correct | `test_mechanics.py::TestMaritimeTrade::test_specific_port_rate_2`, `::test_trade_deducts_correct_amount` |
| Cannot trade a resource for itself; bank must hold the requested resource | `_maritime_trade_actions` (`get != give`, `bank_has`) | correct | `test_mechanics.py::TestMaritimeTrade::test_cannot_trade_same_resource` |

## 7. Development cards

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| Deck is 25 cards: 14 knights, 2 road building, 2 year of plenty, 2 monopoly, 5 VP | `game_state.py` `_DEV_DECK` | correct | `test_rules_audit.py::TestDevCardDeckAndPurchase::test_deck_composition_14_2_2_2_5` |
| No purchase once the deck is empty | `validators.py` `_main_actions` (`len(state.dev_deck) > 0`) | correct | `test_rules_audit.py::TestDevCardDeckAndPurchase::test_cannot_buy_when_deck_empty` |
| At most one dev card played per turn | `player_state.py` `has_played_dev_card`, `validators.py` `_dev_card_actions` | correct | `tests/test_rules.py::TestDevCards::test_cannot_play_two_dev_cards_per_turn`; pre-roll play consumes the allowance per `test_rules_audit.py::TestDevCardBeforeRoll::test_preroll_dev_card_play_consumes_one_per_turn_allowance` |
| A card cannot be played the turn it was bought | `player_state.py` `dev_cards_new` / `end_turn_refresh_dev_cards` | correct | `test_mechanics.py::TestDevCardEffects::test_cannot_play_card_bought_this_turn`, `::test_card_moves_to_playable_after_end_turn` |
| Knight: move the robber (and steal), add to army | `rules.py` `_play_knight` | correct | `test_mechanics.py::TestLargestArmy::test_knight_play_increments_army` |
| Road building: place 2 free roads (1 if only 1 piece/spot remains) | `rules.py` `_play_road_building`, `ROAD_BUILDING_1/2` phases | correct | `test_mechanics.py::TestDevCardEffects::test_road_building_gives_two_free_roads` |
| Year of plenty: take any 2 resources from the bank | `rules.py` `_play_year_of_plenty` | correct | `test_mechanics.py::TestDevCardEffects::test_year_of_plenty_gives_two_from_bank`, `::test_year_of_plenty_same_resource_twice` (if the bank lacks a chosen card, that card is silently skipped — negligible edge case) |
| Monopoly: all opponents surrender all cards of the named resource | `rules.py` `_play_monopoly` | correct | `test_mechanics.py::TestDevCardEffects::test_monopoly_steals_all_of_resource`, `::test_monopoly_does_not_steal_other_resources` |
| VP cards stay hidden and count toward victory automatically; they are never "played" | `player_state.py` `hidden_vp`; catalog slot 253 permanently masked in `_dev_card_actions` | fixed | Commit `fix(rules): VP dev cards count automatically, never played`; `test_rules_audit.py::TestVictoryPointDevCardsAutoCount` (3 tests) |

## 8. Special awards

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| Longest Road: first player with a continuous road of 5+ gets 2 VP | `scoring.py` `update_longest_road`, `LONGEST_ROAD_MIN = 5` | correct | `test_mechanics.py::TestLongestRoad::test_chain_of_5_gives_holder`, `::test_chain_of_4_not_longest_road_holder`; VP per `TestScoring::test_longest_road_worth_2_vp` |
| Longest road counts the single longest continuous route (branches/loops handled) | `scoring.py` `compute_longest_road` (edge-DFS) | correct | `test_mechanics.py::TestLongestRoad::test_chain_of_5_gives_longest_road_5`, `tests/test_rules.py::TestLongestRoad::test_straight_road_length` |
| The route is broken at an opponent's settlement/city | `compute_longest_road` (enemy-vertex cut) | correct | `test_mechanics.py::TestLongestRoad::test_enemy_settlement_breaks_road` |
| A challenger must be **strictly** longer to take the card | `update_longest_road` | correct | `test_mechanics.py::TestLongestRoad::test_equal_length_does_not_transfer`, `::test_longer_road_takes_title` |
| Revocation: if the holder's road is split below 5 (e.g. by an opponent's settlement), the card goes to the unique player at the new maximum ≥ 5, else to nobody; ties (including at first award) leave it unawarded | `update_longest_road` (recomputed on every road **and** settlement build) | fixed | Commit `fix(rules): steal targets must hold cards; longest-road revocation`; `test_rules_audit.py::TestLongestRoadRevocation` (3 tests: revoked-to-none, transfer-to-unique, tie-nobody) |
| Largest Army: first player with 3+ knights played gets 2 VP; strictly more to take it | `scoring.py` `update_largest_army`, `LARGEST_ARMY_MIN = 3` | correct | `test_mechanics.py::TestLargestArmy::test_3_knights_gives_largest_army`, `::test_2_knights_not_enough`, `::test_more_knights_transfers_army`, `::test_tie_does_not_transfer`, `::test_largest_army_worth_2_vp` |

## 9. Victory

| Rule | Engine location | Status | Evidence |
|---|---|---|---|
| 10 VP wins (settlement 1, city 2, awards 2 each, VP cards 1 each) | `scoring.py` `compute_vp` / `check_winner`; `rules_profile.py` `win_vp` | correct | `test_mechanics.py::TestScoring::test_win_at_exactly_10_vp`, `::test_no_win_at_9_vp`, `::test_settlement_worth_1_vp`, `::test_city_worth_2_vp`, `::test_vp_dev_card_counts_toward_win` |
| Exactly one winner; game ends immediately | `rules.py` `_check_win` (sets `GAME_OVER`) | correct | `test_env_smoke.py::TestAECSmoke::test_exactly_one_winner` |
| Victory may only be declared on your own turn | `rules.py` `_check_win` runs after **every** action and scans all players | simplified | Harmless in practice: only the acting player's VP can increase mid-turn, with one exotic exception — a settlement that splits the Longest Road holder's route can transfer the card (and 2 VP) to a third player mid-turn, whom the engine then declares immediately. Officially they would win at the start of their own turn anyway (they keep the card until then, and no game state between can strip 10+ VP without their consent). |
| 3–4 players only in the base game | `game_state.py` `new_game(n_players=4)` default; 4-seat action catalog (`actions.py` steal slots 201–204) | simplified | 5–6 player extension (extra pieces, special building phase) is out of scope and not implemented. |

---

## Findings from the systematic pass

Beyond the four fixed rules above (bank shortage, VP auto-count,
dev-card-before-roll, steal-target + longest-road revocation), the pass found
**no further rule inaccuracies requiring code changes**. Notes:

1. **Dead code**: `scoring.py::longest_road_for_player` is never called
   anywhere in the codebase and computes lengths with an *empty*
   enemy-vertex set (its docstring admits the caller must supply the full
   picture). The live path is `compute_longest_road`, which handles enemy
   interruptions correctly. Not a rule bug, but a trap for future readers —
   candidate for deletion in a cleanup pass.
2. **Year of plenty vs. empty bank**: if a chosen resource is exhausted, the
   engine silently gives fewer than 2 cards. The official rules do not
   address this corner explicitly (tournament rulings vary); the legal-action
   generator does not currently forbid choosing an exhausted resource. Effect
   is negligible and noted in the table row above.
3. **Previously untested-but-correct rules** (dev deck composition, dev card
   cost, deck exhaustion, piece limits, road blocked by enemy building) now
   have direct evidence tests in
   `tests/test_rules_audit.py` (`TestDevCardDeckAndPurchase`,
   `TestPieceLimits`, `TestRoadBlockedByEnemyBuilding`), added with this
   audit.
