from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PromotionProposal:
    baseline_fingerprint: str        # hash of the baseline state_dict, computed live at evaluate() time;
                                      # promote() recomputes live and compares -- catches baseline drift
                                      # between evaluate() and promote(), not just a same-constant echo
    deployment_snapshot: Any         # Optional[_DeploymentSnapshot]; the exact object promote() deploys
    candidate_fingerprint: str       # hash of the live shadow head's sufficient statistics at evaluate()
                                      # time; promote() recomputes from the live shadow and compares --
                                      # rejects a proposal made stale by observe() calls since evaluate()
    config_version: str              # VRSEConfig.version -- informational only, not the security check
    config_fingerprint: str          # hash of the full VRSEConfig, computed live at evaluate() time;
                                      # promote() recomputes live and compares (catches config field
                                      # drift that config_version alone would miss)
    authorized_region: Any           # opaque region descriptor from _SupportBuilder (== deployment_snapshot.authorized_region)
    validation_result: dict          # raw output of _Validator.evaluate()
    passed: bool
    shadow_update_count: int
    issue_token: str                 # single-use token minted by evaluate() and held by the model;
                                      # promote() rejects an unrecognized or already-consumed token --
                                      # closes the forged-proposal attack surface (copying fingerprints
                                      # into a hand-built proposal isn't enough without this)
