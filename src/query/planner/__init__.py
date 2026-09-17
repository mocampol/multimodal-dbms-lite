"""Physical plan construction."""

from .plan_builder import build_select_plan, execute_delete, execute_insert

__all__ = ["build_select_plan", "execute_delete", "execute_insert"]

