"""
Callback management mixin for LCAOWorkflow.

This module provides the CallbackMixin class that handles
callback registration and firing for workflow events.
"""

from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow import LCAOWorkflow


class CallbackMixin:
    """
    Mixin class providing callback management functionality.

    This mixin handles registration, unregistration, and firing
    of callbacks for various workflow events.
    """

    # Event names for callbacks
    EVENTS = [
        'before_scf',        # Called after before_scf()
        'after_iter',        # Called after each SCF iteration
        'before_after_scf',  # Called before after_scf() - main breakpoint
        'after_scf',         # Called after after_scf()
    ]

    def _init_callbacks(self) -> None:
        """Initialize the callback registry."""
        self._callbacks: Dict[str, List[Callable]] = {
            event: [] for event in self.EVENTS
        }

    def register_callback(self, event: str, callback: Callable[['LCAOWorkflow'], None]) -> None:
        """
        Register a callback function for a specific event.

        Parameters
        ----------
        event : str
            Event name. One of: 'before_scf', 'after_iter', 'before_after_scf', 'after_scf'
        callback : Callable[[LCAOWorkflow], None]
            Callback function that takes the workflow instance as argument.
            For 'after_iter', the callback receives (workflow, iter_num).

        Raises
        ------
        ValueError
            If event name is not recognized
        """
        if event not in self.EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Valid events: {self.EVENTS}"
            )
        self._callbacks[event].append(callback)

    def unregister_callback(self, event: str, callback: Callable) -> bool:
        """
        Unregister a callback function.

        Parameters
        ----------
        event : str
            Event name
        callback : Callable
            Callback function to remove

        Returns
        -------
        bool
            True if callback was found and removed, False otherwise
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            return True
        return False

    def clear_callbacks(self, event: Optional[str] = None) -> None:
        """
        Clear all callbacks for an event, or all events if event is None.

        Parameters
        ----------
        event : str, optional
            Event name. If None, clears all callbacks.
        """
        if event is None:
            for e in self.EVENTS:
                self._callbacks[e].clear()
        elif event in self._callbacks:
            self._callbacks[event].clear()

    def _fire_callbacks(self, event: str, *args, **kwargs) -> None:
        """Fire all callbacks for an event."""
        for callback in self._callbacks[event]:
            callback(self, *args, **kwargs)
