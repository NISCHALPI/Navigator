"""This is the dispatch module.

This module provides classes for dispatching multithreaded and multiprocessed tasks for triangulation, parsing, and
heavy computation.

Classes:
    - AbstractDispatcher (abc.ABC): An abstract dispatcher class.
    - Dispatcher (AbstractDispatcher): A concrete dispatcher class.

State:
    Currently, the dispatcher is not implemented.

Example Usage:
    >>> from navigator.dispacth import Dispatcher, ThreadInterface
    >>> dispatcher = Dispatcher(interface=ThreadInterface(thread_count=4)))
    >>> dispatcher.dispatch_tasks()

Note:
    To use the dispatcher, instantiate the `Dispatcher` class and call the `dispatch_tasks` method.

See Also:
    - `AbstractDispatcher`: The abstract base class for dispatchers.
"""
