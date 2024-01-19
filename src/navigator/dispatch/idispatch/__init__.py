"""This is the dispatch interface module.

This module provides an interface and concrete classes for dispatching tasks using different methods, such as threading,
processing, and MPI.

Interface:
    - IDispatcher (abc.ABC): An abstract dispatcher interface.
    - ThreadedDispatcher (IDispatcher): A concrete threaded dispatcher class.
    - ProcessedDispatcher (IDispatcher): A concrete processed dispatcher class.
    - MPIDispatcher (IDispatcher): A concrete MPI dispatcher class.

Example Usage:
    >>> from dispatch_interface import ThreadedDispatcher
    >>> dispatcher = ThreadedDispatcher()
    >>> dispatcher.dispatch_tasks()

Note:
    To use the dispatchers, instantiate the respective classes and call the `dispatch_tasks` method.

See Also:
    - `IDispatcher`: The abstract base class for dispatchers.

References:
    - This module follows the design principles outlined in the Google Python Style Guide.
"""
