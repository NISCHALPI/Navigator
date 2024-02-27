"""EpochCollection Module.

Provides a class for managing a collection of Epoch objects representing points in time related to satellite tracking.

Classes:
    - EpochCollection: Manages a collection of Epoch objects.

Functions:
    - continuous_track: Returns a new collection of Epoch objects that contains a segment where satellites are continuously tracked.
    - append: Appends a new Epoch object to the collection.
    - pop: Pops an Epoch object from the collection.
    - epochs: Returns the list of Epoch objects.
    - start_time: Returns the start time of the epochs.
    - end_time: Returns the end time of the epochs.

Exceptions:
    - ValueError: If epochs is not a list of Epoch objects.

    
Author: Nischal Bhattarai
"""

from typing import Iterator, List

from .epoch import Epoch

__all__ = ["EpochCollection"]


class EpochCollection:
    """A class representing a collection of Epoch objects for satellite tracking.

    This class provides a flexible and efficient way to manage a sequence of Epoch objects,
    each capturing distinct points in time related to satellite tracking.

    Args:
        epochs (List[Epoch]): A list of Epoch objects.

    Raises:
        ValueError: If epochs is not a list of Epoch objects.

    Attributes:
        _epochs (List[Epoch]): The internal list of Epoch objects, sorted by timestamp.

    Methods:
        continuous_track(start: int = 0, end: int = -1) -> EpochCollection:
            Returns a new collection of Epoch objects containing a segment of continuous satellite tracking.

        append(epoch: Epoch) -> None:
            Appends a new Epoch object to the collection without re-sorting the list.

        pop(index: int) -> Epoch:
            Pops an Epoch object from the collection based on the provided index.

        epochs() -> List[Epoch]:
            Returns the list of Epoch objects.

        start_time() -> str:
            Returns the start time of the epochs.

        end_time() -> str:
            Returns the end time of the epochs.

    Properties:
        epochs (List[Epoch]): The list of Epoch objects.
        start_time (str): The start time of the epochs.
        end_time (str): The end time of the epochs.
    """

    # Strict tracking mode i.e each epoch must track the exact same satellites (no more, no less)
    STRICT = "strict"
    # Visibility tracking mode i.e each epoch should have at least the same possibly more set of satellites as the epoch preceding it
    # The next tracking segment starts when the current epoch does not have the same satellites as the start epoch
    VISIBILITY = "visibility"
    # Same as visibility mode but current epoch must have the same satellites as the previous epoch not the start epoch
    # Resumes tracking the same sv until the current epoch does not have the same satellites as the previous epoch
    STRICT_VISIBILITY = "strict_visibility"  # Tracks the sv until one of the initial sv
    # New sv tracking mode i.e same as visibility mode but tracking is resumed when a new satellite is observed
    # Not when the current epoch does not have the same satellites as the start epoch
    EXTENDED_VISIBILITY = "extended_visibility"

    def __init__(self, epochs: List[Epoch], profile: dict[str] = Epoch.DUAL) -> None:
        """Initializes the class with a list of Epoch objects.

        This class provides a flexible and efficient way to manage a sequence of Epoch objects,
        each capturing distinct points in time related to satellite tracking.


        Args:
            epochs (List[Epoch]): A list of Epoch objects.
            profile (dict, optional): The profile of the collection of epochs. Defaults to Epoch.DUAL.

        Raises:
            ValueError: If epochs is not a list of Epoch objects.
        """
        # Check if the epochs is a list of Epoch objects
        if not all(isinstance(epoch, Epoch) for epoch in epochs):
            raise ValueError("epochs must be a list of Epoch objects")

        # Set the epochs
        self._epochs = sorted(epochs, key=lambda x: x.timestamp)

        # Set a profile
        self.profile = profile

    def _criterion_track(
        self, start: int = 0, end: int = -1, mode: str = "visibility"
    ) -> dict:
        """Tracks the indices of the epochs that satisfy the tracking criterion.

        Args:
            start (int, optional): The starting index from which the criterion is checked. Defaults to 0.
            end (int, optional): The ending index until which the criterion is checked. Defaults to -1.
            mode (str, optional): The tracking mode. Defaults to "visibility".

        Returns:
            dict: A dictionary containing the indices of the epochs that satisfy the tracking criterion.
        """
        # Get the start epoch
        initial_sv_to_track = set(self._epochs[start].common_sv)

        # Return dictionary of metadata
        metadata = {
            "start": start,  # Start index of the tracking segment
            "upto": None,  # End index upto with the criterion is valid
            "sv_track": list(initial_sv_to_track),  # The tracked sv
            "resume_from": None,  # The index from which the next tracking segment starts. Default is upto for most modes
        }
        # Track the valid index
        valid = start

        for i, epoch in enumerate(self._epochs[start:end]):
            # Ensure that all the satellites in the start epoch are tracked in the current epoch
            # If strict mode is enabled, then the current epoch must have the same satellites as the start epoch
            if mode == EpochCollection.STRICT:
                if set(epoch.common_sv) != initial_sv_to_track:
                    break
            # If visibility mode is enabled, then the current epoch must have at least the same satellites as the start epoch
            elif mode == EpochCollection.VISIBILITY:
                if not initial_sv_to_track.issubset(epoch.common_sv):
                    break
            # If strict visibility mode is enabled, then the current epoch must have the same satellites as the previous epoch
            elif mode == EpochCollection.STRICT_VISIBILITY:
                if not initial_sv_to_track.issubset(epoch.common_sv):
                    break
                # Update the initial sv to track to current epoch's common sv
                initial_sv_to_track = set(epoch.common_sv)
            # If extended visibility mode is enabled, then the current epoch must have at least the same satellites as the previous epoch
            elif mode == EpochCollection.EXTENDED_VISIBILITY:
                if not initial_sv_to_track.issubset(epoch.common_sv):
                    break
                # Check if there is a new satellite observed
                if (
                    not initial_sv_to_track == set(epoch.common_sv)
                    and metadata["resume_from"] is None
                ):
                    # Add first epoch where new sv is observed
                    metadata["resume_from"] = (
                        start + i
                    )  # This is where the next tracking segment starts
            # Update the valid index
            valid = start + i

        # Update the metadata
        metadata["upto"] = valid
        # If resume_from is not updated, then set it to the end of the continuous epochs
        if metadata["resume_from"] is None:
            metadata["resume_from"] = valid + 1

        return metadata

    def track(self, mode: str = "visiblity") -> list["EpochCollection"]:
        """Returns a list of EpochCollection objects, each representing a segment with consistently tracked satellites.

        Satellite tracking is deemed continuous when the same set of satellites is observed in each epoch. The tracking can be conducted in either strict or non-strict mode.
        In strict mode, each epoch must precisely track the same satellites, with no additional or missing satellites. In non-strict mode, each epoch should have at least the
        same possibly more set of satellites as the epoch preceding it.

        This function organizes continuous segments of epochs into multiple EpochCollection objects, each encapsulating a sequence of consistently tracked satellites.

        Available tracking modes:
        - strict: Each epoch must track the exact same satellites (no more, no less)
        - visibility: Each epoch should have at least the same possibly more set of satellites as the initial epoch
        - strict_visibility: Each epoch should have at least the same possibly more set of satellites as the previous epoch
        - extended_visibility: Each epoch should have same possibly more set of satellites as the previous epoch but next tracking segment starts from where a new sv is observed

        Note:
            The extended_visibility mode might contain duplicate epochs since it doesn't partition the epochs into distinct segments.

        Args:
            mode (str, optional): The tracking mode. Defaults to "visibility". [strict, visibility, new_sv]

        Returns:
            list[EpochCollection]: A list of EpochCollection objects, each representing a segment with consistently tracked satellites.
        """
        # Initialize the list of continuous tracks
        continuous_tracks = []
        start = 0

        # Track the common sv for the rest of the epochs
        while start < len(self._epochs):
            # Track the common sv for the rest of the epochs
            metadata = self._criterion_track(start, -1, mode)
            # Grab the index of the current segment
            collection = EpochCollection(
                self._epochs[metadata["start"] : metadata["upto"] + 1],
                profile=self.profile,
            )
            # Append the collection to the list of continuous tracks
            continuous_tracks.append(collection)

            # Update the start index for the next segment
            start = metadata["resume_from"]

        # Sort the continuous tracks in descending order
        return sorted(continuous_tracks, reverse=True)

    def append(self, epoch: Epoch) -> None:
        """Appends a new Epoch object to the collection.

        Add the new epoch to where it belongs in the list of epochs without sorting the list again.

        Args:
            epoch (Epoch): The Epoch object to append.

        Raises:
            ValueError: If epoch is not an Epoch object.

        Add Docs Here!
        """
        if not isinstance(epoch, Epoch):
            raise ValueError("epoch must be an Epoch object")

        # Linear search to find the position to insert the new epoch
        for i, e in enumerate(self._epochs):
            if epoch.timestamp < e.timestamp:
                self._epochs.insert(i, epoch)
                return
        # Append the epoch at the end of the list
        self._epochs.append(epoch)

    def pop(self, index: int) -> Epoch:
        """Pops an Epoch object from the collection.

        Args:
            index (int): The index from which to pop the Epoch object.

        Returns:
            Epoch: The popped Epoch object.
        """
        return self._epochs.pop(index)

    @property
    def profile(self) -> str:
        """Returns the profile of the collection."""
        return self._profile

    @profile.setter
    def profile(self, value: dict) -> None:
        """Set the profile of the epoch.

        Args:
            value (dict): The value to set.

        """
        # Check if the value contains the necessary keys
        if not all(key in value for key in Epoch.MANDATORY_PROFILE_KEYS):
            raise ValueError(
                f"Profile must contain the following keys: {Epoch.MANDATORY_PROFILE_KEYS}. Got {value.keys()} instead."
            )
        self._profile = value

    @property
    def epochs(self) -> List[Epoch]:
        """Returns the list of Epoch objects."""
        return self._epochs

    @property
    def start_time(self) -> str:
        """Returns the start time of the epochs."""
        return self._epochs[0].timestamp if len(self._epochs) > 0 else "None"

    @property
    def end_time(self) -> str:
        """Returns the end time of the epochs."""
        return self._epochs[-1].timestamp if len(self._epochs) > 0 else "None"

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        return f"EpochCollection(num_epochs={len(self._epochs)}, start_time={self.start_time}, end_time={self.end_time})"

    def __len__(self) -> int:
        """Returns the number of Epoch objects in the collection."""
        return len(self._epochs)

    def __iter__(self) -> Iterator[Epoch]:
        """Returns an iterator over the Epoch objects."""
        # Update the profile of the epochs with the profile of the collection
        for epoch in self._epochs:
            epoch.profile = self._profile
            yield epoch

    def __getitem__(self, index: int) -> Epoch:
        """Returns the Epoch object at the given index."""
        # Update the profile of the epoch with the profile of the collection
        self._epochs[index].profile = self._profile
        return self._epochs[index]

    def __setitem__(self, index: int, value: Epoch) -> None:
        """Sets the Epoch object at the given index.

        Args:
            index (int): The index to set the Epoch object.
            value (Epoch): The Epoch object to set.

        Raises:
            ValueError: If value is not an Epoch object.
        """
        if not isinstance(value, Epoch):
            raise ValueError("value must be an Epoch object")
        self._epochs[index] = value

    def __delitem__(self, index: int) -> None:
        """Deletes the Epoch object at the given index.

        Args:
            index (int): The index to delete the Epoch object.
        """
        del self._epochs[index]

    def __lt__(self, other: "EpochCollection") -> bool:
        """Returns True if the current collection is less than the other collection.

        Args:
            other (EpochCollection): The other collection to compare.

        Returns:
            bool: True if the current collection is less than the other collection.
        """
        return len(self._epochs) < len(other._epochs)

    def __gt__(self, other: "EpochCollection") -> bool:
        """Returns True if the current collection is greater than the other collection.

        Args:
            other (EpochCollection): The other collection to compare.

        Returns:
            bool: True if the current collection is greater than the other collection.
        """
        return len(self._epochs) > len(other._epochs)

    def __eq__(self, other: "EpochCollection") -> bool:
        """Returns True if the current collection is equal to the other collection.

        Args:
            other (EpochCollection): The other collection to compare.

        Returns:
            bool: True if the current collection is equal to the other collection.
        """
        return len(self._epochs) == len(other._epochs)

    def __add__(self, other: "EpochCollection") -> "EpochCollection":
        """Adds two EpochCollection objects together.

        Args:
            other (EpochCollection): The other EpochCollection object to add.

        Returns:
            EpochCollection: The new EpochCollection object containing the concatenated epochs.
        """
        return EpochCollection(self._epochs + other._epochs, profile=self.profile)

    def __iadd__(self, other: "EpochCollection") -> "EpochCollection":
        """Adds another EpochCollection object to the current object.

        Args:
            other (EpochCollection): The other EpochCollection object to add.

        Returns:
            EpochCollection: The current EpochCollection object containing the concatenated epochs.
        """
        self._epochs += other._epochs
        return self

    def __sub__(self, other: "EpochCollection") -> "EpochCollection":
        """Subtracts two EpochCollection objects.

        Args:
            other (EpochCollection): The other EpochCollection object to subtract.

        Returns:
            EpochCollection: The new EpochCollection object containing the difference of epochs.
        """
        return EpochCollection(
            [epoch for epoch in self._epochs if epoch not in other._epochs],
            profile=self.profile,
        )

    def __isub__(self, other: "EpochCollection") -> "EpochCollection":
        """Subtracts another EpochCollection object from the current object.

        Args:
            other (EpochCollection): The other EpochCollection object to subtract.

        Returns:
            EpochCollection: The current EpochCollection object containing the difference of epochs.
        """
        self._epochs = [epoch for epoch in self._epochs if epoch not in other._epochs]
        return self

    def __and__(self, other: "EpochCollection") -> "EpochCollection":
        """Returns the intersection of two EpochCollection objects.

        Args:
            other (EpochCollection): The other EpochCollection object to intersect.

        Returns:
            EpochCollection: The new EpochCollection object containing the intersection of epochs.
        """
        return EpochCollection(
            [epoch for epoch in self._epochs if epoch in other._epochs],
            profile=self.profile,
        )

    def __iand__(self, other: "EpochCollection") -> "EpochCollection":
        """Returns the intersection of the current object with another object.

        Args:
            other (EpochCollection): The other EpochCollection object to intersect.

        Returns:
            EpochCollection: The current EpochCollection object containing the intersection of epochs.
        """
        self._epochs = [epoch for epoch in self._epochs if epoch in other._epochs]
        return self
