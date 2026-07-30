"""Core body model: a graph of masses connected by distance constraints.

This is the simulation-time representation, and it is deliberately *not* a
transform hierarchy. It mirrors Rain World's ``PhysicalObject``, which owns a
flat ``BodyChunk[]`` plus a flat ``BodyChunkConnection[]`` and has no notion of
a parent transform while the simulation is running.

Two consequences follow, and both are intentional:

* **Cycles are legal.** A hip girdle, a ribcage and a palm are all closed loops
  of chunks. The previous ``ArmatureManager`` rejected them outright.
* **Hierarchy is a serialization concern only.** When a tree is needed (to
  write XML, or to walk a chain for a controller) it is *derived* by
  :meth:`Body.spanning_forest`, which is total: every chunk lands in the forest
  exactly once and non-tree edges are reported separately rather than dropped.

Storage is struct-of-arrays so the solver vectorizes over numpy rather than
looping in Python.

.. warning::
   Structural edits invalidate three things, and all three bite silently:

   * **Views.** The array properties are live views into the backing storage.
     Adding chunks or links may reallocate, so re-read the property after any
     structural edit rather than caching it across one.
   * **Slots.** ``link_a`` and ``link_b`` store *slot indices*, not ids, because
     that is what lets the solver gather without a lookup table.
     :meth:`Body.remove_chunk` relocates the last chunk into the freed slot and
     rewrites those arrays in place, so any slot index you cached is stale.
   * **The tail.** Growth uses ``np.resize``, which *tiles* the existing data
     into the new tail instead of zeroing it. Everything past ``n_chunks`` /
     ``n_links`` is a plausible-looking duplicate of live data, never zeros.
     Only ever read through the properties, which slice to the live prefix.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy as np

__all__ = ["LinkType", "Chunk", "Link", "Body"]

_INITIAL_CAPACITY = 16


class LinkType(IntEnum):
    """How a distance constraint responds to being violated.

    This is Rain World's ``BodyChunkConnection.Type``. The distinction is what
    separates a spine from a rope: ``NORMAL`` holds a rest length in both
    directions, ``PULL`` lets chunks drift together freely but not apart, and
    ``PUSH`` keeps chunks from interpenetrating while letting them separate.
    """

    NORMAL = 0
    PULL = 1
    PUSH = 2


@dataclass(frozen=True)
class Chunk:
    """A read-only snapshot of one chunk.

    Construction and serialization traffic in these records; the solver works
    on :class:`Body`'s arrays directly. Mutating a ``Chunk`` does not affect the
    body it came from.

    ``meta`` is an opaque per-chunk dict that the core never reads. It exists so
    importers can round-trip fields the simulation has no concept of — a v1
    file's ``mesh type``, ``height`` and ``material``, for instance. Resist
    promoting anything out of it into a real field without a simulation reason:
    a v1 capsule's ``height`` describes a *segment*, which in this model is a
    link's rest length, not a property of either chunk it joins.
    """

    id: int
    name: str
    pos: tuple[float, ...]
    radius: float
    inv_mass: float
    meta: dict

    @property
    def pinned(self) -> bool:
        return self.inv_mass == 0.0

    @property
    def mass(self) -> float:
        return float("inf") if self.inv_mass == 0.0 else 1.0 / self.inv_mass


@dataclass(frozen=True)
class Link:
    """A read-only snapshot of one distance constraint, keyed by chunk id."""

    a: int
    b: int
    rest: float
    type: LinkType
    elasticity: float


class Body:
    """A graph of chunks joined by distance constraints.

    Chunk identity is a stable integer id, never a name and never an array
    slot. Names are an editable field and may repeat; ids may not. Slots are an
    implementation detail that shifts as chunks are removed.
    """

    def __init__(self, dim: int = 3, name: str = "body") -> None:
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")
        self.dim = dim
        self.name = name

        self._n = 0
        self._cap = _INITIAL_CAPACITY
        self._pos = np.zeros((self._cap, dim), dtype=np.float32)
        self._prev_pos = np.zeros((self._cap, dim), dtype=np.float32)
        self._radius = np.zeros(self._cap, dtype=np.float32)
        self._inv_mass = np.zeros(self._cap, dtype=np.float32)
        self._ids = np.zeros(self._cap, dtype=np.int32)
        self._names: list[str] = []
        self._meta: list[dict] = []

        self._n_links = 0
        self._link_cap = _INITIAL_CAPACITY
        self._link_a = np.zeros(self._link_cap, dtype=np.int32)
        self._link_b = np.zeros(self._link_cap, dtype=np.int32)
        self._link_rest = np.zeros(self._link_cap, dtype=np.float32)
        self._link_type = np.zeros(self._link_cap, dtype=np.uint8)
        self._link_elasticity = np.ones(self._link_cap, dtype=np.float32)

        self._slot_of: dict[int, int] = {}
        self._link_keys: set[tuple[int, int]] = set()
        self._next_id = 0

    # ------------------------------------------------------------------
    # array views — the solver's interface
    # ------------------------------------------------------------------

    @property
    def n_chunks(self) -> int:
        return self._n

    @property
    def n_links(self) -> int:
        return self._n_links

    @property
    def pos(self) -> np.ndarray:
        """``(N, dim) float32`` current world positions. Writable."""
        return self._pos[: self._n]

    @property
    def prev_pos(self) -> np.ndarray:
        """``(N, dim) float32`` positions at the previous tick. Writable."""
        return self._prev_pos[: self._n]

    @property
    def radius(self) -> np.ndarray:
        """``(N,) float32`` chunk radii."""
        return self._radius[: self._n]

    @property
    def inv_mass(self) -> np.ndarray:
        """``(N,) float32`` inverse masses; ``0.0`` means pinned."""
        return self._inv_mass[: self._n]

    @property
    def link_a(self) -> np.ndarray:
        """``(M,) int32`` slot index of each link's first chunk."""
        return self._link_a[: self._n_links]

    @property
    def link_b(self) -> np.ndarray:
        """``(M,) int32`` slot index of each link's second chunk."""
        return self._link_b[: self._n_links]

    @property
    def link_rest(self) -> np.ndarray:
        """``(M,) float32`` rest distance of each link."""
        return self._link_rest[: self._n_links]

    @property
    def link_type(self) -> np.ndarray:
        """``(M,) uint8`` :class:`LinkType` of each link."""
        return self._link_type[: self._n_links]

    @property
    def link_elasticity(self) -> np.ndarray:
        """``(M,) float32`` correction scale in ``[0, 1]`` per solver iteration."""
        return self._link_elasticity[: self._n_links]

    # ------------------------------------------------------------------
    # chunks
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        name: str,
        pos,
        radius: float = 0.5,
        mass: float = 1.0,
        pinned: bool = False,
        id: int | None = None,
        meta: dict | None = None,
    ) -> int:
        """Append a chunk and return its stable id.

        ``pos`` is world space. Names are not checked for uniqueness — that was
        the old model's silent-overwrite bug, and it is fixed here by making the
        returned id the only identity that matters.

        ``id`` forces a specific id instead of allocating the next one, which
        deserializers need: a rig that has had chunks removed has gaps in its id
        sequence, and those ids are what its links reference. Supplying one that
        is already live is an error. Without this, a loader has no choice but to
        reach into private state to restore identity.
        """
        pos = np.asarray(pos, dtype=np.float32)
        if pos.shape != (self.dim,):
            raise ValueError(f"pos must have shape ({self.dim},), got {pos.shape}")
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        if not pinned and mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}")

        if id is None:
            chunk_id = self._next_id
        else:
            if id in self._slot_of:
                raise ValueError(f"chunk id {id} is already in use")
            if id < 0:
                raise ValueError(f"chunk id must be non-negative, got {id}")
            chunk_id = id
        self._next_id = max(self._next_id, chunk_id + 1)

        if self._n == self._cap:
            self._grow_chunks()

        slot = self._n
        self._pos[slot] = pos
        self._prev_pos[slot] = pos
        self._radius[slot] = radius
        self._inv_mass[slot] = 0.0 if pinned else 1.0 / mass
        self._ids[slot] = chunk_id
        self._names.append(name)
        self._meta.append(dict(meta) if meta else {})
        self._slot_of[chunk_id] = slot
        self._n += 1
        return chunk_id

    def remove_chunk(self, chunk_id: int) -> None:
        """Remove a chunk and every link that touches it.

        Implemented as swap-with-last, so slots are not stable across this call
        but ids are.
        """
        slot = self._require_slot(chunk_id)

        for link in list(self.links()):
            if link.a == chunk_id or link.b == chunk_id:
                self.remove_link(link.a, link.b)

        last = self._n - 1
        if slot != last:
            self._pos[slot] = self._pos[last]
            self._prev_pos[slot] = self._prev_pos[last]
            self._radius[slot] = self._radius[last]
            self._inv_mass[slot] = self._inv_mass[last]
            self._ids[slot] = self._ids[last]
            self._names[slot] = self._names[last]
            self._meta[slot] = self._meta[last]
            self._slot_of[int(self._ids[last])] = slot
            # links store slots, so the moved chunk's references follow it
            np.putmask(self.link_a, self.link_a == last, slot)
            np.putmask(self.link_b, self.link_b == last, slot)

        self._names.pop()
        self._meta.pop()
        del self._slot_of[chunk_id]
        self._n -= 1

    def has_chunk(self, chunk_id: int) -> bool:
        return chunk_id in self._slot_of

    def slot(self, chunk_id: int) -> int:
        """Array index of ``chunk_id``. Invalidated by :meth:`remove_chunk`."""
        return self._require_slot(chunk_id)

    def id_at(self, slot: int) -> int:
        """Stable id of the chunk currently occupying ``slot``."""
        if not 0 <= slot < self._n:
            raise IndexError(f"slot {slot} out of range (n={self._n})")
        return int(self._ids[slot])

    def chunk(self, chunk_id: int) -> Chunk:
        slot = self._require_slot(chunk_id)
        return Chunk(
            id=chunk_id,
            name=self._names[slot],
            pos=tuple(float(v) for v in self._pos[slot]),
            radius=float(self._radius[slot]),
            inv_mass=float(self._inv_mass[slot]),
            meta=dict(self._meta[slot]),
        )

    def meta_of(self, chunk_id: int) -> dict:
        """The live opaque metadata dict for a chunk. Mutating it is intended."""
        return self._meta[self._require_slot(chunk_id)]

    def chunks(self) -> Iterator[Chunk]:
        for slot in range(self._n):
            yield self.chunk(int(self._ids[slot]))

    def chunk_ids(self) -> list[int]:
        return [int(self._ids[s]) for s in range(self._n)]

    def name_of(self, chunk_id: int) -> str:
        return self._names[self._require_slot(chunk_id)]

    def rename(self, chunk_id: int, name: str) -> None:
        """Rename a chunk. Cheap and total, because identity is the id."""
        self._names[self._require_slot(chunk_id)] = name

    def set_pos(self, chunk_id: int, pos) -> None:
        """Teleport a chunk, zeroing its implicit Verlet velocity."""
        slot = self._require_slot(chunk_id)
        self._pos[slot] = pos
        self._prev_pos[slot] = pos

    def set_radius(self, chunk_id: int, radius: float) -> None:
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        self._radius[self._require_slot(chunk_id)] = radius

    def set_pinned(self, chunk_id: int, pinned: bool, mass: float = 1.0) -> None:
        """Pin or unpin a chunk.

        The mass check is not decoration. A negative inverse mass inverts the
        sign of the solver's correction, so the chunk accelerates *away* from
        every constraint acting on it and the body diverges within a few ticks.
        """
        slot = self._require_slot(chunk_id)
        if not pinned and mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}")
        self._inv_mass[slot] = 0.0 if pinned else 1.0 / mass

    # ------------------------------------------------------------------
    # links
    # ------------------------------------------------------------------

    def add_link(
        self,
        id_a: int,
        id_b: int,
        rest: float | None = None,
        type: LinkType = LinkType.NORMAL,
        elasticity: float = 1.0,
    ) -> bool:
        """Connect two chunks. Returns ``False`` if the link already exists.

        Links are undirected: ``add_link(a, b)`` and ``add_link(b, a)`` name the
        same constraint, and the second call is a no-op. The old model treated
        them as distinct edges, which made a chunk its own ancestor and silently
        emptied the entire export.

        ``rest`` defaults to the chunks' current separation, so authoring a rig
        in the viewport captures its own rest pose.
        """
        slot_a = self._require_slot(id_a)
        slot_b = self._require_slot(id_b)
        if id_a == id_b:
            raise ValueError(f"cannot link chunk {id_a} to itself")

        key = (min(id_a, id_b), max(id_a, id_b))
        if key in self._link_keys:
            return False

        if rest is None:
            rest = float(np.linalg.norm(self._pos[slot_a] - self._pos[slot_b]))
        if rest < 0.0:
            raise ValueError(f"rest must be non-negative, got {rest}")
        if not 0.0 <= elasticity <= 1.0:
            raise ValueError(f"elasticity must be in [0, 1], got {elasticity}")

        if self._n_links == self._link_cap:
            self._grow_links()

        i = self._n_links
        self._link_a[i] = slot_a
        self._link_b[i] = slot_b
        self._link_rest[i] = rest
        self._link_type[i] = int(type)
        self._link_elasticity[i] = elasticity
        self._link_keys.add(key)
        self._n_links += 1
        return True

    def remove_link(self, id_a: int, id_b: int) -> bool:
        key = (min(id_a, id_b), max(id_a, id_b))
        if key not in self._link_keys:
            return False

        slot_a = self._slot_of[id_a]
        slot_b = self._slot_of[id_b]
        for i in range(self._n_links):
            pair = (int(self._link_a[i]), int(self._link_b[i]))
            if pair == (slot_a, slot_b) or pair == (slot_b, slot_a):
                last = self._n_links - 1
                if i != last:
                    self._link_a[i] = self._link_a[last]
                    self._link_b[i] = self._link_b[last]
                    self._link_rest[i] = self._link_rest[last]
                    self._link_type[i] = self._link_type[last]
                    self._link_elasticity[i] = self._link_elasticity[last]
                self._n_links -= 1
                self._link_keys.discard(key)
                return True
        return False

    def has_link(self, id_a: int, id_b: int) -> bool:
        return (min(id_a, id_b), max(id_a, id_b)) in self._link_keys

    def set_link(
        self,
        id_a: int,
        id_b: int,
        rest: float | None = None,
        type: LinkType | None = None,
        elasticity: float | None = None,
    ) -> None:
        """Edit an existing link's fields, validating as :meth:`add_link` does.

        Without this the only way to retune a link is to write into the
        ``link_rest`` / ``link_type`` / ``link_elasticity`` views, which skips
        every bounds check — an out-of-range elasticity reaches the solver as a
        per-iteration gain above one, which overshoots and then diverges.
        """
        index = self._link_index(id_a, id_b)
        if index is None:
            raise KeyError(f"no link between {id_a} and {id_b}")
        if rest is not None:
            if rest < 0.0:
                raise ValueError(f"rest must be non-negative, got {rest}")
            self._link_rest[index] = rest
        if type is not None:
            self._link_type[index] = int(LinkType(type))
        if elasticity is not None:
            if not 0.0 <= elasticity <= 1.0:
                raise ValueError(f"elasticity must be in [0, 1], got {elasticity}")
            self._link_elasticity[index] = elasticity

    def link(self, id_a: int, id_b: int) -> Link:
        index = self._link_index(id_a, id_b)
        if index is None:
            raise KeyError(f"no link between {id_a} and {id_b}")
        return Link(
            a=int(self._ids[self._link_a[index]]),
            b=int(self._ids[self._link_b[index]]),
            rest=float(self._link_rest[index]),
            type=LinkType(int(self._link_type[index])),
            elasticity=float(self._link_elasticity[index]),
        )

    def _link_index(self, id_a: int, id_b: int) -> int | None:
        if (min(id_a, id_b), max(id_a, id_b)) not in self._link_keys:
            return None
        slot_a = self._slot_of[id_a]
        slot_b = self._slot_of[id_b]
        for i in range(self._n_links):
            pair = (int(self._link_a[i]), int(self._link_b[i]))
            if pair == (slot_a, slot_b) or pair == (slot_b, slot_a):
                return i
        return None

    def links(self) -> Iterator[Link]:
        for i in range(self._n_links):
            yield Link(
                a=int(self._ids[self._link_a[i]]),
                b=int(self._ids[self._link_b[i]]),
                rest=float(self._link_rest[i]),
                type=LinkType(int(self._link_type[i])),
                elasticity=float(self._link_elasticity[i]),
            )

    def neighbors(self, chunk_id: int) -> list[int]:
        self._require_slot(chunk_id)
        out = []
        for link in self.links():
            if link.a == chunk_id:
                out.append(link.b)
            elif link.b == chunk_id:
                out.append(link.a)
        return out

    # ------------------------------------------------------------------
    # derived hierarchy — serialization and chain walking only
    # ------------------------------------------------------------------

    def spanning_forest(
        self, roots: list[int] | None = None
    ) -> tuple[dict[int, list[int]], list[int], list[tuple[int, int]]]:
        """Derive a tree view of the graph by breadth-first search.

        Returns ``(children, root_ids, extra_edges)`` where ``children`` maps a
        chunk id to its child ids, ``root_ids`` lists the forest roots in visit
        order, and ``extra_edges`` holds every link the tree could not express.

        This is total by construction: every chunk appears exactly once, so a
        cyclic or multiply-parented graph degrades to a tree plus a list of
        leftover edges instead of losing chunks. Callers that need lossless
        round-tripping must write ``extra_edges`` out alongside the tree.
        """
        adjacency: dict[int, list[int]] = {cid: [] for cid in self.chunk_ids()}
        for link in self.links():
            adjacency[link.a].append(link.b)
            adjacency[link.b].append(link.a)

        children: dict[int, list[int]] = {cid: [] for cid in adjacency}
        parent: dict[int, int | None] = {}
        root_ids: list[int] = []
        extra: list[tuple[int, int]] = []
        seen: set[int] = set()

        preferred = list(roots or [])
        order = preferred + [c for c in self.chunk_ids() if c not in preferred]

        for start in order:
            if start in seen:
                continue
            seen.add(start)
            parent[start] = None
            root_ids.append(start)
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        parent[neighbor] = node
                        children[node].append(neighbor)
                        queue.append(neighbor)
                    elif parent.get(node) != neighbor:
                        edge = (min(node, neighbor), max(node, neighbor))
                        if edge not in extra and neighbor not in children[node]:
                            extra.append(edge)

        return children, root_ids, extra

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_slot(self, chunk_id: int) -> int:
        try:
            return self._slot_of[chunk_id]
        except KeyError:
            raise KeyError(f"no chunk with id {chunk_id}") from None

    def _grow_chunks(self) -> None:
        self._cap *= 2
        self._pos = np.resize(self._pos, (self._cap, self.dim))
        self._prev_pos = np.resize(self._prev_pos, (self._cap, self.dim))
        self._radius = np.resize(self._radius, self._cap)
        self._inv_mass = np.resize(self._inv_mass, self._cap)
        self._ids = np.resize(self._ids, self._cap)

    def _grow_links(self) -> None:
        self._link_cap *= 2
        self._link_a = np.resize(self._link_a, self._link_cap)
        self._link_b = np.resize(self._link_b, self._link_cap)
        self._link_rest = np.resize(self._link_rest, self._link_cap)
        self._link_type = np.resize(self._link_type, self._link_cap)
        self._link_elasticity = np.resize(self._link_elasticity, self._link_cap)

    def __repr__(self) -> str:
        return (
            f"<Body {self.name!r} dim={self.dim} "
            f"chunks={self._n} links={self._n_links}>"
        )
