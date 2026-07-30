"""Shader assembly, compilation, and diagnostics you can actually act on.

Two things live here, and the split is the point.

**Source assembly is pure.** :func:`assemble` resolves ``#include`` directives
into one flat string and records where every resulting line came from. GLSL has
no include mechanism of its own, and a raymarcher needs one: the SDF evaluation
is shared between the primary march, the normal estimate, and eventually a
shadow pass, and copy-pasting it into each is how the viewport ends up
disagreeing with itself. Because assembly touches no GL, it is unit-tested
headlessly, which matters because it is the part most likely to be wrong.

**Failure has to be readable.** ``glGetShaderInfoLog`` reports a line number
against the string the driver was handed — which, once includes are spliced in,
is not a line number in any file the author has open. So a compile failure
raises with the driver log *and* the assembled source, numbered, with the
failing lines marked and annotated with the file and line they originally came
from. A raymarch shader is several hundred lines; "compile failed" is not a
diagnosis.

PyOpenGL is used rather than ``QOpenGLShaderProgram``, which the refactor plan
sketched. Two reasons: the Qt wrapper reformats info logs and drops the raw
driver text this module exists to surface, and the primitive array from
:mod:`mesh_xml.render.sdf` is a numpy buffer that PyOpenGL uploads directly
while the Qt buffer classes want a round trip.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "SHADER_DIR",
    "AssembledSource",
    "ShaderCompileError",
    "ShaderError",
    "ShaderLinkError",
    "ShaderProgram",
    "ShaderSourceError",
    "assemble",
    "build_error_message",
    "error_lines",
    "file_resolver",
    "format_source",
]

SHADER_DIR = Path(__file__).parent / "shaders"

Resolver = Callable[[str], str]

_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+[<"]([^>"]+)[>"][ \t]*$')

# Every vendor invented its own info-log syntax. The number captured is always
# the line within the shader string; the leading integer is the source-string
# index, which is always 0 because a single string is uploaded.
_LOG_LINE_PATTERNS = (
    re.compile(r"^\s*\d+\((\d+)\)"),  # NVIDIA:  0(42) : error C1503: ...
    re.compile(r"^\s*\d+:(\d+)\(\d+\)"),  # Mesa:    0:42(15): error: ...
    re.compile(r"\b(?:ERROR|WARNING):\s*\d+:(\d+)"),  # AMD/ANGLE: ERROR: 0:42: ...
)


class ShaderError(RuntimeError):
    """Base for every failure in this module."""


class ShaderSourceError(ShaderError):
    """Assembly failed: a missing include, or a cycle. Raised without a GL context."""


class ShaderCompileError(ShaderError):
    """A stage failed to compile. The message carries the log and the source."""


class ShaderLinkError(ShaderError):
    """The stages compiled but would not link together."""


@dataclass(frozen=True)
class AssembledSource:
    """Flattened GLSL plus the provenance of each of its lines.

    ``origins[i]`` is the ``(unit, line)`` that produced line ``i + 1`` of
    :attr:`text`. Without it a driver complaint about line 312 of a shader
    assembled from four files is a scavenger hunt.
    """

    text: str
    origins: tuple[tuple[str, int], ...]
    name: str = "<source>"

    def origin(self, line: int) -> tuple[str, int] | None:
        """Where assembled line ``line`` (1-based) came from, or ``None``."""
        if 1 <= line <= len(self.origins):
            return self.origins[line - 1]
        return None

    def __str__(self) -> str:
        return self.text


def file_resolver(directory: Path | str = SHADER_DIR) -> Resolver:
    """Resolve include names against a directory.

    Names are taken as relative paths under ``directory`` and refused if they
    escape it, so a shader cannot be made to read arbitrary files by way of a
    ``../`` in an include.
    """
    root = Path(directory).resolve()

    def resolve(name: str) -> str:
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root):
            raise ShaderSourceError(f'#include "{name}" escapes {root}')
        return candidate.read_text(encoding="utf-8")

    return resolve


def assemble(
    source: str,
    name: str = "<source>",
    resolve: Resolver | Mapping[str, str] | None = None,
) -> AssembledSource:
    """Flatten ``#include`` directives into a single translation unit.

    Includes are resolved once per name. A second request for an already
    included unit becomes a comment rather than a second copy: sharing an SDF
    header into two shaders that both pull in a common helper is the normal
    case, and duplicated definitions fail to compile with an error that blames
    the innocent second copy.

    Line count is preserved — one input line always yields exactly one output
    line, except where an include expands — so :attr:`AssembledSource.origins`
    stays a straight index and nothing has to guess at offsets.
    """
    if resolve is None:
        resolver: Resolver = file_resolver()
    elif callable(resolve):
        resolver = resolve
    else:
        mapping = resolve

        def resolver(target: str) -> str:
            return mapping[target]

    lines: list[str] = []
    origins: list[tuple[str, int]] = []
    included: set[str] = set()

    def emit(text: str, unit: str, stack: tuple[str, ...]) -> None:
        for number, line in enumerate(text.splitlines(), start=1):
            match = _INCLUDE.match(line)
            if match is None:
                lines.append(line)
                origins.append((unit, number))
                continue

            target = match.group(1)
            if target in stack:
                chain = " -> ".join((*stack, target))
                raise ShaderSourceError(f"circular #include: {chain}")
            if target in included:
                lines.append(f'// #include "{target}" (already included)')
                origins.append((unit, number))
                continue

            try:
                nested = resolver(target)
            except ShaderSourceError:
                raise
            except (OSError, LookupError) as exc:
                raise ShaderSourceError(
                    f'{unit}:{number}: cannot resolve #include "{target}"'
                ) from exc

            included.add(target)
            emit(nested, target, (*stack, target))

    emit(source, name, (name,))
    return AssembledSource("\n".join(lines), tuple(origins), name)


def error_lines(log: str) -> tuple[int, ...]:
    """Line numbers a driver info log points at, sorted and deduplicated.

    Returns empty for a log in a format none of the patterns know, which is the
    signal to dump the whole source rather than guess at a window.
    """
    found: set[int] = set()
    for entry in log.splitlines():
        for pattern in _LOG_LINE_PATTERNS:
            match = pattern.search(entry)
            if match is not None:
                found.add(int(match.group(1)))
                break
    return tuple(sorted(found))


def format_source(
    source: AssembledSource | str,
    highlight: Iterable[int] = (),
    context: int | None = 6,
) -> str:
    """Number a shader source, marking ``highlight`` lines.

    With no highlights, or ``context=None``, the whole source is emitted. When
    the driver did name lines, only windows around them are shown — a
    three-hundred-line raymarch shader printed in full buries the one line that
    matters.
    """
    assembled = (
        source if isinstance(source, AssembledSource) else AssembledSource(source, ())
    )
    lines = assembled.text.splitlines()
    marked = {n for n in highlight if 1 <= n <= len(lines)}

    if marked and context is not None:
        wanted: set[int] = set()
        for line in marked:
            low = max(1, line - context)
            high = min(len(lines), line + context)
            wanted.update(range(low, high + 1))
        shown = sorted(wanted)
    else:
        shown = list(range(1, len(lines) + 1))

    width = len(str(len(lines))) if lines else 1
    out: list[str] = []
    previous = 0
    for number in shown:
        # Also fires before the first line of a truncated listing: a reader who
        # cannot see that lines were skipped will trust the numbering to be the
        # whole file.
        if number != previous + 1:
            out.append("      ...")
        prefix = ">>>" if number in marked else "   "
        entry = f"{prefix} {number:>{width}} | {lines[number - 1]}"
        origin = assembled.origin(number)
        if number in marked and origin is not None and origin[0] != assembled.name:
            entry = f"{entry}    <- {origin[0]}:{origin[1]}"
        out.append(entry)
        previous = number
    if shown and shown[-1] != len(lines):
        out.append("      ...")
    return "\n".join(out)


def build_error_message(stage: str, log: str, source: AssembledSource | str) -> str:
    """Compose the text a :class:`ShaderCompileError` carries.

    Kept separate from the GL call so it can be tested against captured logs
    from vendors this machine does not have.
    """
    name = source.name if isinstance(source, AssembledSource) else "<source>"
    lines = error_lines(log)
    body = format_source(source, lines)
    header = f"{stage} shader {name!r} failed to compile"
    return f"{header}\n\n--- driver log ---\n{log.strip()}\n\n--- assembled source ---\n{body}"


class ShaderProgram:
    """A linked GL program. Construction requires a current context.

    Uniform locations are cached, and :meth:`set_uniform` reports a missing
    uniform by returning ``False`` instead of raising. That is not laziness:
    GLSL compilers strip uniforms that do not affect the output, so a viewport
    that always sets ``u_time`` would crash the moment someone loads a static
    test shader. The caller that genuinely requires a uniform can check the
    return value.
    """

    def __init__(
        self,
        vertex: AssembledSource | str,
        fragment: AssembledSource | str,
        name: str = "program",
    ) -> None:
        from OpenGL.GL import (
            GL_FRAGMENT_SHADER,
            GL_LINK_STATUS,
            GL_VERTEX_SHADER,
            glAttachShader,
            glCreateProgram,
            glDeleteProgram,
            glDeleteShader,
            glDetachShader,
            glGetProgramInfoLog,
            glGetProgramiv,
            glLinkProgram,
        )

        self.name = name
        self._uniforms: dict[str, int] = {}
        self.id = 0

        vertex_shader = _compile(GL_VERTEX_SHADER, "vertex", vertex)
        try:
            fragment_shader = _compile(GL_FRAGMENT_SHADER, "fragment", fragment)
        except ShaderError:
            glDeleteShader(vertex_shader)
            raise

        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        linked = bool(glGetProgramiv(program, GL_LINK_STATUS))
        log = _decode(glGetProgramInfoLog(program))

        glDetachShader(program, vertex_shader)
        glDetachShader(program, fragment_shader)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        if not linked:
            glDeleteProgram(program)
            raise ShaderLinkError(
                f"program {name!r} failed to link\n\n"
                f"--- driver log ---\n{log.strip()}\n\n"
                f"--- vertex ---\n{format_source(vertex, context=None)}\n\n"
                f"--- fragment ---\n{format_source(fragment, context=None)}"
            )

        self.id = int(program)

    def use(self) -> None:
        from OpenGL.GL import glUseProgram

        glUseProgram(self.id)

    def uniform(self, name: str) -> int:
        """Cached uniform location, or ``-1`` when the compiler removed it."""
        location = self._uniforms.get(name)
        if location is None:
            from OpenGL.GL import glGetUniformLocation

            location = int(glGetUniformLocation(self.id, name))
            self._uniforms[name] = location
        return location

    def set_uniform(self, name: str, value: object) -> bool:
        """Upload a scalar, a 2/3/4-vector, or a 4x4 matrix. False if absent.

        Matrices are transposed on upload: this codebase writes them row-major
        in the ``M @ v`` sense (see :mod:`mesh_xml.render.gl.camera`) and GL
        reads column-major.
        """
        location = self.uniform(name)
        if location < 0:
            return False

        from OpenGL.GL import (
            GL_TRUE,
            glUniform1f,
            glUniform1i,
            glUniform2fv,
            glUniform3fv,
            glUniform4fv,
            glUniformMatrix4fv,
        )

        if isinstance(value, (bool, int)) and not isinstance(value, np.ndarray):
            glUniform1i(location, int(value))
            return True
        if isinstance(value, float):
            glUniform1f(location, float(value))
            return True

        array = np.ascontiguousarray(value, dtype=np.float32)
        if array.shape == ():
            glUniform1f(location, float(array))
        elif array.shape == (2,):
            glUniform2fv(location, 1, array)
        elif array.shape == (3,):
            glUniform3fv(location, 1, array)
        elif array.shape == (4,):
            glUniform4fv(location, 1, array)
        elif array.shape == (4, 4):
            glUniformMatrix4fv(location, 1, GL_TRUE, array)
        else:
            raise TypeError(f"uniform {name!r}: unsupported shape {array.shape}")
        return True

    def dispose(self) -> None:
        """Release the program. Needs the owning context current."""
        if not self.id:
            return
        from OpenGL.GL import glDeleteProgram

        glDeleteProgram(self.id)
        self.id = 0
        self._uniforms.clear()

    @classmethod
    def from_files(
        cls,
        vertex: str,
        fragment: str,
        name: str | None = None,
        resolve: Resolver | Mapping[str, str] | None = None,
    ) -> ShaderProgram:
        """Load, assemble, and build from two names under ``shaders/``."""
        resolver = file_resolver() if resolve is None else resolve
        loader = resolver if callable(resolver) else resolver.__getitem__
        return cls(
            assemble(loader(vertex), vertex, resolver),
            assemble(loader(fragment), fragment, resolver),
            name or f"{vertex}+{fragment}",
        )

    def __repr__(self) -> str:
        return f"<ShaderProgram {self.name!r} id={self.id}>"


def _compile(stage_enum: int, stage: str, source: AssembledSource | str) -> int:
    from OpenGL.GL import (
        GL_COMPILE_STATUS,
        glCompileShader,
        glCreateShader,
        glDeleteShader,
        glGetShaderInfoLog,
        glGetShaderiv,
        glShaderSource,
    )

    text = source.text if isinstance(source, AssembledSource) else source
    shader = glCreateShader(stage_enum)
    glShaderSource(shader, text)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = _decode(glGetShaderInfoLog(shader))
        glDeleteShader(shader)
        raise ShaderCompileError(build_error_message(stage, log, source))
    return int(shader)


def _decode(log: bytes | str | Sequence[object] | None) -> str:
    if log is None:
        return ""
    if isinstance(log, bytes):
        return log.decode("utf-8", errors="replace")
    return str(log)
