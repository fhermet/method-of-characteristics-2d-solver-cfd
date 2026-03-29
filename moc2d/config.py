"""Configuration dataclasses for the MOC 2D solver."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GasProperties:
    """Calorically perfect gas properties."""
    gamma: float = 1.4


@dataclass(frozen=True)
class WallPoint:
    """A single point on a wall boundary."""
    x: float
    y: float
    theta: float | None = None


@dataclass(frozen=True)
class WallDefinition:
    """Wall boundary defined by a sequence of points."""
    points: tuple[WallPoint, ...]
    name: str = ""
    interpolation: str = "linear"


@dataclass(frozen=True)
class InletCondition:
    """Uniform supersonic inlet condition."""
    mach: float
    theta: float = 0.0
    p0: float = 1.0
    T0: float = 1.0


@dataclass(frozen=True)
class GeometryType:
    """Geometry type: planar or axisymmetric."""
    kind: str = "planar"


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    gas: GasProperties
    inlet: InletCondition
    walls: tuple[WallDefinition, ...]
    geometry_type: GeometryType
    n_char_lines: int = 50
