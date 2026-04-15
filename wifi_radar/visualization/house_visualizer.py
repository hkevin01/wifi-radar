"""House visualizer — optional OpenGL/pygame 3-D view of detected poses."""
from __future__ import annotations

import logging
import threading
from typing import List


class HouseVisualizer:
    """Renders a top-down or perspective 3-D house view with pose overlays.

    Requires ``pygame`` and an OpenGL-capable display.  When a display is
    unavailable the class degrades gracefully and logs a warning instead of
    raising an exception.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        fps: int = 30,
        wall_transparency: float = 0.5,
    ) -> None:
        self.logger = logging.getLogger("HouseVisualizer")
        self.width = width
        self.height = height
        self.fps = fps
        self.wall_transparency = wall_transparency

        self._running = False
        self._thread: threading.Thread | None = None
        self._people: List[dict] = []
        self._lock = threading.Lock()

        # Lazy-import pygame so the rest of the system works without a display.
        try:
            import pygame  # noqa: F401
            self._pygame_available = True
        except ImportError:
            self.logger.warning(
                "pygame is not installed — house visualizer disabled. "
                "Install it with:  pip install pygame"
            )
            self._pygame_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the visualizer in a background thread."""
        if not self._pygame_available:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self.logger.info("HouseVisualizer started (%dx%d @ %d fps)", self.width, self.height, self.fps)

    def stop(self) -> None:
        """Stop the visualizer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("HouseVisualizer stopped")

    def update_people(self, people: List[dict]) -> None:
        """Thread-safe update of detected people for the next render frame."""
        with self._lock:
            self._people = list(people)

    # ------------------------------------------------------------------
    # Internal rendering loop (placeholder — extend with real pygame/OpenGL)
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        """Main rendering loop.  Replace with a real pygame render loop."""
        import time
        import pygame

        try:
            pygame.init()
            screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("WiFi-Radar — House View")
            clock = pygame.time.Clock()

            while self._running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False

                screen.fill((20, 20, 30))
                with self._lock:
                    self._draw_people(screen)

                pygame.display.flip()
                clock.tick(self.fps)

        except Exception as exc:
            self.logger.error("Render loop error: %s", exc)
        finally:
            pygame.quit()

    def _draw_people(self, screen) -> None:  # type: ignore[no-untyped-def]
        """Draw detected people onto *screen*.  Extend as needed."""
        import pygame

        for person in self._people:
            kp = person.get("keypoints")
            conf = person.get("confidence")
            if kp is None or conf is None:
                continue
            # Centre of mass — average of high-confidence keypoints
            valid = conf > 0.3
            if not valid.any():
                continue
            cx = float(kp[valid, 0].mean())
            cy = float(kp[valid, 1].mean())
            sx = int((cx + 1) / 2 * self.width)
            sy = int((cy + 1) / 2 * self.height)
            pygame.draw.circle(screen, (0, 200, 255), (sx, sy), 12)
