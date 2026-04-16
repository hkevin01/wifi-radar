"""
ID: WR-VIZ-HOUSE-001
Requirement: Provide an optional real-time 2-D top-down / perspective house view
             rendered with pygame, overlaying detected person positions on a
             schematic floor plan.
Purpose: Gives operators a spatial, room-level overview of where people are
         located inside the monitored area, complementing the 3-D skeleton view
         in the Dash dashboard.
Thread-safety model:
    ``update_people()`` is the only method intended to be called from the
    processing thread.  All shared state (``self._people``) is protected by
    ``self._lock`` (threading.Lock).  The render loop reads ``self._people``
    under the same lock, ensuring no partial writes are visible.
Assumptions:
    - pygame must be installed; if not, the visualiser degrades gracefully and
      all public methods become no-ops.
    - A display server (X11, Wayland, or headless framebuffer) must be available.
Failure Modes:
    Any pygame or OpenGL exception in ``_render_loop()`` is caught, logged, and
    causes the loop to exit cleanly without crashing the main process.
"""
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
        """Start the pygame render loop in a background daemon thread.

        Preconditions:
            ``self._pygame_available`` must be True (pygame installed and importable).
            Must not be called while already running.

        Side Effects:
            Sets ``self._running = True``.
            Spawns ``self._thread`` as a daemon thread targeting ``_render_loop()``.
            The daemon flag ensures the thread is killed automatically if the main
            process exits before ``stop()`` is called.

        Thread Safety:
            Only call from the main thread before starting the processing thread.
            ``_running`` is a plain bool; no lock is used because ``start()`` and
            ``stop()`` are only called from the main thread.
        """
        if not self._pygame_available:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        self.logger.info("HouseVisualizer started (%dx%d @ %d fps)", self.width, self.height, self.fps)

    def stop(self) -> None:
        """Signal the render loop to exit and wait for the thread to join.

        Side Effects:
            Sets ``self._running = False``.
            Joins ``self._thread`` with a 2-second timeout so the caller is not
            blocked indefinitely if pygame hangs on ``pygame.quit()``.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("HouseVisualizer stopped")

    def update_people(self, people: List[dict]) -> None:
        """Thread-safe replacement of the detected-people list for the next frame.

        Called by the inference/processing thread after each forward pass.
        Acquires ``self._lock`` before overwriting ``self._people`` so the
        render thread never sees a partially-updated list.

        Args:
            people: List of person dicts, each containing at minimum:
                      ``keypoints``  — (17, 3) numpy array of normalised 3-D coords.
                      ``confidence`` — (17,) numpy array of per-keypoint scores.

        Thread Safety:
            This method is the ONLY external entry point that mutates
            ``self._people``.  The render loop reads the same field under the
            same lock, guaranteeing mutual exclusion.
        """
        with self._lock:
            self._people = list(people)

    # ------------------------------------------------------------------
    # Internal rendering loop (placeholder — extend with real pygame/OpenGL)
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        """Pygame event-and-render loop running in the background daemon thread.

        Loop structure (each iteration at ``self.fps``):
            1. Drain the pygame event queue; exit on QUIT event.
            2. Fill the screen with the background colour.
            3. Acquire ``self._lock`` and call ``_draw_people()`` to render
               the current snapshot without holding the lock during the full
               draw cycle (only the list copy is protected).
            4. Flip the display buffer.
            5. ``clock.tick(fps)`` caps the frame rate and yields CPU time.

        Side Effects:
            Calls ``pygame.init()`` and ``pygame.quit()`` (in finally block)
            inside the thread.  Any exception is caught and logged; pygame is
            always quit to release the display resource.

        Note:
            Replace the body of this method with a full room-geometry render
            (floor plan walls, router antenna positions, etc.) once the spatial
            layout model is available.
        """
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
        """Render each detected person as a circle at their projected 2-D position.

        Algorithm:
            For each person in ``self._people``:
            1. Skip entries without ``keypoints`` or ``confidence``.
            2. Select keypoints whose confidence exceeds 0.3 (valid mask).
            3. Compute the mean X and Y of valid keypoints as the person's
               centre of mass in normalised space [-1, 1].
            4. Map the normalised centre of mass to screen pixels:
               ``pixel = (norm + 1) / 2 \u00d7 dimension``.
            5. Draw a filled circle at the screen position.

        Thread Safety:
            Must be called from inside the ``with self._lock`` block in
            ``_render_loop()`` to ensure a consistent view of ``self._people``.

        Args:
            screen: Active ``pygame.Surface`` (the display window).
        """
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
